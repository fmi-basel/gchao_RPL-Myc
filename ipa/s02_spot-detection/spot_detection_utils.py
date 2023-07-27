import logging
from skimage.measure import regionprops
from skimage.morphology import h_maxima, ball
from scipy.optimize import curve_fit

from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_laplace
from skimage.util import img_as_float32, img_as_uint
import pandas as pd

import numpy as np


def gauss_3d(bg, amp, mu_x, mu_y, mu_z, sig_x, sig_y, sig_z):
    def fun(coords):
        return amp * np.exp(
            -0.5 * ( 
            ( (coords[:, 2] - mu_x)**2 )/( sig_x**2 ) + 
            ( (coords[:, 1] - mu_y)**2 )/( sig_y**2 ) + 
            ( (coords[:, 0] - mu_z)**2 )/( sig_z**2 )
            )
        ) + bg
    
    return fun


def eval_gauss_3d(x, bg, amp, mu_x, mu_y, mu_z, sig_x, sig_y, sig_z):
    return gauss_3d(bg=bg, amp=amp, mu_x=mu_x, mu_y=mu_y, mu_z=mu_z, sig_x=sig_x, sig_y=sig_y, sig_z=sig_z)(x)


def subpixel_localization_3d(spot_img, spacing):
    init_params = [
        spot_img.min(), 
        spot_img.max(), 
        spot_img.shape[2]/2 * spacing[2], 
        spot_img.shape[1]/2 * spacing[1], 
        spot_img.shape[0]/2 * spacing[0],
        1,
        1,
        1
    ]

    zz = np.arange(spot_img.shape[0]) * spacing[0]
    yy = np.arange(spot_img.shape[1]) * spacing[1] 
    xx = np.arange(spot_img.shape[2]) * spacing[2]
    z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
    coords_zyx = np.stack([z.ravel(), y.ravel(), x.ravel()], -1)

    bounds = [
            (0, init_params[1] * 0.5, init_params[2] - 3 * spacing[2], init_params[3] - 3 * spacing[1], 0 * spacing[0], -3, -3, -6),
            (init_params[0] * 2, init_params[1] * 2, init_params[2] + 3 * spacing[2], init_params[3] + 3 * spacing[1], (spot_img.shape[0] - 1) * spacing[0], 3, 3, 6),
        ]
    
    popt, pcov = curve_fit(
        eval_gauss_3d,
        coords_zyx,
        spot_img.ravel(),
        p0=init_params,
        bounds=bounds
    )

    return popt


def crop_cells(img, cyto_seg, nuc_seg):
    cells = {}

    cell_features = regionprops(cyto_seg)

    for cf in cell_features:
        bbox = cf.bbox
        start_y = max(0, bbox[0] - 10)
        start_x = max(0, bbox[1] - 10)
        end_y = min(img.shape[1], bbox[2] + 10)
        end_x = min(img.shape[2], bbox[3] + 10)

        cells[cf.label] = {
            "raw": img[:, start_y:end_y, start_x:end_x],
            "cyto_mask": np.pad(cf.image, 10),
            "nuc_mask": nuc_seg[:, start_y:end_y, start_x:end_x] > 0,
            "offset_yx": (start_y, start_x)
        }

    return cells


def localize_spots(img, spot_coords, spacing, logger=logging.Logger("localize-spots")):
    subpix_spots = {
        "background": [],
        "amplitude": [],
        "centroid_x": [],
        "centroid_y": [],
        "centroid_z": [],
        "sigma_x": [],
        "sigma_y": [],
        "sigma_z": [],
    }
    
    for coords in spot_coords:
        start_z = int(max(0, coords[0] - 2))
        start_y = int(max(0, coords[1] - 4))
        start_x = int(max(0, coords[2] - 4))
        end_z = int(min(img.shape[0], coords[0] + 3))
        end_y = int(min(img.shape[1], coords[1] + 5))
        end_x = int(min(img.shape[2], coords[2] + 5))
        spot_img = img[start_z:end_z, start_y:end_y, start_x:end_x]
        try:
            bg, amp, mu_x, mu_y, mu_z, sig_x, sig_y, sig_z = subpixel_localization_3d(spot_img, spacing=spacing)
            mu_x = mu_x + start_x * spacing[2]
            mu_y = mu_y + start_y * spacing[1]
            mu_z = mu_z + start_z * spacing[0]

            subpix_spots["background"].append(bg)
            subpix_spots["amplitude"].append(amp)
            subpix_spots["centroid_x"].append(mu_x)
            subpix_spots["centroid_y"].append(mu_y)
            subpix_spots["centroid_z"].append(mu_z)
            subpix_spots["sigma_x"].append(abs(sig_x))
            subpix_spots["sigma_y"].append(abs(sig_y))
            subpix_spots["sigma_z"].append(abs(sig_z))
        except RuntimeError as e:
            logger.warning(f"No sub-pixel localization found. Discarding spot at {coords}.")

    return pd.DataFrame(subpix_spots)


def classify_spots(subpix_spots, cyto_mask, nuc_mask, spacing):
    def filter_fun(row):
        z = min(max(0, int(round(row["centroid_z"] / spacing[0]))), nuc_mask.shape[0] - 1)
        y = min(max(0, int(round(row["centroid_y"] / spacing[1]))), nuc_mask.shape[1] - 1)
        x = min(max(0, int(round(row["centroid_x"] / spacing[2]))), nuc_mask.shape[2] - 1)
        
        in_nuc = nuc_mask[z, y, x] > 0
        if not in_nuc:
            in_cyto = cyto_mask[y, x] > 0
        else:
            in_cyto = False

        row["in_nuclei"] = in_nuc
        row["in_cyto"] = in_cyto
        return row
        
    subpix_spots = subpix_spots.apply(filter_fun, axis=1)
    return subpix_spots
        

def get_symmetric_spot_filter(threshold):
    def fun(row):
        if row["sigma_x"] > row["sigma_y"]:
            return (row["sigma_y"]/row["sigma_x"]) > threshold
        else:
            return (row["sigma_x"]/row["sigma_y"]) > threshold
    
    return fun


def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.flatten(),target.flatten())
    alpha = cov[0,1] / (cov[0,0]+1e-10)
    beta = target.mean() - alpha*x.mean()
    return alpha*x + beta


def detect_spots(cells: dict[dict], h: float, wavelength: int, NA: float, spacing: tuple[float, float, float], logger=logging.Logger("localize-spots")):

    spots = []
    for cell_id, cell in cells.items():
        logger.info(f"Processing cell with ID = {cell_id} and YX-offsets = {cell['offset_yx']}.")
        sigma = wavelength/ ( 2 * NA ) / np.sqrt(2) / (spacing[1] * 1000)
        log_img = -gaussian_laplace(img_as_float32(cell['raw']), sigma=sigma) * sigma**2
        log_img = img_as_uint(normalize_minmse(log_img, img_as_float32(cell['raw'])))
        spot_coords = np.stack(np.where(h_maxima(log_img, h, ball(int(sigma))))).T
        
        subpix_spots = localize_spots(cell['raw'], spot_coords, spacing=spacing, logger=logger)
        
        if len(subpix_spots) > 0:
            subpix_spots = classify_spots(subpix_spots, cell["cyto_mask"], cell["nuc_mask"], spacing)

            final_spots = pd.DataFrame(subpix_spots.query("in_cyto | in_nuclei"))
            final_spots["symmetric"] = final_spots.apply(get_symmetric_spot_filter(0.75), axis=1)
            final_spots["centroid_y"] += cell['offset_yx'][0] * spacing[1]
            final_spots["centroid_x"] += cell['offset_yx'][1] * spacing[2]
            final_spots["cell_id"] = cell_id
            final_spots["spot_id"] = final_spots.apply(lambda r: str(r["cell_id"]) + "_" + str(r.name), axis=1)
            
            spots.append(final_spots)

    return pd.concat(spots, ignore_index=True)
        