import logging
from datetime import datetime
import os
from glob import glob
from aicsimageio.aics_image import AICSImage
import multiprocessing
from tifffile import imread
from tqdm import tqdm

import yaml

from spot_detection_utils import crop_cells, detect_spots


logger = logging.Logger('Spot Detection')
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
handler = logging.FileHandler(f"{now}-spot_detection.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run_spot_detection(file: str, nuc_cyto_seg_dir: str,
                       channel_index_spots_1: int, channel_index_spots_2: int,
                       h_01: float, h_02: float, wl_01: int, wl_02: int,
                       NA: float, spacing: tuple[float, float, float],
                       output_dir: str):
    name, _ = os.path.splitext(os.path.basename(file))
    nuc_seg_file = os.path.join(nuc_cyto_seg_dir, name + "_NUC-SEG-3D.tif")
    cyto_seg_file = os.path.join(nuc_cyto_seg_dir, name + "_CYTO-SEG-2D.tif")

    logger.info(f"Loading image data from: {file}")
    img = AICSImage(file)
    raw_01 = img.data[0, channel_index_spots_1]
    raw_02 = img.data[0, channel_index_spots_2]

    logger.info(f"Loading nuclei segmentation from: {nuc_seg_file}")
    nuc_seg = imread(nuc_seg_file)
    logger.info(f"Loading cyto segmentation from: {cyto_seg_file}")
    cyto_seg = imread(cyto_seg_file)

    logger.warning(f"Hard coded image spacing [Z, Y, X]: {spacing}")

    cells_01 = crop_cells(raw_01, cyto_seg, nuc_seg)
    cells_02 = crop_cells(raw_02, cyto_seg, nuc_seg)

    spots_01 = detect_spots(cells_01, h=h_01, wavelength=wl_01, NA=NA, spacing=spacing, logger=logger)
    spots_02 = detect_spots(cells_02, h=h_02, wavelength=wl_02, NA=NA, spacing=spacing, logger=logger)

    spots_01.to_csv(os.path.join(output_dir, name + "_SPOTS_C01.csv"))
    spots_02.to_csv(os.path.join(output_dir, name + "_SPOTS_C02.csv"))


if __name__ == "__main__":
    with open("spot_detection_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    files = glob(os.path.join(config['raw_data_dir'], '*.nd'))

    logger.info(f"Found {len(files)} files for processing.")

    pool = multiprocessing.Pool(8)
    progress = tqdm(total=len(files), smoothing=0)
    for file in files:
        pool.apply_async(
            run_spot_detection,
            kwds={
                "file":file,
                "nuc_cyto_seg_dir": config['nuc_cyto_seg_dir'],
                "channel_index_spots_1": config['channel_index_spots_1'],
                "channel_index_spots_2": config['channel_index_spots_2'],
                "h_01": config['h_01'],
                "h_02": config['h_02'],
                "wl_01": config['wl_01'],
                "wl_02": config['wl_02'],
                "NA": config['NA'],
                "spacing": config['spacing'],
                "output_dir": config['output_dir'],
            },
            callback=lambda _: progress.update()
        )

    pool.close()
    pool.join()

    logger.info("Done!")
