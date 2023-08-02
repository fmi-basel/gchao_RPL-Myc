from aicsimageio import AICSImage
from glob import glob
import multiprocessing

from skimage.filters import threshold_triangle, median
from skimage.measure import label, regionprops_table, regionprops
from skimage.morphology import binary_opening, remove_small_holes, area_opening, disk, binary_erosion
from skimage.segmentation import clear_border, watershed

from scipy.ndimage import distance_transform_edt

import pandas as pd
import numpy as np

from copy import copy
import os
from datetime import datetime

from tifffile import imwrite

from tqdm import tqdm
import logging

import yaml

logger = logging.Logger('Instance Segmentation')
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
handler = logging.FileHandler(f"{now}-instance_segmentation.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def threshold_mad(im: np.ndarray, k=6):
    med = np.median(im)
    mad = np.median(np.abs(im.astype(np.float32) - med))
    return med + mad * k * 1.4826


def segment_nuclei_3d(img):
    ths = threshold_mad(img)
    mask = median(img, np.ones((3, 3, 3))) > ths
    mask = remove_small_holes(area_opening(binary_opening(mask, footprint=np.ones((3,3,3))), 100), 100)
    distance = distance_transform_edt(mask)
    eroded = np.max(mask, axis=0)
    for i in range(10):
        features = regionprops(label(eroded))
        for ft in features:
            if ft.area > 500:
                bb = ft.bbox
                eroded[bb[0]:bb[2], bb[1]:bb[3]] = binary_erosion(ft.image, footprint=np.ones((7,7)))

    seeds = np.zeros_like(distance, dtype=int)
    seeds[seeds.shape[0]//2] = eroded
    seeds = label(seeds)
    return watershed(-distance, seeds, mask=mask)


def clean_nuc_labeling_3d(labeling, spacing=(0.24, 0.107, 0.107)):
    features = regionprops(labeling, spacing=spacing)

    clean_labeling = copy(labeling)

    for ft in features:
        if ft.area < 200:
            clean_labeling[clean_labeling == ft.label] = 0
        elif ft.solidity < 0.5:
            clean_labeling[clean_labeling == ft.label] = 0

    return clean_labeling

def segment_cyto_2d(img, seeds):
    foreground = median(np.max(img, axis=(0, 1)), np.ones((3, 3)))
    foreground_mask = binary_opening(foreground > np.mean(foreground), disk(3))
    watershed_mask = remove_small_holes(area_opening(binary_opening(foreground > np.median(foreground), np.ones((7,7))), 1000), 100)
    distance = distance_transform_edt(foreground_mask)
    return watershed(-distance, seeds, mask=watershed_mask)

def clean_nuc_cyto_labelings(nuc_labeling_3d, nuc_labeling_2d, cyto_labeling_2d):
    clean_cyto_labeling_2d = np.zeros_like(cyto_labeling_2d)

    for label_id in filter(None, np.unique(nuc_labeling_2d)):
        clean_cyto_labeling_2d[cyto_labeling_2d == label_id] = label_id

    clean_nuc_labeling_2d_ = np.zeros_like(clean_cyto_labeling_2d)
    clean_nuc_labeling = np.zeros_like(nuc_labeling_3d)

    for label_id in filter(None, np.unique(clean_cyto_labeling_2d)):
        clean_nuc_labeling_2d_[nuc_labeling_2d == label_id] = label_id
        clean_nuc_labeling[nuc_labeling_3d == label_id] = label_id

    clean_nuc_labeling_2d = copy(clean_nuc_labeling_2d_)

    return clean_nuc_labeling, clean_nuc_labeling_2d, clean_cyto_labeling_2d

def segment_nuclei_and_cyto(file: str, output_dir: str):
    logger.info(f'Load image from: {file}')
    img = AICSImage(file)

    logger.info('Segment nuclei in 3D.')
    nuc_labeling = segment_nuclei_3d(img.data[0, 0])

    logger.info('Clean 3D nuclei labels.')
    clean_nuc_labeling = clean_nuc_labeling_3d(nuc_labeling)

    logger.info('Clear border.')
    clean_labeling_2d = clear_border(np.max(clean_nuc_labeling, axis=0))

    logger.info('Segment cytoplasma in 2D.')
    cyto_labeling_2d = segment_cyto_2d(img.data[0, 1:], np.max(nuc_labeling, axis=0))

    logger.info('Clean up labelings.')
    clean_nuc_labeling, clean_nuc_labeling_2d, clean_cyto_labeling_2d = clean_nuc_cyto_labelings(
        clean_nuc_labeling,
        clean_labeling_2d,
        cyto_labeling_2d,
    )

    name, _ = os.path.splitext(os.path.basename(file))
    logger.info(f'Save labeling to: {output_dir}/{name}...')
    imwrite(
        os.path.join(output_dir, f'{name}_NUC-SEG-3D.tif'),
        clean_nuc_labeling,
        compression='zlib',
    )
    imwrite(
        os.path.join(output_dir, f'{name}_NUC-SEG-2D.tif'),
        clean_nuc_labeling_2d,
        compression='zlib',
    )
    imwrite(
        os.path.join(output_dir, f'{name}_CYTO-SEG-2D.tif'),
        clean_cyto_labeling_2d,
        compression='zlib',
    )


if __name__ == "__main__":
    with open("nuclei_cyto_segmentation_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    files = glob(os.path.join(config['raw_data_dir'], '*.nd'))

    logger.info(f'Found {len(files)} files for processing.')

    pool = multiprocessing.Pool(8)
    progress = tqdm(total=len(files), smoothing=0)
    for file in files:
        pool.apply_async(
            segment_nuclei_and_cyto,
            kwds={
                "file": file,
                "output_dir": config['output_dir'],
            },
            callback=lambda _: progress.update()
        )
        # segment_nuclei_and_cyto(
        #     file=file,
        #     output_dir=config['output_dir']
        # )
        # progress.update()

    pool.close()
    pool.join()

    logger.info('Done!')