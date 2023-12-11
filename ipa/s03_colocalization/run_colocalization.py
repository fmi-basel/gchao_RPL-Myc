from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import yaml
from scipy.spatial import KDTree
import logging
from datetime import datetime

logger = logging.Logger('Colocalization')
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
handler = logging.FileHandler(f"{now}-colocalization.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def identify_closest_spot(src_spots, query_spots, col_name):
    src_spots[col_name] = None
    src_spots[f"distance_to_{col_name}"] = None
    for cell_id in query_spots["cell_id"].unique():
        cell_spots = query_spots.query(f"cell_id == {cell_id}")
        tree = KDTree(
            cell_spots[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(

            ).astype(float))

        def get_coloc(row):
            if row["cell_id"] == cell_id:
                dist, idx = tree.query(
                    row[["centroid_x", "centroid_y",
                         "centroid_z"]].to_numpy().astype(float))
                row[col_name] = cell_spots.iloc[idx]["spot_id"]
                row[f"distance_to_{col_name}"] = dist
            return row

        src_spots = src_spots.apply(get_coloc, axis=1)

    return src_spots


def identify_coloc_pairs(spots_01, spots_02):
    for i, row in spots_01.iterrows():
        nearest_spot_id = row["nearest_c02_spot"]
        for j, r in spots_02.query(
            f"spot_id == '{nearest_spot_id}'").iterrows():
            coloc_pair = r["nearest_c01_spot"] == row["spot_id"]
            spots_01.at[i, "coloc_pair"] = coloc_pair
            spots_02.at[j, "coloc_pair"] = coloc_pair

    return spots_01, spots_02


def get_coloc_summary(c01_file, c02_file):
    logger.info(f"Run colocalization for {c01_file} and {c02_file}.")
    spots_01 = pd.read_csv(c01_file, index_col=0)
    spots_02 = pd.read_csv(c02_file, index_col=0)

    spots_01 = identify_closest_spot(spots_01, spots_02, "nearest_c02_spot")
    spots_02 = identify_closest_spot(spots_02, spots_01, "nearest_c01_spot")

    spots_01, spots_02 = identify_coloc_pairs(spots_01, spots_02)

    spots_01['file_name'] = os.path.basename(c01_file).replace(
        "_SPOTS_C01.csv", "")
    spots_02['file_name'] = os.path.basename(c02_file).replace(
        "_SPOTS_C02.csv", "")

    return spots_01, spots_02


if __name__ == "__main__":
    with open("colocalization_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    c01_files = glob(os.path.join(config['spots_dir'], '*_SPOTS_C01.csv'))

    spots_01, spots_02 = [], []
    for c01_file in tqdm(c01_files):
        s01, s02 = get_coloc_summary(c01_file,
                                     c01_file.replace("_C01.csv", "_C02.csv"))

        spots_01.append(s01)
        spots_02.append(s02)

    summary_01 = pd.concat(spots_01, ignore_index=True)
    summary_02 = pd.concat(spots_02, ignore_index=True)

    summary_01.to_csv(os.path.join(config['output_dir'], "summary_c01.csv"))
    summary_02.to_csv(os.path.join(config['output_dir'], "summary_c02.csv"))

    logger.info("Done!")

