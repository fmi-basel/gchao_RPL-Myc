import os
import questionary
import yaml

def build_config():
    cwd = os.getcwd()

    raw_data_dir = questionary.path("Path to raw data directory:").ask()
    nuc_cyto_seg_dir = questionary.path("Path to nuc & cyto segmentation directory:").ask()
    channel_index_spots_1 = int(questionary.text(
        "Index of first spot channel:",
        default="1",
        validate=lambda v: v.isdigit()
        ).ask())
    channel_index_spots_2 = int(questionary.text(
        "Index of second spot channel:",
        default="2",
        validate=lambda v: v.isdigit()
        ).ask())
    h_01 = float(questionary.text(
        "Spot height relativ to background C02:",
        default="357",
        validate=lambda v: v.replace(".", "").isdigit()
        ).ask())
    h_02 = float(questionary.text(
        "Spot height relativ to background C03:",
        default="681",
        validate=lambda v: v.replace(".", "").isdigit()
        ).ask())
    wl_01 = int(questionary.text(
        "Emission wavelength C02:",
        default="640",
        validate=lambda v: v.isdigit()
        ).ask())
    wl_02 = int(questionary.text(
        "Emission wavelength C03:",
        default="561",
        validate=lambda v: v.isdigit()
        ).ask())
    NA = float(questionary.text(
        "NA:",
        default="1.4",
        validate=lambda v: v.replace(".", "").isdigit()
        ).ask())
    spacing_str = questionary.text(
        "Spacing (z, y, x):",
        default="0.2, 0.103, 0.103",
        validate=lambda v: v.replace(" ", "").replace(",", "").replace(".",
                                                                       "").isdigit()
    ).ask()
    spacing = tuple(float(v) for v in spacing_str.split(","))

    output_dir = questionary.path("Path to output directory:").ask()

    output_dir = os.path.join(output_dir, "02_spot-detection")

    config = {
        "raw_data_dir": os.path.relpath(raw_data_dir, cwd),
        "nuc_cyto_seg_dir": os.path.relpath(nuc_cyto_seg_dir, cwd),
        "channel_index_spots_1": channel_index_spots_1,
        "channel_index_spots_2": channel_index_spots_2,
        "h_01": h_01,
        "h_02": h_02,
        "wl_01": wl_01,
        "wl_02": wl_02,
        "NA": NA,
        "output_dir": os.path.relpath(output_dir, cwd),
        "spacing": spacing,
    }

    os.makedirs(output_dir)

    with open(os.path.join(cwd, "spot_detection_config.yaml"), "w") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    build_config()
