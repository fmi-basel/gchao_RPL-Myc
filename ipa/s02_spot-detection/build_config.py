import os
import questionary
import yaml

def build_config():
    cwd = os.getcwd()

    raw_data_dir = questionary.path("Path to raw data directory:").ask()
    nuc_cyto_seg_dir = questionary.path("Path to nuc & cyto segmentation directory:").ask()
    h_01 = float(questionary.text(
        "Spot height relativ to background C01:",
        default="357",
        validate=lambda v: v.replace(".", "").isdigit()
        ).ask())
    h_02 = float(questionary.text(
        "Spot height relativ to background C02:",
        default="681",
        validate=lambda v: v.replace(".", "").isdigit()
        ).ask())
    wl_01 = int(questionary.text(
        "Emission wavelength C01:",
        default="640",
        validate=lambda v: v.isdigit()
        ).ask())
    wl_02 = int(questionary.text(
        "Emission wavelength C02:",
        default="561",
        validate=lambda v: v.isdigit()
        ).ask())
    NA = float(questionary.text(
        "NA:",
        default="1.4",
        validate=lambda v: v.replace(".", "").isdigit()
        ).ask())
    output_dir = questionary.path("Path to output directory:").ask()

    output_dir = os.path.join(output_dir, "03_spot-detection")

    config = {
        "raw_data_dir": os.path.relpath(raw_data_dir, cwd),
        "nuc_cyto_seg_dir": os.path.relpath(nuc_cyto_seg_dir, cwd),
        "h_01": h_01,
        "h_02": h_02,
        "wl_01": wl_01,
        "wl_02": wl_02,
        "NA": NA,
        "output_dir": os.path.relpath(output_dir, cwd),
    }

    os.makedirs(output_dir)

    with open(os.path.join(cwd, "spot_detection_config.yaml"), "w") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    build_config()
