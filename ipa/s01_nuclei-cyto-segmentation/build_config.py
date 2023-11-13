import os
import questionary
import yaml

def build_config():
    cwd = os.getcwd()

    raw_data_dir = questionary.path("Path to raw data directory:").ask()
    output_dir = questionary.path("Path to output directory:").ask()
    spacing_str = questionary.text(
        "Spacing (z, y, x):",
        default="0.2, 0.103, 0.103",
        validate=lambda v: v.replace(" ", "").replace(",", "").replace(".",
                                                                       "").isdigit()
    ).ask()
    spacing = tuple(float(v) for v in spacing_str.split(","))

    output_dir = os.path.join(output_dir, "01_nuclei-cyto-segmentation")

    config = {
        "raw_data_dir": os.path.relpath(raw_data_dir, cwd),
        "output_dir": os.path.relpath(output_dir, cwd),
        "spacing": spacing,
    }

    os.makedirs(output_dir)

    with open(os.path.join(cwd, "nuclei_cyto_segmentation_config.yaml"), "w") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    build_config()