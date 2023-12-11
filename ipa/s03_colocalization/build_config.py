import os
import questionary
import yaml

def build_config():
    cwd = os.getcwd()

    spots_dir = questionary.path("Path to spot data directory:").ask()
    output_dir = questionary.path("Path to output directory:").ask()


    config = {
        "spots_dir": os.path.relpath(spots_dir, cwd),
        "output_dir": os.path.relpath(output_dir, cwd),
    }

    with open(os.path.join(cwd, "colocalization_config.yaml"), "w") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    build_config()
