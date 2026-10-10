from pathlib import Path

import yaml

def load_config(config_path=None):
    if config_path is None: config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config
