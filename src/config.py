from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
