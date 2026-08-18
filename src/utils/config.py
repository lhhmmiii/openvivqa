import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    return config
