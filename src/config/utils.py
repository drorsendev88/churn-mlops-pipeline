from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """
    Läser in en YAML-konfigurationsfil relativt projektroten.

    Args:
        path: Sökväg till config-filen (t.ex. "config.yaml"),
        relativt projektroten.
    Returns:
        Konfigurationen som en ordbok.
    Raises:
        FileNotFoundError: Om filen inte finns.
        RuntimeError: Om YAML-parsning misslyckas.
    """
    # Gå upp tre nivåer från src/config/utils.py till projektroten
    base_path = Path(__file__).resolve().parents[2]
    config_path = base_path / path

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")
