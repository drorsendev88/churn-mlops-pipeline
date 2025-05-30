import pytest

from config.utils import load_config


def test_load_config_valid():
    config = load_config("config.yaml")
    assert isinstance(config, dict)
    # Basala nycklar enligt din config.yaml
    assert "experiment_name" in config
    assert "model_name" in config


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_load_config_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("this: is: invalid: yaml: :::")
    with pytest.raises(RuntimeError):
        load_config(str(bad))
