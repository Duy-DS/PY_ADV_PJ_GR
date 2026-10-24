from src.utils.config import load_config


def test_load_config():
    config = load_config()

    assert "eye" in config
    assert "ear_threshold" in config["eye"]
    assert config["eye"]["ear_threshold"] == 0.25
