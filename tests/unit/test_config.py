from pathlib import Path


def test_load_config_includes_musetalk_runtime_settings():
    from avatarpipeline import ROOT
    from avatarpipeline.core.config import load_config

    cfg = load_config(ROOT / "configs" / "settings.yaml")

    assert cfg.musetalk_dir == Path("~/MuseTalk").expanduser()
    assert cfg.musetalk_fps == 25
    assert cfg.musetalk_default_bbox_shift == 0
    assert cfg.musetalk_default_batch_size == 8
    assert cfg.default_voice == "af_heart"
