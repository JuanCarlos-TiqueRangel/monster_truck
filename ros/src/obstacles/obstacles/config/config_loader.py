from pathlib import Path
from types import SimpleNamespace
import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _to_namespace(data)

cfg_params = load_config()