import json
import pandas as pd
from pathlib import Path

def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".json", ".geojson"]:
        return pd.read_json(path)
    raise ValueError(f"Unsupported format: {path.suffix}")
