import requests
from .config import RAW_DIR

def collect_from_url(url: str, out_name: str):
    print(f"Downloading from {url}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path = RAW_DIR / out_name
    out_path.write_bytes(r.content)
    print(f"Saved: {out_path}")
