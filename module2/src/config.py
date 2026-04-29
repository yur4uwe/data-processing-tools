from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
FIG_DIR = Path("artifacts/figures")
REP_DIR = Path("artifacts/reports")

def ensure_dirs():
    for p in [RAW_DIR, PROC_DIR, FIG_DIR, REP_DIR]:
        p.mkdir(parents=True, exist_ok=True)
