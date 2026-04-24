import argparse
from src.pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()
    
    run_pipeline(args.config)

if __name__ == "__main__":
    main()
