import argparse
import json
from pathlib import Path

from src.config import ensure_dirs, REP_DIR
from src.io_utils import load_any, save_json
from src.collection import collect_from_url
from src.cleaning import clean_dataframe
from src.plotting import visualize
from src.reporting import make_report


def main():
    ensure_dirs()

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("collect")
    c1.add_argument(
        "--url",
        default="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv",
    )
    c1.add_argument("--out", default="salary_survey_raw.csv")

    c2 = sub.add_parser("clean")
    c2.add_argument("--infile", default="data/raw/salary_survey_raw.csv")
    c2.add_argument("--outfile", default="data/processed/clean.csv")

    c3 = sub.add_parser("viz")
    c3.add_argument("--infile", default="data/processed/clean.csv")

    c4 = sub.add_parser("report")
    c4.add_argument("--raw", default="data/raw/salary_survey_raw.csv")
    c4.add_argument("--clean", default="data/processed/clean.csv")
    c4.add_argument("--cleaninfo", default="artifacts/reports/clean_info.json")

    run_parser = sub.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Skip the collection stage and use existing data",
    )

    args = ap.parse_args()

    # Determine if we should skip collection
    skip_collect = getattr(args, "no_collect", False)

    if args.cmd == "collect" or (args.cmd == "run" and not skip_collect):
        # For 'run', we use the defaults from the 'collect' logic or args
        url = getattr(
            args,
            "url",
            "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv",
        )
        out = getattr(args, "out", "salary_survey_raw.csv")
        collect_from_url(url, out)

    if args.cmd == "clean" or args.cmd == "run":
        infile = getattr(args, "infile", "data/raw/salary_survey_raw.csv")
        outfile = getattr(args, "outfile", "data/processed/clean.csv")
        df = load_any(Path(infile))
        df_clean, info = clean_dataframe(df)
        df_clean.to_csv(outfile, index=False)
        save_json(info, REP_DIR / "clean_info.json")
        print(f"Saved: {outfile}")

    if args.cmd == "viz" or args.cmd == "run":
        infile = getattr(args, "infile", "data/processed/clean.csv")
        df = load_any(Path(infile))
        figs = visualize(df)
        save_json(figs, REP_DIR / "fig_paths.json")
        print("Saved figures and metadata.")

    if args.cmd == "report" or args.cmd == "run":
        raw = getattr(args, "raw", "data/raw/salary_survey_raw.csv")
        clean = getattr(args, "clean", "data/processed/clean.csv")
        cleaninfo = getattr(args, "cleaninfo", "artifacts/reports/clean_info.json")

        info_data = json.loads(Path(cleaninfo).read_text(encoding="utf-8"))
        fig_paths = json.loads((REP_DIR / "fig_paths.json").read_text(encoding="utf-8"))
        make_report(Path(raw), Path(clean), info_data, fig_paths)


if __name__ == "__main__":
    main()
