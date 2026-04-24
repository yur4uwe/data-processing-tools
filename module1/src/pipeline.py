import json
import platform
from typing import cast
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import Config
from .logger import setup_logger
from .io import ensure_dirs, load_csv, save_csv
from .preprocess import build_preprocessor
from .train import build_regression
from .evaluate import evaluate_regression


def run_pipeline(config_path: str):
    cfg = Config.load(config_path)

    seed = cfg.project.seed
    np.random.seed(cfg.project.seed)

    run_id = cfg.project.run_id
    if cfg.project.run_id_mode == "timestamp":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(cfg.output.artifacts_dir) / f"run_{run_id}"
    ensure_dirs(
        str(run_dir / "models"),
        str(run_dir / "predictions"),
        str(run_dir / "metrics"),
        str(run_dir / "reports"),
    )

    logger, log_file = setup_logger(cfg.output.logs_dir, run_id)
    logger.info(f"Run ID: {run_id}")
    logger.info(
        f"Python: {platform.python_version()} | Platform: {platform.platform()}"
    )
    logger.info(f"Config: {config_path} | Log: {log_file}")

    input_csv = cfg.data.input_csv
    target = cfg.data.target
    num_cols = cfg.data.num_cols
    cat_cols = cfg.data.cat_cols
    test_size = cfg.data.test_size

    # Extract params dict from dataclass
    params = cfg.model.params.to_dict()

    # Load
    df = load_csv(input_csv)
    logger.info(f"Loaded data: {df.shape} from {input_csv}")

    # Variant 12: Memory usage logging
    mem_usage = df.memory_usage(deep=True).sum() / (1024**2)
    logger.info(f"DataFrame memory usage: {mem_usage:.2f} MB")

    df = df.drop_duplicates().copy()

    # Validate columns
    needed = set([target] + num_cols + cat_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[num_cols + cat_cols].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train = cast(pd.DataFrame, X_train)
    X_test = cast(pd.DataFrame, X_test)
    y_train = cast(pd.Series, y_train)
    y_test = cast(pd.Series, y_test)

    pre = build_preprocessor(num_cols, cat_cols)
    model = build_regression(params)

    pipe = Pipeline([("preprocess", pre), ("model", model)])

    # Train
    pipe.fit(X_train, y_train)
    logger.info("Model trained.")

    # Predict
    y_pred = pipe.predict(X_test)

    metrics = evaluate_regression(y_test, y_pred)
    logger.info(f"Metrics: {metrics}")

    # Save model
    if cfg.output.save_model:
        model_path = run_dir / "models" / "model.joblib"
        joblib.dump(pipe, model_path)

    # Save metrics
    metrics_path = run_dir / "metrics" / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save predictions
    if cfg.output.save_predictions:
        pred_df = X_test.copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = y_pred
        save_csv(pred_df, str(run_dir / "predictions" / "predictions.csv"))

    # Save report
    if cfg.output.save_report:
        report = []
        report.append(f"# Report: run_{run_id}")
        report.append(f"- Input: {input_csv}")
        report.append(f"- Target: {target}")
        report.append(f"- Num cols: {num_cols}")
        report.append(f"- Cat cols: {cat_cols}")
        report.append(f"- Params: {params}")
        report.append(f"- Metrics: {metrics}")
        report.append(f"- Memory usage: {mem_usage:.2f} MB")
        report.append("")
        report.append("## Data quality, assumptions, and next steps.")

        report_path = run_dir / "reports" / "report.md"
        report_path.write_text("\n".join(report), encoding="utf-8")

        # Save config snapshot
        (run_dir / "reports" / "config_snapshot.yaml").write_text(
            Path(config_path).read_text(encoding="utf-8"), encoding="utf-8"
        )

    logger.info(f"Artifacts saved to: {run_dir}")
