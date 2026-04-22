import json
from typing import cast
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

VARIANT = 12
INPUT_FILE = "variant12.csv"
TASK = "regression"
TARGET = "engagement"

NUM_COLS = ["followers", "posts", "likes_avg"]
CAT_COLS = ["niche", "country"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = "linreg"


def build_preprocess(num_cols, cat_cols):
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_model():
    return LinearRegression()


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def quality_summary(df: pd.DataFrame) -> dict:
    miss = df.isna().sum()
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_total": int(miss.sum()),
        "missing_by_col": {k: int(v) for k, v in miss.to_dict().items()},
        "duplicates": int(df.duplicated().sum()),
    }


def generate_data(path: str, n_rows: int = 500):
    np.random.seed(RANDOM_STATE)
    data = {
        "followers": np.random.randint(100, 1000000, n_rows),
        "posts": np.random.randint(10, 5000, n_rows),
        "likes_avg": np.random.randint(5, 50000, n_rows),
        "niche": np.random.choice(["fashion", "tech", "food", "travel", "fitness"], n_rows),
        "country": np.random.choice(["USA", "UK", "Ukraine", "Germany", "France"], n_rows),
    }
    # Synthetic target logic
    data["engagement"] = (
        data["followers"] * 0.05 + 
        data["likes_avg"] * 0.8 + 
        data["posts"] * 0.1 + 
        np.random.normal(0, 100, n_rows)
    )
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Generated synthetic data: {path}")


def main():
    if not Path(INPUT_FILE).exists():
        generate_data(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    # Check which columns are actually present
    available_num = [c for c in NUM_COLS if c in df.columns]
    available_cat = [c for c in CAT_COLS if c in df.columns]

    print(f"Using numeric columns: {available_num}")
    print(f"Using categorical columns: {available_cat}")

    # Min cleaning
    df = df.drop_duplicates().copy()

    X = df[available_num + available_cat].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train = cast(pd.DataFrame, X_train)
    X_test = cast(pd.DataFrame, X_test)
    y_train = cast(pd.Series, y_train)
    y_test = cast(pd.Series, y_test)

    pre = build_preprocess(available_num, available_cat)
    model = build_model()

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )

    print("Training model...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    # Save artifacts
    out_model = f"model_variant{VARIANT:02d}.joblib"
    joblib.dump(pipe, out_model)

    out_metrics = f"metrics_variant{VARIANT:02d}.json"
    Path(out_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Predictions
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(f"predictions_variant{VARIANT:02d}.csv", index=False)

    # Report
    q = quality_summary(df)
    lines = [
        f"ЛР №10 — Повний data-pipeline (Варіант {VARIANT:02d})",
        f"Input: {INPUT_FILE}",
        f"Task: {TASK}",
        f"Target: {TARGET}",
        f"Features: {available_num + available_cat}",
        "",
        "== Data Quality ==",
        json.dumps(q, indent=2),
        "",
        "== Metrics ==",
        json.dumps(metrics, indent=2),
        "",
        "== Висновки ==",
        "1. Дані успішно завантажені та перевірені на якість. Пропуски та дублікати оброблені у пайплайні.",
        "2. Побудовано Pipeline з використанням StandardScaler для числових ознак та OneHotEncoder для категоріальних.",
        "3. Модель LinearRegression навчена для прогнозування engagement. Метрики показують якість моделі на тестовій вибірці.",
    ]

    Path(f"report_variant{VARIANT:02d}.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Saved artifacts and report_variant{VARIANT:02d}.txt")


if __name__ == "__main__":
    main()
