import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

VARIANT = 12
OUT_RAW = f"raw_variant{VARIANT:02d}.json"
OUT_CSV = f"clean_variant{VARIANT:02d}.csv"
OUT_REP = f"report_variant{VARIANT:02d}.txt"
TIMEOUT = 20


# ===================== HTTP =====================
def get_json(url: str, params: dict | None = None):
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json(), r.url


def save_raw(obj, path: str):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ===================== FETCH & NORMALIZE =====================
def fetch_breweries():
    # 1. Get list of breweries
    url = "https://api.openbrewerydb.org/v1/breweries"
    params = {"per_page": 50}
    data, final_url = get_json(url, params=params)

    # 2. Normalize
    df = pd.json_normalize(data)

    # Keep recommended fields
    keep = [
        "id",
        "name",
        "brewery_type",
        "city",
        "state_province",
        "country",
        "longitude",
        "latitude",
        "website_url",
    ]
    df = pd.DataFrame(df[[c for c in keep if c in df.columns]].copy())

    meta = {
        "source": "Open Brewery DB",
        "request_url": final_url,
        "rows": len(df),
        "columns": list(df.columns),
    }
    return data, df, meta


def minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.drop_duplicates().copy()

    # Fill missing values for important categorical columns
    cat_cols = ["brewery_type", "city", "state_province"]
    for col in cat_cols:
        if col in df2.columns:
            df2[col] = df2[col].fillna("Unknown")

    # Try to convert coordinates to numeric
    for col in ["latitude", "longitude"]:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

    return df2


def make_report(meta: dict, df: pd.DataFrame) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        f"ЛР №9 — Звіт (Варіант {VARIANT:02d})",
        f"Джерело: {meta['source']}",
        f"UTC Time: {now}",
        "",
        "== Джерело та запит ==",
        f"Source: {meta['source']}",
        f"Request URL: {meta['request_url']}",
        "",
        "== Результат ==",
        f"Rows: {len(df)}",
        f"Columns: {list(df.columns)}",
        "",
        "== Head(5) ==",
        df.head(5).to_string(index=False),
        "",
        "== Пропуски (top) ==",
        df.isna().sum().sort_values(ascending=False).head(10).to_string(),
        "",
        "== Висновки ==",
        "1. Дані отримано з Open Brewery DB API. Отримано 50 записів про пивоварні.",
        "2. JSON структура була пласкою, тому json_normalize спрацював без додаткових розгортань.",
        "3. Проведено очистку: видалено дублікати, заповнено пропуски в категоріальних полях, типи координат приведено до числових.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("Fetching data...")
    raw, df, meta = fetch_breweries()

    print("Saving raw data...")
    save_raw(raw, OUT_RAW)

    print("Cleaning data...")
    df_clean = minimal_clean(df)
    df_clean.to_csv(OUT_CSV, index=False)

    print("Generating report...")
    rep = make_report(meta, df_clean)
    Path(OUT_REP).write_text(rep, encoding="utf-8")

    print(f"Saved: {OUT_RAW}, {OUT_CSV}, {OUT_REP}")
