from pathlib import Path

import numpy as np
import pandas as pd

# ==== ПАРАМЕТРИ ВАРІАНТУ 12 ====
VARIANT = 12
FILE_NAME = "variant12.csv"
CLEAN_FILE = f"clean_variant{VARIANT}.csv"
OUTLIERS_FILE = f"outliers_variant{VARIANT}.csv"
REPORT_FILE = f"report_variant{VARIANT}.txt"

DUP_KEYS = ["order_id"]
CAT_COLS = ["dish", "category"]
NUM_COLS = ["price", "rating"]
OUTLIER_COL = "price"


# --- 1. ГЕНЕРАЦІЯ ДАНИХ З ПУСТИМИ КЛІТИНКАМИ ---
def generate_data():
    np.random.seed(42)
    n_rows = 100

    data = {
        "order_id": [
            i if i != 10 else 9 for i in range(1, n_rows + 1)
        ],  # Дублікат ID 9
        # Генеруємо пусті клітинки (None та np.nan перетворюються на пусті поля в CSV)
        "dish": np.random.choice(
            ["Pasta", "Pizza", "Salad", "Steak", "Soup", None], n_rows
        ),
        "category": np.random.choice(["Main", "Starter", "Dessert", None], n_rows),
        "price": np.random.normal(250, 50, n_rows).tolist(),
        "rating": np.random.choice(
            [1, 2, 3, 4, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], n_rows
        ),
    }

    df = pd.DataFrame(data)

    # Додаткові явні пропуски
    df.loc[5:10, "price"] = np.nan  # Пусті ціни
    df.loc[15:20, "category"] = np.nan  # Пусті категорії
    df.loc[25:30, "dish"] = None  # Пусті страви

    # Аномалії для price (щоб спрацював Z-score > 3)
    df.loc[0, "price"] = 2000.0
    df.loc[1, "price"] = -500.0

    df.to_csv(FILE_NAME, index=False)
    print(f"Згенеровано {FILE_NAME} з {n_rows} рядками.")


def missing_summary(df):
    total = len(df)
    miss = df.isna().sum()
    pct = (miss / max(total, 1) * 100).round(2)
    return pd.DataFrame({"missing": miss, "missing_%": pct})


def zscore_mask(series, threshold=3.0):
    s = series.astype(float)
    mu = s.mean()
    sd = s.std()
    z = (s - mu) / (sd + 1e-12)
    return z.abs() > threshold


if __name__ == "__main__":
    # Спочатку генеруємо файл
    generate_data()

    df = pd.read_csv(FILE_NAME)

    # Звіт "ДО"
    miss_before = missing_summary(df)
    dup_before = df.duplicated(subset=DUP_KEYS).sum()

    # --- ЕТАП 1: ОБРОБКА ПРОПУСКІВ ---
    # 1.1 Спочатку числові (ціна)
    df["price"] = df["price"].fillna(df["price"].median())

    # 1.2 Категоріальні (dish, category) - ВАЖЛИВО заповнити ДО групування
    for col in CAT_COLS:
        # Знаходимо моду, ігноруючи NaN
        mode_val = df[col].mode().dropna()
        fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        # Заповнюємо NaN і пусті рядки
        df[col] = df[col].fillna(fill_val).replace("", fill_val)

    # 1.3 Рейтинг (залежить від категорії)
    # Тепер category заповнена, тому groupby не втратить рядки
    df["rating"] = df["rating"].fillna(
        df.groupby("category")["rating"].transform("median")
    )
    # Якщо в якійсь категорії не було оцінок, заповнюємо загальною медіаною
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # --- ЕТАП 2: ДУБЛІКАТИ ---
    df = df.drop_duplicates(subset=DUP_KEYS, keep="first")

    # --- ЕТАП 3: АНОМАЛІЇ (Z-score) ---
    out_mask = zscore_mask(df[OUTLIER_COL])
    df[f"is_outlier_{OUTLIER_COL}"] = out_mask.astype(int)

    # Розділяємо на чисті та аномальні
    df_outliers = df[df[f"is_outlier_{OUTLIER_COL}"] == 1].copy()
    df_clean = df[df[f"is_outlier_{OUTLIER_COL}"] == 0].copy()

    # --- 3. ЗБЕРЕЖЕННЯ ТА ЗВІТ ---
    df_clean.to_csv(CLEAN_FILE, index=False)
    df_outliers.to_csv(OUTLIERS_FILE, index=False)

    report_lines = [
        f"Лабораторна робота №4. Варіант {VARIANT}",
        f"Вхідний файл: {FILE_NAME}",
        "",
        "== ДО ОЧИЩЕННЯ ==",
        f"Рядків: {len(pd.read_csv(FILE_NAME))}",
        f"Дублікатів за ключем {DUP_KEYS}: {dup_before}",
        "Пропуски (top):",
        miss_before.to_string(),
        "",
        "== ПІСЛЯ ОЧИЩЕННЯ ==",
        f"Рядків у чистому файлі: {len(df_clean)}",
        f"Виявлено аномалій (price): {len(df_outliers)} (дія: drop)",
        "Пропуски у чистому файлі:",
        missing_summary(df_clean).to_string(),
        "",
        "Згенеровані файли:",
        f"{CLEAN_FILE}, {OUTLIERS_FILE}, {REPORT_FILE}",
    ]

    Path(REPORT_FILE).write_text("\n".join(report_lines), encoding="utf-8")

    # Вивід у консоль для перевірки
    print("\n".join(report_lines))
