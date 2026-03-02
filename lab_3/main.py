from pathlib import Path

import pandas as pd

# === НАЛАШТУВАННЯ ПІД ВАРІАНТ 12 ===
VARIANT = 12
FILE_A = f"variant{VARIANT:02d}_A.csv"
FILE_B = f"variant{VARIANT:02d}_B.csv"

# Ключ злиття та тип
MERGE_KEY = "pharmacy_id"
MERGE_HOW = "left"


# Фільтр: qty >= 3
def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["qty"] >= 3]  # pyright: ignore[reportReturnType]


def group_report(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("pharmacy_id")
        .agg(
            sum_qty=("qty", "sum"),
            mean_price=("price", "mean"),
            count_sales=("sale_id", "count"),
        )
        .reset_index()
    )


# === ЗВІТ ===
def make_text_report(dfA, dfB, dfF, grp, merged) -> str:
    lines = []
    lines.append(f"Лабораторна робота №3 - Звіт (Варіант {VARIANT})")
    lines.append("")
    lines.append("== Таблиця A ==")
    lines.append(f"shape: {dfA.shape}")
    lines.append(f"columns: {list(dfA.columns)}")
    lines.append("")
    lines.append("== Таблиця B ==")
    lines.append(f"shape: {dfB.shape}")
    lines.append(f"columns: {list(dfB.columns)}")
    lines.append("")
    lines.append("== Після фільтрації A ==")
    lines.append(f"shape: {dfF.shape}")
    lines.append("")
    lines.append("== GroupBy report (перші 10 рядків) ==")
    lines.append(grp.head(10).to_string(index=False))
    lines.append("")
    lines.append("== Merge result ==")
    lines.append(f"merge key: {MERGE_KEY}, how: {MERGE_HOW}")
    lines.append(f"shape: {merged.shape}")
    lines.append("")
    return "\n".join(lines)


# === MAIN ===
def main():
    dfA = pd.read_csv(FILE_A)
    dfB = pd.read_csv(FILE_B)

    print("Справжні назви колонок A:", dfA.columns.tolist())  # 2) Фільтрація
    dfF = apply_filter(dfA)

    # 3) Groupby + agg
    grp = group_report(dfF)

    # 4) Merge
    merged = pd.merge(dfF, dfB, on=MERGE_KEY, how=MERGE_HOW)

    # 5) Збереження
    dfF.to_csv(f"filtered_variant{VARIANT:02d}.csv", index=False)
    grp.to_csv(f"group_report_variant{VARIANT:02d}.csv", index=False)
    merged.to_csv(f"merged_variant{VARIANT:02d}.csv", index=False)

    report = make_text_report(dfA, dfB, dfF, grp, merged)
    Path(f"report_variant{VARIANT:02d}.txt").write_text(report, encoding="utf-8")

    print("Saved: filtered, group_report, merged, report")


if __name__ == "__main__":
    main()
