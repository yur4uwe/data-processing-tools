import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Any, Dict, cast, List

# ============================ SETTINGS FOR VARIANT 12 ============================
VARIANT = 12
INPUT_FILE = f"variant{VARIANT:02d}.csv"
ALPHA = 0.05

# 2 numeric columns for descriptive stats and correlation
NUMERIC_COL_1 = "followers"
NUMERIC_COL_2 = "engagement"

# 1 categorical column for frequencies
CAT = "niche"

# Hypothesis test type: "ttest" | "mannwhitney" | "anova" | "chi2"
TEST_TYPE = "anova"

# Settings for ANOVA
ANOVA_GROUP_COL = "niche"
ANOVA_VALUE_COL = "engagement"

# ============================ HELPER FUNCTIONS ============================


def descriptive(series: pd.Series) -> Dict[str, Any]:
    s = cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
    if s.empty:
        return {}

    return {
        "n": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": cast(float, s.std(ddof=1)) if s.count() > 1 else float("nan"),
        "min": float(s.min()),
        "max": float(s.max()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
    }


def cohens_d(x: Any, y: Any) -> float:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    # Filter NaNs from numpy arrays
    x_arr = x_arr[~np.isnan(x_arr)]
    y_arr = y_arr[~np.isnan(y_arr)]
    nx, ny = len(x_arr), len(y_arr)
    if nx < 2 or ny < 2:
        return float("nan")
    sx = np.var(x_arr, ddof=1)
    sy = np.var(y_arr, ddof=1)
    sp = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    return float((np.mean(x_arr) - np.mean(y_arr)) / (sp + 1e-12))


def find_correlation(lines: list[str], df: pd.DataFrame):
    lines.append("== Кореляційний аналіз ==")
    if NUMERIC_COL_1 not in df.columns and NUMERIC_COL_2 not in df.columns:
        lines.append(f"Колонки {NUMERIC_COL_1}/{NUMERIC_COL_2} не знайдено.")
        return

    # Cast return of pd.to_numeric to Series
    x_ser = cast(pd.Series, pd.to_numeric(df[NUMERIC_COL_1], errors="coerce"))
    y_ser = cast(pd.Series, pd.to_numeric(df[NUMERIC_COL_2], errors="coerce"))

    # Valid indices mask
    valid_mask = x_ser.notna() & y_ser.notna()
    # Ensure mask is interpreted as boolean series for indexing
    x_valid = x_ser[valid_mask]
    y_valid = y_ser[valid_mask]

    if len(x_valid) >= 3:
        pearson_r, pearson_p = stats.pearsonr(x_valid, y_valid)
        spearman_r, spearman_p = stats.spearmanr(x_valid, y_valid)
        lines.append(
            f"Pearson r = {float(pearson_r):.4f}, p = {float(pearson_p):.6f}"  # pyright: ignore[reportArgumentType]
        )
        lines.append(
            f"Spearman ρ = {float(spearman_r):.4f}, p = {float(spearman_p):.6f}"  # pyright: ignore[reportArgumentType]
        )
    else:
        lines.append("Недостатньо даних для кореляції.")
    lines.append("")


def hypothesis_test(lines: list[str], df: pd.DataFrame):
    lines.append("== Перевірка статистичної гіпотези ==")
    if ANOVA_GROUP_COL not in df.columns and ANOVA_VALUE_COL not in df.columns:
        lines.append("Недостатньо груп/даних для ANOVA.")
        return

    # Narrow to DataFrame for subsetting
    sub = df[[ANOVA_GROUP_COL, ANOVA_VALUE_COL]].copy()
    sub[ANOVA_VALUE_COL] = pd.to_numeric(sub[ANOVA_VALUE_COL], errors="coerce")
    sub = sub.dropna()

    groups: List[np.ndarray] = []
    for _, g in sub.groupby(ANOVA_GROUP_COL):
        val_col = g[ANOVA_VALUE_COL]
        if not isinstance(val_col, pd.Series):
            continue
        arr = val_col.to_numpy()
        if arr.size > 1:
            groups.append(arr)

    if len(groups) < 2:
        lines.append("Недостатньо груп/даних для ANOVA.")
        return

    # Use *groups to unpack the list of arrays
    f_stat, p_val = stats.f_oneway(*groups)
    lines.append(
        f"Тест: One-way ANOVA для {ANOVA_VALUE_COL} між групами {ANOVA_GROUP_COL}"
    )
    lines.append(f"F-stat: {float(f_stat):.4f}, p-value: {float(p_val):.6f}")

    # Simple Eta-squared calculation
    all_data = np.concatenate(groups)
    overall_mean = np.mean(all_data)
    ss_total = np.sum((all_data - overall_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - overall_mean) ** 2 for g in groups)
    eta_sq = ss_between / (ss_total + 1e-12)
    lines.append(f"Effect size (η^2): {float(eta_sq):.4f}")

    if p_val < ALPHA:
        lines.append("Висновок: Відхиляємо H0, є різниця між нішами")
    else:
        lines.append("Висновок: Немає підстав відхилити H0, немає суттєвої різниці")


def main():
    if not Path(INPUT_FILE).exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    lines: List[str] = []

    # 1) Basic Info
    lines.append(f"Лабораторна робота №7 — Звіт (Варіант {VARIANT:02d})")
    lines.append(f"Вхідний файл: {INPUT_FILE}")
    lines.append(f"Рівень значущості α = {ALPHA}")
    lines.append(f"Розмір датасету: {df.shape}")
    lines.append("")

    # 2) Descriptive Statistics
    lines.append("== Описова статистика ==")

    for col_name in [NUMERIC_COL_1, NUMERIC_COL_2]:
        if col_name in df.columns:
            col_data = df[col_name]
            # Narrow type to Series
            if isinstance(col_data, pd.Series):
                res = descriptive(col_data)
                lines.append(f"{col_name}: {res}")
    lines.append("")

    # 3) Categorical Frequencies
    lines.append("== Частоти категоріальної змінної ==")
    if CAT in df.columns:
        cat_data = df[CAT]
        if isinstance(cat_data, pd.Series):
            freq = cat_data.astype("string").value_counts(dropna=False).head(10)
            lines.append(freq.to_string())
    else:
        lines.append(f"Колонка {CAT} не знайдена.")
    lines.append("")

    # 4) Correlation (Pearson + Spearman)
    find_correlation(lines, df)
    # 5) Hypothesis Testing
    hypothesis_test(lines, df)

    # 6) Final Conclusions (Summary)
    lines.append("")
    lines.append("== Підсумкові висновки ==")
    lines.append(
        "1. Описова статистика показує розподіл підписників та рівня залученості."
    )
    lines.append(
        "2. Кореляційний аналіз дозволяє оцінити зв'язок між розміром акаунту та залученістю."
    )
    lines.append(
        "3. Тест ANOVA перевіряє, чи впливає тематика (ніша) на рівень залученості."
    )

    output_file = f"report_variant{VARIANT:02d}.txt"
    Path(output_file).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved ✅: {output_file}")


if __name__ == "__main__":
    main()
