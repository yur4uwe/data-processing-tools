# Варіант 12 - Соцмережі
# followers, posts, likes_avg, niche, country, engagement
# Histogram: engagement
# Scatter: followers vs engagement
# Box: engagement by niche
# Bar: country top-10

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === НАЛАШТУВАННЯ ПІД ВАРІАНТ 12 ===
VARIANT = 12
INPUT_FILE = f"variant{VARIANT:02d}.csv"  # Файл має називатись variant12.csv

# Колонки під датасет Варіанту 12
NUM_HIST = "engagement"  # гістограма
X_SCAT = "followers"  # scatter X
Y_SCAT = "engagement"  # scatter Y
CAT_GROUP = "niche"  # категорія для boxplot
NUM_GROUP = "engagement"  # числова для boxplot
BAR_GROUP = "country"  # категорія для bar chart (Топ-10)

# Тренду у 12 варіанті немає
DATE_COL = None
TREND_VALUE = None
TREND_FREQ = "D"


# === ДОПОМІЖНІ ФУНКЦІЇ ===
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path, title: str):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    return pd.DataFrame({"missing": miss, "missing_%": pct}).sort_values(
        "missing", ascending=False
    )


# === ОСНОВНИЙ БЛОК ===
def main():
    # Завантаження даних
    # Якщо файлу немає, розкоментуй створення тестового датасету нижче
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(
            f"Файл {INPUT_FILE} не знайдено! Будь ласка, переконайся, що він є у папці."
        )
        return

    plots_dir = Path(f"plots_variant{VARIANT:02d}")
    ensure_dir(plots_dir)

    # 1) Базовий EDA
    miss = missing_report(df)

    # 2) Гістограма
    plt.figure()
    sns.histplot(data=df, x=NUM_HIST, kde=True)
    save_fig(plots_dir / "01_hist.png", f"Histogram: {NUM_HIST}")

    # 3) Scatter
    plt.figure()
    sns.scatterplot(data=df, x=X_SCAT, y=Y_SCAT)
    save_fig(plots_dir / "02_scatter.png", f"Scatter: {X_SCAT} vs {Y_SCAT}")

    # 4) Boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x=CAT_GROUP, y=NUM_GROUP)
    plt.xticks(rotation=45)
    save_fig(plots_dir / "03_box.png", f"Boxplot: {NUM_GROUP} by {CAT_GROUP}")

    # 5) Bar frequencies (Топ-10 країн)
    plt.figure(figsize=(10, 4))
    top = df[BAR_GROUP].value_counts().head(10)
    sns.barplot(x=top.index.astype(str), y=top.values)
    plt.xticks(rotation=45)
    save_fig(plots_dir / "04_bar_top.png", f"Top categories: {BAR_GROUP}")

    # 7) Heatmap кореляцій (числові)
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")  # Додав annot=True для наочності
        save_fig(plots_dir / "06_corr_heatmap.png", "Correlation heatmap")

    # 8) Звіт
    lines = []
    lines.append(f"Лабораторна робота №6 - Звіт (Варіант {VARIANT})")
    lines.append(f"Вхідний файл: {INPUT_FILE}")
    lines.append("")
    lines.append(f"Розмір датасету: {df.shape}")
    lines.append("Колонки та типи: ")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("Пропуски (top-10):")
    lines.append(miss.head(10).to_string())
    lines.append("")
    lines.append("Згенеровані графіки: ")
    lines.append(f"- 01_hist.png (розподіл {NUM_HIST})")
    lines.append(f"- 02_scatter.png (зв'язок {X_SCAT} vs {Y_SCAT})")
    lines.append(f"- 03_box.png (порівняння {NUM_GROUP} між {CAT_GROUP})")
    lines.append(f"- 04_bar_top.png (частоти категорій {BAR_GROUP})")
    lines.append("- 06_corr_heatmap.png (кореляції числових ознак)")

    # Інтерпретація для звіту
    lines.append("")
    lines.append("--- Інтерпретація результатів ---")
    lines.append(
        "1. Гістограма (engagement): Демонструє форму розподілу рівня залученості, дозволяє оцінити наявність асиметрії або аномально високих/низьких значень."
    )
    lines.append(
        "2. Діаграма розсіювання (followers vs engagement): Відображає, чи існує лінійна або нелінійна залежність між кількістю підписників та рівнем залученості."
    )
    lines.append(
        "3. Boxplot (engagement by niche): Показує медіану, квартилі та можливі викиди залученості в розрізі кожної ніші, що дозволяє порівняти їх ефективність."
    )
    lines.append(
        "4. Bar Chart (country top-10): Візуалізує 10 найпопулярніших країн за кількістю акаунтів у вибірці."
    )
    lines.append(
        "5. Heatmap: Виявляє найсильніші математичні кореляції (взаємозв'язки) між усіма числовими колонками (наприклад, між followers, posts та likes_avg)."
    )

    report_path = Path(f"report_variant{VARIANT:02d}.txt")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Готово! Збережено у папку: {plots_dir} та файл звіту: {report_path}")


if __name__ == "__main__":
    main()
