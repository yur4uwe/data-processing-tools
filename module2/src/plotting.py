import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .config import FIG_DIR

def visualize(df: pd.DataFrame, prefix="module2"):
    fig_paths = {}

    # 1) Histogram: Salary Distribution (Distribution)
    plt.figure(figsize=(10, 6))
    sns.histplot(df["annual_salary"], bins=30, kde=True, log_scale=(True, False))
    plt.title("Annual Salary Distribution (USD) - X-Axis Log Scale")
    plt.xlabel("Salary (Log Scale)")
    plt.ylabel("Frequency")
    p1 = FIG_DIR / f"{prefix}_hist.png"
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    fig_paths["hist"] = str(p1)

    # 2) Boxplot: Salary by Industry (Anomaly Detection / Distribution)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="industry", y="annual_salary", data=df, hue="industry", palette="Set3", legend=False)
    plt.title("Salary by Industry (Outlier Detection)")
    plt.xticks(rotation=45, ha="right")
    p2 = FIG_DIR / f"{prefix}_boxplot.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    fig_paths["boxplot"] = str(p2)

    # 3) Bar chart: Average Salary by State (Comparison)
    plt.figure(figsize=(12, 6))
    state_avg = df.groupby("state")["annual_salary"].mean().sort_values(ascending=False)
    state_avg.plot(kind="bar", color="skyblue")
    plt.title("Average Annual Salary by State (Top 10)")
    plt.ylabel("Avg Salary (USD)")
    plt.xticks(rotation=45, ha="right")
    p3 = FIG_DIR / f"{prefix}_bar.png"
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    fig_paths["bar"] = str(p3)

    # 4) Scatter Plot: Salary vs. Experience (Dependency)
    if "exp_years" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="exp_years", y="annual_salary", data=df, alpha=0.3)
        # Add a trend line
        sns.regplot(x="exp_years", y="annual_salary", data=df, scatter=False, color="red")
        plt.yscale("log")  # Set Y-axis to log scale
        plt.title("Salary vs. Years of Experience (Log Scale)")
        plt.xlabel("Years of Experience (Estimated)")
        plt.ylabel("Annual Salary (USD, Log Scale)")
        p4 = FIG_DIR / f"{prefix}_scatter.png"
        plt.savefig(p4, bbox_inches="tight")
        plt.close()
        fig_paths["scatter"] = str(p4)

    # 5) Heatmap: Industry vs. State (Explanatory)
    pivot_table = df.pivot_table(
        values="annual_salary", index="industry", columns="state", aggfunc="mean"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Salary Heatmap: Industry vs. State")
    p5 = FIG_DIR / f"{prefix}_heatmap.png"
    plt.savefig(p5, bbox_inches="tight")
    plt.close()
    fig_paths["heatmap"] = str(p5)

    return fig_paths
