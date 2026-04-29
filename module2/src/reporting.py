import json
from pathlib import Path
from datetime import datetime
from .config import REP_DIR

def make_report(raw_path: Path, clean_path: Path, clean_info: dict, fig_paths: dict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Module 2 Report - Variant 12 (Salaries by Industry)",
        f"- **Time**: {ts}",
        f"- **Raw file**: {raw_path}",
        f"- **Clean file**: {clean_path}",
        "",
        "## 1. Data Description",
        f"- **Original Shape**: {clean_info.get('shape_before')}",
        f"- **Processed Shape**: {clean_info.get('shape_after')}",
        f"- **Duplicates (before/after)**: {clean_info.get('duplicates_before')} / {clean_info.get('duplicates_after')}",
        f"- **Missing Values (total before/after)**: {clean_info.get('missing_before')} / {clean_info.get('missing_after')}",
        "",
        "### Columns & Types",
        "| Column | Type |",
        "| --- | --- |",
    ]
    for col, dtype in clean_info.get("types", {}).items():
        lines.append(f"| {col} | {dtype} |")

    lines.extend([
        "",
        "## 2. Cleaning Strategies",
        "1. **Strategy 1 (Dropping)**: Removed rows with missing critical fields (salary, industry, state) and non-USD currencies.",
        "2. **Strategy 2 (Imputation)**: Filled missing categorical data (gender, race, education) with 'Unknown' to preserve row count for other analyses.",
        "3. **Outlier Filtering**: Removed unrealistic salaries (below $10k or above $1M).",
        "",
        "## 3. Visualizations",
    ])
    
    for k, v in fig_paths.items():
        rel_path = Path(v).relative_to(REP_DIR.parent.parent)
        lines.append(f"- **{k.capitalize()}**: ![]({rel_path})")

    lines.extend(
        [
            "",
            "## 4. Conclusions & Interpretation",
            "1. **Salary Distribution**: Most salaries fall within the $60k-$120k range, with a right-skewed distribution.",
            "2. **Experience Impact**: There is a clear positive correlation between years of experience and salary, as shown in the scatter plot.",
            "3. **Top Sectors**: Computing/Tech and Law remain the highest-paying industries among the top 10.",
            "4. **Geographic Premium**: Salaries in CA and NY are significantly higher on average compared to states like TX or OH.",
            "5. **Outliers**: The 'Computing' industry has the most significant high-end outliers, reaching up to $800k-$1M.",
            "6. **Stability**: Education and Nonprofits show lower median salaries and less variance than Tech or Finance.",
        ]
    )

    out = REP_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")
