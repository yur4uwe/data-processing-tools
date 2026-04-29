import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {}
    info["shape_before"] = list(df.shape)
    info["columns"] = list(df.columns)
    info["types"] = {str(k): str(v) for k, v in df.dtypes.items()}
    info["missing_before"] = int(df.isna().sum().sum())
    info["duplicates_before"] = int(df.duplicated().sum())
    info["non_usd_currency"] = len(df[df["currency"] != "USD"])

    # 1. Basic Cleaning: Drop duplicates
    df = df.drop_duplicates().copy()

    # 2. Filter for USD to keep salaries comparable
    if "currency" in df.columns:
        df = df[df["currency"] == "USD"]

    # 3. Missing Value Strategy 2: Fill missing categorical values with 'Unknown'
    categorical_cols = ["gender", "race", "highest_level_of_education_completed"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 4. Type Casting: Clean 'annual_salary'
    df["annual_salary"] = pd.to_numeric(df["annual_salary"], errors="coerce")

    # 5. Missing Value Strategy 1: Drop rows with missing critical values
    df = df.dropna(subset=["annual_salary", "industry", "state"])

    # 6. Outlier Removal: Filter realistic salary range
    df = df[(df["annual_salary"] > 10000) & (df["annual_salary"] < 1000000)]

    # 7. Focused Data: Take top 10 industries and top 10 states
    top_industries = df["industry"].value_counts().head(10).index
    top_states = df["state"].value_counts().head(10).index
    df = df[df["industry"].isin(top_industries) & df["state"].isin(top_states)]

    # 8. Feature Engineering for visualization (Experience mapping)
    exp_map = {
        "1 year or less": 1,
        "2 - 4 years": 3,
        "5-7 years": 6,
        "8 - 10 years": 9,
        "11 - 20 years": 15,
        "21 - 30 years": 25,
        "31 - 40 years": 35,
        "41 years or more": 45,
    }
    if "overall_years_of_professional_experience" in df.columns:
        df["exp_years"] = df["overall_years_of_professional_experience"].map(exp_map)

    info["shape_after"] = list(df.shape)
    info["missing_after"] = int(df.isna().sum().sum())
    info["duplicates_after"] = int(df.duplicated().sum())

    return df, info
