# Варіант 12 - Соцмережі (регресія)
# followers, posts, likes_avg, niche, country, engagement
# TARGET: engagement
# NUM: followers, posts, likes_avg
# CAT: niche, country
# Scaler: MinMaxScaler

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

NUM_COLS = ["followers", "posts", "likes_avg"]
CAT_COLS = ["niche", "country"]
TARGET = "engagement"


def generate_dataset(size: int = 1000):
    np.random.seed(230420069)
    niches = ["fitness", "travel", "food", "tech", "fashion"]
    countries = ["USA", "UK", "Germany", "France", "India"]

    df = pd.DataFrame(
        {
            "followers": np.random.randint(1000, 1000000, size),
            "posts": np.random.randint(10, 5000, size),
            "likes_avg": np.random.uniform(10, 50000, size),
            "niche": np.random.choice(niches, size),
            "country": np.random.choice(countries, size),
            "engagement": np.random.uniform(0.5, 20.0, size),  # target for regression
        }
    )

    df.to_csv("social_media.csv", index=False)
    print(df.head())
    return df


def preprocessing_pipeline():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUM_COLS),
            ("cat", categorical_pipeline, CAT_COLS),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    if not os.path.exists("social_media.csv"):
        generate_dataset()
    df = pd.read_csv("social_media.csv")
    print(df.head())

    preprocessor = preprocessing_pipeline()

    X = df.drop(columns=TARGET)
    y = df[TARGET].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=230420069
    )

    X_train = np.array(preprocessor.fit_transform(X_train_raw))
    X_test = np.array(preprocessor.transform(X_test_raw))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    one_hot_en_cols = preprocessor.named_transformers_["cat"][
        "encoder"
    ].get_feature_names_out(CAT_COLS)
    all_cols = NUM_COLS + list(one_hot_en_cols)

    transformed_df = pd.DataFrame(X_train, columns=all_cols)
    transformed_df["engagement"] = y_train
    transformed_df.to_csv("features.csv", index=False)

    report = f"""
    Variant 12 - Social Media (Regression)
    ========================================
    TARGET:   {TARGET}
    NUM_COLS: {NUM_COLS}
    CAT_COLS: {CAT_COLS}
    Scaler:   MinMaxScaler
    Imputer:  numeric=median, categorical=most_frequent

    Dataset shape:  {df.shape}
    X_train shape:  {X_train.shape}
    X_test shape:   {X_test.shape}
    y_train shape:  {y_train.shape}
    y_test shape:   {y_test.shape}

    Features after encoding: {all_cols}
    """

    with open("report_variant12.txt", "w") as f:
        f.write(report)

    print(report)
