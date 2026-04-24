from sklearn.ensemble import RandomForestRegressor


def build_regression(params: dict):
    return RandomForestRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", None),
        random_state=params.get("seed", 42),
        n_jobs=-1,
    )
