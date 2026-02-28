import numpy as np
from numpy._typing import NDArray

ROWS = 600
COLUMNS = 4

COL_DIM = 0
ROW_DIM = 1


def minmax_norm(X):
    return (X - X.min(axis=COL_DIM)) / (
        X.max(axis=COL_DIM) - X.min(axis=COL_DIM) + 1e-8
    )


def zscore_norm(X):
    return (X - X.mean(axis=COL_DIM)) / (X.std(axis=COL_DIM) + 1e-8)


rng = np.random.default_rng(23067021)

X = rng.normal(0, 1, (ROWS, COLUMNS))

np.save("X_variant12.npy", X)

print("Array shape:", X.shape)
print("Array dtype:", X.dtype)
print("Array ndim:", X.ndim)
print("Column maxima:", X.max(axis=COL_DIM))

# Variant 12 vector operation: row-wise normalization
norms = np.linalg.norm(X, axis=ROW_DIM, keepdims=True)
# Avoid division by zero for rows that are all zeros
norms_safe = np.where(norms == 0, 1e-8, norms)
X_norm: NDArray[np.float64] = X / norms_safe

# Column-wise aggregation: mean, std, median
print("Column means:", X.mean(axis=COL_DIM))
print("Column standard deviations:", X.std(axis=COL_DIM))
print("Column medians:", np.median(X, axis=COL_DIM))

X_minmax_norm = minmax_norm(X)
X_zscore_norm = zscore_norm(X)

np.save("X_minmax_norm_variant12.npy", X_minmax_norm)
np.save("X_zscore_std_variant12.npy", X_zscore_norm)

lines = []
lines.append("Лабораторна робота No2 — Звіт (Варіант 12)")
lines.append(f"Матриця X: shape={X.shape}, dtype={X.dtype}, ndim={X.ndim}")
lines.append("")
lines.append("Агрегати по колонках (X):")
lines.append(f"- mean:{np.round(X.mean(axis=COL_DIM), 4)}")
lines.append(f"- std:{np.round(X.std(axis=COL_DIM), 4)}")
lines.append(f"- median:{np.round(np.median(X, axis=COL_DIM), 4)}")
lines.append(f"- min:{np.round(X.min(axis=COL_DIM), 4)}")
lines.append(f"- max:{np.round(X.max(axis=COL_DIM), 4)}")
lines.append("")
lines.append("Перевірка нормалізації / стандартизації:")
lines.append(f"- X_norm min~ {np.round(X_minmax_norm.min(axis=COL_DIM), 4)}")
lines.append(f"- X_norm max~ {np.round(X_minmax_norm.max(axis=COL_DIM), 4)}")
lines.append(f"- X_std mean~ {np.round(X_zscore_norm.mean(axis=COL_DIM), 4)}")
lines.append(f"- X_std std~ {np.round(X_zscore_norm.std(axis=COL_DIM), 4)}")
lines.append("")
lines.append("Результати операції варіанту (коротко):")
lines.append("")
lines.append(f"X_norm_variant12: array shape={X_norm.shape}")
lines.append(f" first 5 items: {np.round(X_norm.ravel()[:5], 4).tolist()}")

with open("report_variant12.txt", "w") as f:
    f.write("\n".join(lines))

print("Routine Successfully Completed")
