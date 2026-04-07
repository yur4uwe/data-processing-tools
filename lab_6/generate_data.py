import pandas as pd
import numpy as np

# Фіксуємо seed, щоб дані не змінювалися при кожному запуску
np.random.seed(42)
n = 500  # Кількість акаунтів

# 1. Категорії з різними ймовірностями випадіння
niches = ["Beauty", "Tech", "Gaming", "Lifestyle", "Education"]
countries = [
    "USA",
    "UK",
    "Ukraine",
    "Canada",
    "Germany",
    "France",
    "Poland",
    "Spain",
    "Italy",
    "Japan",
]

data = {
    "niche": np.random.choice(niches, n, p=[0.3, 0.2, 0.25, 0.15, 0.1]),
    "country": np.random.choice(
        countries, n, p=[0.3, 0.15, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05, 0.03, 0.02]
    ),
}
df = pd.DataFrame(data)

# 2. Підписники (log-normal: багато малих акаунтів і одиниці мільйонників)
df["followers"] = np.random.lognormal(mean=9, sigma=1.5, size=n).astype(int)
df["followers"] = np.clip(df["followers"], 100, 5000000)

# 3. Кількість постів (залежить від кількості підписників + трохи рандому)
df["posts"] = (df["followers"] ** 0.3 * np.random.uniform(2, 10, n)).astype(int)

# 4. Залученість (Engagement)
# Формула: базова залученість падає при зростанні followers + бонус від ніші + шум
base_eng = 10 - np.log10(df["followers"]) * 1.5
niche_bonus = {
    "Education": 3.0,
    "Gaming": 2.0,
    "Tech": 1.0,
    "Beauty": 0.0,
    "Lifestyle": -1.0,
}
niche_effect = df["niche"].map(lambda x: niche_bonus[x])
noise = np.random.normal(0, 1.2, n)

df["engagement"] = np.clip(base_eng + niche_effect + noise, 0.5, 25.0).round(2)

# 5. Середня кількість лайків (вираховуємо математично: followers * engagement / 100)
df["likes_avg"] = ((df["followers"] * df["engagement"]) / 100).astype(int)

# Зберігаємо файл
df.to_csv("variant12.csv", index=False)
print("Файл variant12.csv успішно згенеровано!")
