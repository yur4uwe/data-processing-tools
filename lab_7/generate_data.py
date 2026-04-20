import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(230490210420)
n = 500

niches = ["Beauty", "Tech", "Gaming", "Lifestyle", "Education"]

data = {
    "niche": np.random.choice(niches, n, p=[0.3, 0.2, 0.25, 0.15, 0.1]),
}
df = pd.DataFrame(data)

# Followers (log-normal distribution)
df["followers"] = np.random.lognormal(mean=9, sigma=1.5, size=n).astype(int)
df["followers"] = np.clip(df["followers"], 100, 5000000)

# Engagement
# Base engagement decreases as followers increase + niche bonus + noise
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

# Save to CSV
df.to_csv("variant12.csv", index=False)
print("variant12.csv generated successfully!")
