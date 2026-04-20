import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(213467210420)

# Table B: accounts (account_id, owner, currency)
num_accounts = 20
owners = [f"Owner_{i}" for i in range(1, num_accounts + 1)]
currencies = ["USD", "EUR", "UAH"]

df_accounts = pd.DataFrame(
    {
        "account_id": range(1, num_accounts + 1),
        "owner": owners,
        "currency": np.random.choice(currencies, num_accounts),
    }
)

# Table A: ops (op_id, account_id, amount, type)
num_ops = 100
df_ops = pd.DataFrame(
    {
        "op_id": range(1, num_ops + 1),
        "account_id": np.random.randint(1, num_accounts + 1, num_ops),
        "amount": np.random.uniform(10, 5000, num_ops).round(2),
        "type": np.random.choice(["Deposit", "Withdrawal"], num_ops),
    }
)

# Save to CSV
df_accounts.to_csv("variant12_B.csv", index=False)
df_ops.to_csv("variant12_A.csv", index=False)

print("Generated variant12_A.csv and variant12_B.csv")
