# Diamonds Price Prediction Pipeline

This project provides a modular and reproducible machine learning pipeline designed to automate the process of diamond valuation. By analyzing physical characteristics and quality metrics, the system trains a regression model to predict market prices with high precision. It is built to be configuration-driven, allowing users to easily swap datasets, adjust preprocessing steps, and tune model parameters without modifying the core logic.

## Dataset Description
The project utilizes the **Diamonds dataset**, which contains the prices and attributes of approximately 54,000 diamonds. It serves as a classic benchmark for regression tasks due to its mix of continuous and categorical variables.

### Features:
- **Price:** Price in US dollars (\$326-\$18,823). This is the **target** variable.
- **Carat:** Weight of the diamond (0.2-5.01).
- **Cut:** Quality of the cut (Fair, Good, Very Good, Premium, Ideal).
- **Color:** Diamond colour, from J (worst) to D (best).
- **Clarity:** A measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)).
- **x, y, z:** Length, width, and depth in mm.
- **Depth:** Total depth percentage = z / mean(x, y) = 2 * z / (x + y).
- **Table:** Width of top of diamond relative to widest point.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python run.py --config configs/config.yml
```

## Outputs
- `logs/run_<id>.log`: Execution logs including memory usage and performance metrics.
- `artifacts/run_<id>/`:
  - `models/model.joblib`: The trained scikit-learn pipeline.
  - `metrics/metrics.json`: MAE, RMSE, and R² scores.
  - `predictions/predictions.csv`: Model predictions compared against true values.
  - `reports/report.md`: A summary report of the run.
  - `reports/config_snapshot.yaml`: A copy of the configuration used for the run.
