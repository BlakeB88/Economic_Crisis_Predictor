import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "imputed_dataset.csv"
OUTPUT_FILE = "crisis_summary_columnNormalized.csv"
COUNTRY_COL = "Country"
YEAR_COL = "Year"
LABEL_COL = "External_Debt_Crisis"
ROLLING_YEARS = 5
FUTURE_YEARS = 3


def rolling_slope(x):
    """Compute linear trend (slope) for the last 5 years."""
    if len(x) < ROLLING_YEARS or x.isna().any():
        return np.nan
    t = np.arange(len(x))
    slope, _ = np.polyfit(t, x, 1)
    return slope


df = pd.read_csv(INPUT_FILE)
df = df.sort_values([COUNTRY_COL, YEAR_COL]).reset_index(drop=True)


feature_cols = [c for c in df.columns if c not in [COUNTRY_COL, YEAR_COL, LABEL_COL]]
print(f"Detected {len(feature_cols)} features: {feature_cols}")


summaries = []
for country, g in df.groupby(COUNTRY_COL):
    g = g.copy().sort_values(YEAR_COL)
    for c in feature_cols:
        g[f"{c}_mean5"] = g[c].rolling(window=ROLLING_YEARS, min_periods=ROLLING_YEARS).mean()
        g[f"{c}_std5"] = g[c].rolling(window=ROLLING_YEARS, min_periods=ROLLING_YEARS).std()
        g[f"{c}_trend5"] = g[c].rolling(window=ROLLING_YEARS, min_periods=ROLLING_YEARS).apply(rolling_slope, raw=False)
    y = g[LABEL_COL].values
    future_crisis = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        future_crisis[i] = int(y[i+1:i+1+FUTURE_YEARS].max()) if i+1 < len(y) else 0
    g["target_crisis_next_3y"] = future_crisis
    summaries.append(g)

df_sum = pd.concat(summaries, ignore_index=True)
df_sum = df_sum.dropna().reset_index(drop=True)
columns_to_scale = [
    c for c in df_sum.columns 
    if c not in [COUNTRY_COL, YEAR_COL, LABEL_COL, "target_crisis_next_3y"]
]
scaler = StandardScaler()
df_sum[columns_to_scale] = scaler.fit_transform(df_sum[columns_to_scale])
df_sum.to_csv(OUTPUT_FILE, index=False)

