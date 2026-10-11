import pandas as pd
import numpy as np
from pathlib import Path

SOURCE = Path("data/processed/event_study_inference.csv")
OUTPUT = Path("data/processed/event_study_inference_adjusted.csv")

df = pd.read_csv(SOURCE)

p = df["p_value_two_sided"].to_numpy(dtype=float)
valid = np.isfinite(p)
pv = p[valid]
m = len(pv)

# Holm family-wise error correction.
order = np.argsort(pv)
sorted_p = pv[order]
holm_sorted = np.maximum.accumulate(
    (m - np.arange(m)) * sorted_p
)
holm_sorted = np.minimum(holm_sorted, 1.0)

holm = np.empty(m)
holm[order] = holm_sorted

# Benjamini-Hochberg false discovery rate correction.
order_bh = np.argsort(pv)
sorted_bh = pv[order_bh]
ranks = np.arange(1, m + 1)

bh_sorted = sorted_bh * m / ranks
bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]
bh_sorted = np.minimum(bh_sorted, 1.0)

bh = np.empty(m)
bh[order_bh] = bh_sorted

df["p_holm"] = np.nan
df["p_bh"] = np.nan
df.loc[valid, "p_holm"] = holm
df.loc[valid, "p_bh"] = bh

df["significant_holm_5pct"] = df["p_holm"] < 0.05
df["significant_bh_5pct"] = df["p_bh"] < 0.05

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Tests adjusted:", m)
print("Unadjusted p < 0.05:", int((pv < 0.05).sum()))
print("Holm-adjusted p < 0.05:", int((holm < 0.05).sum()))
print("BH-adjusted p < 0.05:", int((bh < 0.05).sum()))

print("\nDay-0 results:")
cols = [
    "event_id",
    "ticker",
    "car_percent_approx",
    "p_value_two_sided",
    "p_holm",
    "p_bh",
    "significant_holm_5pct",
    "significant_bh_5pct",
]
print(
    df.loc[df["window"] == "day_0", cols]
    .sort_values("p_value_two_sided")
    .round(5)
    .to_string(index=False)
)
