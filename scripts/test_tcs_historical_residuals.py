import numpy as np
import pandas as pd
from pathlib import Path

SOURCE = Path("data/processed/daily_returns.csv")
OUTPUT = Path("data/processed/tcs_historical_residual_test.csv")

STOCK = "TCS.NS"
MARKET = "^NSEI"
EVENT_DATE = pd.Timestamp("2024-06-04")

df = pd.read_csv(SOURCE, parse_dates=["date"])
df["date"] = df["date"].dt.normalize()

wide = df.pivot(
    index="date",
    columns="ticker",
    values="daily_return",
).sort_index()

dates = list(wide.index)
date_to_idx = {date: i for i, date in enumerate(dates)}
event_idx = date_to_idx[EVENT_DATE]

# Baseline estimation window: trading days -120 to -21.
est_dates = dates[event_idx - 120:event_idx - 20]
est = wide.loc[est_dates, [STOCK, MARKET]].dropna()

X = np.column_stack([
    np.ones(len(est)),
    est[MARKET].to_numpy(),
])
y = est[STOCK].to_numpy()

alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
residuals = y - X @ np.array([alpha, beta])

# June 4 event-day abnormal return.
market_event = float(wide.loc[EVENT_DATE, MARKET])
actual_event = float(wide.loc[EVENT_DATE, STOCK])
expected_event = alpha + beta * market_event
event_ar = actual_event - expected_event

# Empirical two-sided residual-tail diagnostic.
extreme_count = int(
    np.sum(np.abs(residuals) >= abs(event_ar))
)
empirical_p = (extreme_count + 1) / (len(residuals) + 1)

result = pd.DataFrame([{
    "event_date": EVENT_DATE.date(),
    "ticker": STOCK,
    "estimation_observations": len(residuals),
    "actual_return": actual_event,
    "expected_return": expected_event,
    "event_abnormal_return": event_ar,
    "residual_std": residuals.std(ddof=2),
    "historical_residuals_at_least_as_extreme": extreme_count,
    "empirical_two_sided_p_approx": empirical_p,
    "historical_residual_95th_percentile_abs": np.quantile(
        np.abs(residuals), 0.95
    ),
}])

result.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print(result.to_string(index=False))

print(
    "\nInterpretation: this is an empirical residual-tail diagnostic, "
    "not a definitive event-study significance test. The residuals are "
    "in-sample and may not be independent or identically distributed."
)
