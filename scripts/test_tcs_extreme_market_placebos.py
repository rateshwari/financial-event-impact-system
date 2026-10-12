
import numpy as np
import pandas as pd
from pathlib import Path

SOURCE = Path("data/processed/daily_returns.csv")
OUTPUT = Path("data/processed/tcs_extreme_market_placebos.csv")

STOCK = "TCS.NS"
MARKET = "^NSEI"
EVENT_DATE = pd.Timestamp("2024-06-04")

# Match the original market-model estimation design.
ESTIMATION_LOOKBACK = 120
ESTIMATION_GAP = 20
MIN_OBSERVATIONS = 100

# Compare dates with large market moves.
MARKET_MOVE_THRESHOLD = 0.03  # absolute NIFTY return >= 3%

df = pd.read_csv(SOURCE, parse_dates=["date"])
df["date"] = df["date"].dt.normalize()

wide = (
    df.pivot(index="date", columns="ticker", values="daily_return")
    .sort_index()
)

dates = list(wide.index)
date_to_idx = {date: i for i, date in enumerate(dates)}

if EVENT_DATE not in date_to_idx:
    raise ValueError(f"{EVENT_DATE.date()} not found in trading dates.")

event_idx = date_to_idx[EVENT_DATE]


def calculate_abnormal_return(idx):
    start = idx - ESTIMATION_LOOKBACK
    end = idx - ESTIMATION_GAP

    if start < 0:
        return None

    est_dates = dates[start:end]
    est = wide.loc[est_dates, [STOCK, MARKET]].dropna()

    if len(est) < MIN_OBSERVATIONS:
        return None

    X = np.column_stack([
        np.ones(len(est)),
        est[MARKET].to_numpy(),
    ])
    y = est[STOCK].to_numpy()

    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]

    target_date = dates[idx]
    target = wide.loc[target_date, [STOCK, MARKET]]

    if target.isna().any():
        return None

    actual = float(target[STOCK])
    market = float(target[MARKET])
    expected = float(alpha + beta * market)

    return {
        "date": target_date.date().isoformat(),
        "actual_return": actual,
        "market_return": market,
        "expected_return": expected,
        "abnormal_return": actual - expected,
        "estimation_observations": len(est),
    }


event = calculate_abnormal_return(event_idx)

if event is None:
    raise ValueError("Unable to calculate the event-day abnormal return.")

# Exclude the election episode and dates whose estimation window
# includes either June 3 or June 4.
june3_idx = date_to_idx[pd.Timestamp("2024-06-03")]
event_indices = {june3_idx, event_idx}

rows = []

for idx in range(ESTIMATION_LOOKBACK, len(dates)):
    if any(abs(idx - e_idx) <= 10 for e_idx in event_indices):
        continue

    estimation_indices = range(
        idx - ESTIMATION_LOOKBACK,
        idx - ESTIMATION_GAP,
    )

    if any(e_idx in estimation_indices for e_idx in event_indices):
        continue

    result = calculate_abnormal_return(idx)

    if result is None:
        continue

    # Require a market move at least as large as the chosen threshold.
    if abs(result["market_return"]) < MARKET_MOVE_THRESHOLD:
        continue

    rows.append(result)

placebos = pd.DataFrame(rows)

if placebos.empty:
    raise ValueError(
        "No qualifying extreme-market placebo dates. "
        "Try lowering MARKET_MOVE_THRESHOLD to 0.03."
    )

event_abs_ar = abs(event["abnormal_return"])
placebo_abs_ar = placebos["abnormal_return"].abs()

extreme_count = int((placebo_abs_ar >= event_abs_ar).sum())
empirical_p = (extreme_count + 1) / (len(placebos) + 1)

print("EVENT DAY")
print(f"Date: {EVENT_DATE.date()}")
print(f"TCS abnormal return: {event['abnormal_return']:.4%}")
print(f"NIFTY return: {event['market_return']:.4%}")

print("\nEXTREME-MARKET PLACEBO SUMMARY")
print(f"Market-move threshold: {MARKET_MOVE_THRESHOLD:.1%}")
print(f"Qualifying placebo dates: {len(placebos)}")
print(f"Placebo 95th percentile |AR|: {placebo_abs_ar.quantile(0.95):.4%}")
print(f"Placebo 99th percentile |AR|: {placebo_abs_ar.quantile(0.99):.4%}")
print(f"Dates with |AR| >= event |AR|: {extreme_count}")
print(f"Empirical tail estimate: {empirical_p:.6f}")

# Save the event and the qualifying placebo observations.
event_row = pd.DataFrame([{
    **event,
    "comparison_group": "event",
}])

placebos["comparison_group"] = "extreme_market_placebo"

output = pd.concat([event_row, placebos], ignore_index=True)
output.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
print(
    "\nCaution: this is a conditional historical comparison, not a "
    "randomized test. The sample may be small, and the selected market "
    "threshold affects the result."
)
