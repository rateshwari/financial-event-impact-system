
import numpy as np
import pandas as pd
from pathlib import Path

SOURCE = Path("data/processed/daily_returns.csv")
OUTPUT = Path("data/processed/tcs_placebo_test.csv")

STOCK = "TCS.NS"
MARKET = "^NSEI"
EVENT_DATE = pd.Timestamp("2024-06-04")

# Keep the baseline estimation design consistent with the main event study.
ESTIMATION_START = 120
ESTIMATION_END = 20  # exclusive index boundary: estimation days -120 to -21
MIN_ESTIMATION_OBS = 100

df = pd.read_csv(SOURCE, parse_dates=["date"])
df["date"] = df["date"].dt.normalize()

wide = (
    df.pivot(index="date", columns="ticker", values="daily_return")
    .sort_index()
)

if STOCK not in wide.columns or MARKET not in wide.columns:
    raise ValueError("Required TCS or NIFTY return series is missing.")

dates = list(wide.index)
date_to_idx = {date: i for i, date in enumerate(dates)}

if EVENT_DATE not in date_to_idx:
    raise ValueError(f"Event date {EVENT_DATE.date()} is not a trading date.")

event_idx = date_to_idx[EVENT_DATE]


def calculate_abnormal_return(target_idx):
    """
    Estimate the market model using trading days -120 to -21
    relative to the target date, then calculate target-day AR.
    """
    start_idx = target_idx - ESTIMATION_START
    end_idx = target_idx - ESTIMATION_END

    if start_idx < 0:
        return None

    estimation_dates = dates[start_idx:end_idx]
    estimation = wide.loc[estimation_dates, [STOCK, MARKET]].dropna()

    if len(estimation) < MIN_ESTIMATION_OBS:
        return None

    X = np.column_stack([
        np.ones(len(estimation)),
        estimation[MARKET].to_numpy(),
    ])
    y = estimation[STOCK].to_numpy()

    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]

    target_date = dates[target_idx]
    target = wide.loc[target_date, [STOCK, MARKET]]

    if target.isna().any():
        return None

    actual = float(target[STOCK])
    market_return = float(target[MARKET])
    expected = float(alpha + beta * market_return)
    abnormal = actual - expected

    return {
        "date": target_date.date().isoformat(),
        "actual_return": actual,
        "market_return": market_return,
        "expected_return": expected,
        "abnormal_return": abnormal,
        "estimation_observations": len(estimation),
        "alpha": float(alpha),
        "beta": float(beta),
    }


# Calculate the observed event-day abnormal return.
event_result = calculate_abnormal_return(event_idx)

if event_result is None:
    raise ValueError("Could not calculate the event-day abnormal return.")

event_ar = event_result["abnormal_return"]

# Placebo dates must have their entire estimation window before the date
# and must not fall within 10 trading days of either June 3 or June 4.
# This reduces contamination from the two-day election episode.
placebo_rows = []

june3_idx = date_to_idx[pd.Timestamp("2024-06-03")]
event_indices = {june3_idx, event_idx}

for idx in range(ESTIMATION_START, len(dates)):
    target_date = dates[idx]

    # Exclude dates close to either event day.
    if any(abs(idx - e_idx) <= 10 for e_idx in event_indices):
        continue

    # Exclude placebo dates whose estimation window (-120 to -21)
    # contains either event day.
    estimation_indices = range(idx - 120, idx - 20)

    if any(e_idx in estimation_indices for e_idx in event_indices):
        continue

    result = calculate_abnormal_return(idx)

    if result is not None:
        result["is_event_date"] = False
        placebo_rows.append(result)

placebos = pd.DataFrame(placebo_rows)

if placebos.empty:
    raise ValueError("No valid placebo dates were generated.")

placebo_abs = placebos["abnormal_return"].abs()
extreme_count = int((placebo_abs >= abs(event_ar)).sum())

# Add-one empirical tail probability.
empirical_p = (extreme_count + 1) / (len(placebos) + 1)

percentile = float(
    (placebo_abs < abs(event_ar)).mean() * 100
)

event_row = pd.DataFrame([{
    **event_result,
    "is_event_date": True,
}])

output = pd.concat(
    [event_row, placebos],
    ignore_index=True,
)

output.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("\nEVENT-DAY RESULT")
print(f"Date: {EVENT_DATE.date()}")
print(f"TCS actual return: {event_result['actual_return']:.4%}")
print(f"Expected return: {event_result['expected_return']:.4%}")
print(f"Abnormal return: {event_ar:.4%}")

print("\nPLACEBO-DATE SUMMARY")
print(f"Number of placebo dates: {len(placebos)}")
print(f"Placebo abnormal-return mean: {placebos['abnormal_return'].mean():.4%}")
print(f"Placebo abnormal-return standard deviation: {placebos['abnormal_return'].std(ddof=1):.4%}")
print(f"Placebo 95th percentile of absolute AR: {placebo_abs.quantile(0.95):.4%}")
print(f"Placebo 99th percentile of absolute AR: {placebo_abs.quantile(0.99):.4%}")
print(f"Placebo dates with |AR| >= event |AR|: {extreme_count}")
print(f"Empirical two-sided tail estimate: {empirical_p:.6f}")
print(f"Event absolute AR percentile: {percentile:.2f}")

print(
    "\nCaution: placebo dates are historical observations, not randomized "
    "experiments. Returns can be dependent over time, and the empirical "
    "tail estimate does not establish that news caused the event-day move."
)
