"""Calculate a consistent-model, two-session descriptive CAR for June 3–4, 2024.

Uses the June 3 event model for both dates. The result is descriptive, not
causal, and this script does not calculate a new significance test.
"""

from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/event_study_abnormal_returns.csv")
OUTPUT = Path("data/processed/combined_election_episode_car.csv")

MODEL_EVENT_ID = "EVT_2024_06_03_001"
EPISODE_DATES = pd.to_datetime(["2024-06-03", "2024-06-04"])
EXPECTED_TICKERS = {
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "RELIANCE.NS",
    "TCS.NS",
}

daily = pd.read_csv(INPUT, parse_dates=["event_date", "date"])

required = {"event_id", "ticker", "window", "date", "abnormal_return"}
missing = required - set(daily.columns)
if missing:
    raise ValueError(f"Abnormal-return file missing columns: {sorted(missing)}")

# Use one consistent event model for both dates.
episode = daily[
    daily["event_id"].eq(MODEL_EVENT_ID)
    & daily["date"].isin(EPISODE_DATES)
].copy()

if episode.empty:
    raise ValueError(f"No observations found for model {MODEL_EVENT_ID}.")

# Different event-window labels repeat observations. Confirm all repeated
# values agree within this one model before deduplicating.
consistency = episode.groupby(["ticker", "date"])["abnormal_return"].nunique()
if (consistency > 1).any():
    raise ValueError(
        "Conflicting abnormal returns remain within the June 3 event model."
    )

unique_daily = episode.drop_duplicates(subset=["ticker", "date"]).copy()

if unique_daily.duplicated(["ticker", "date"]).any():
    raise ValueError("Duplicate ticker/date rows remain after deduplication.")

observed_tickers = set(unique_daily["ticker"])
if observed_tickers != EXPECTED_TICKERS:
    raise ValueError(
        f"Unexpected ticker set. Expected {sorted(EXPECTED_TICKERS)}, "
        f"found {sorted(observed_tickers)}."
    )

counts = unique_daily.groupby("ticker")["date"].nunique()
if not counts.eq(2).all():
    raise ValueError(f"Expected two dates per ticker; found:\n{counts}")

combined = (
    unique_daily.groupby("ticker", as_index=False)
    .agg(
        episode_start=("date", "min"),
        episode_end=("date", "max"),
        trading_days=("date", "nunique"),
        combined_car=("abnormal_return", "sum"),
    )
)

combined["combined_car_percent_approx"] = combined["combined_car"] * 100
combined["model_event_id"] = MODEL_EVENT_ID
combined["interpretation"] = (
    "Descriptive two-session sum under the June 3 event model; "
    "not causal and not a significance test."
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT, index=False)

print("Model used:", MODEL_EVENT_ID)
print("Distinct ticker/date observations:", len(unique_daily))
print("\nCombined election-episode CAR:")
print(
    combined[
        [
            "ticker",
            "trading_days",
            "combined_car",
            "combined_car_percent_approx",
        ]
    ].to_string(index=False)
)
print(f"\nSaved: {OUTPUT}")
