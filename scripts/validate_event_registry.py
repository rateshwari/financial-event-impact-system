import pandas as pd
from pathlib import Path

REGISTRY = Path("data/processed/event_registry_updated.csv")
RETURNS = Path("data/processed/daily_returns.csv")

events = pd.read_csv(REGISTRY, parse_dates=["event_date"])
returns = pd.read_csv(RETURNS, parse_dates=["date"])

returns["date"] = returns["date"].dt.normalize()
events["event_date"] = events["event_date"].dt.normalize()

trading_dates = sorted(returns["date"].dropna().unique())
date_to_index = {date: i for i, date in enumerate(trading_dates)}

results = []

for _, event in events.iterrows():
    event_date = event["event_date"]
    exact_match = event_date in date_to_index

    previous_dates = [d for d in trading_dates if d < event_date]
    next_dates = [d for d in trading_dates if d > event_date]

    results.append({
        "event_id": event["event_id"],
        "event_date": event_date.date(),
        "event_status": event["event_status"],
        "exact_trading_date": exact_match,
        "previous_trading_date": (
            previous_dates[-1].date() if previous_dates else None
        ),
        "next_trading_date": (
            next_dates[0].date() if next_dates else None
        ),
    })

result_df = pd.DataFrame(results)

print("Unique trading dates:", len(trading_dates))
print("Return dataset date range:",
      trading_dates[0].date(), "to", trading_dates[-1].date())
print("\nEvent-date validation:")
print(result_df.to_string(index=False))

output = Path("data/processed/event_date_validation.csv")
result_df.to_csv(output, index=False)
print("\nSaved:", output)
