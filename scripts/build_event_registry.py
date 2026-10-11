import pandas as pd
from pathlib import Path

SOURCE = Path("data/processed/event_review_batch1.csv")
OUTPUT = Path("data/processed/event_registry.csv")

articles = pd.read_csv(SOURCE)

events = [
    {
        "event_id": "EVT_2024_06_03_001",
        "event_date": "2024-06-03",
        "event_category": "market_wide",
        "target_ticker": "MARKET",
        "event_name": "Indian equity market rally following exit-poll projections",
        "event_cluster_id": "EXIT_POLL_RALLY_2024-06-03",
        "event_status": "provisional",
        "timing_notes": (
            "Candidate event date based on observed market reaction. "
            "Verify the first public-information timestamp before event study."
        ),
        "source_article_count": 3,
    },
    {
        "event_id": "EVT_2024_06_03_002",
        "event_date": "2024-06-03",
        "event_category": "corporate_event",
        "target_ticker": "SBIN",
        "event_name": "SBI reported market-capitalisation milestone",
        "event_cluster_id": "SBI_MCAP_MILESTONE_2024-06-03",
        "event_status": "review_required",
        "timing_notes": (
            "Outside the current five-stock universe. "
            "Do not include in the primary analysis without expanding the universe."
        ),
        "source_article_count": 1,
    },
]

df = pd.DataFrame(events)

df["event_date"] = pd.to_datetime(
    df["event_date"], errors="raise"
).dt.date

assert df["event_id"].is_unique, "Event IDs must be unique."
assert df["event_date"].notna().all(), "Missing event dates."

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Event registry created:", OUTPUT)
print("Total provisional/review events:", len(df))
print("\nEvents by status:")
print(df["event_status"].value_counts().to_string())
print("\nRegistry:")
print(df.to_string(index=False))