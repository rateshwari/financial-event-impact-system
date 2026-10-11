import pandas as pd
from pathlib import Path

REGISTRY = Path("data/processed/event_registry.csv")
OUTPUT = Path("data/processed/event_registry_updated.csv")

df = pd.read_csv(REGISTRY)

event_id = "EVT_2024_06_04_001"

if event_id in df["event_id"].astype(str).values:
    raise ValueError(f"{event_id} already exists; refusing to duplicate it.")

new_event = {
    "event_id": event_id,
    "event_date": "2024-06-04",
    "event_category": "market_wide",
    "target_ticker": "MARKET",
    "event_name": "Indian equity market decline during election-result counting",
    "event_cluster_id": "ELECTION_RESULT_SELL_OFF_2024-06-04",
    "event_status": "provisional",
    "timing_notes": (
        "Two articles describe the same market episode. The Hindu timestamp "
        "is 2024-06-04 04:26:26 UTC; Devdiscourse is 2024-06-04 06:17:38 UTC. "
        "Verify the first public-information time and market reaction before "
        "finalizing event timing."
    ),
    "source_article_count": 2,
}

updated = pd.concat([df, pd.DataFrame([new_event])], ignore_index=True)

assert updated["event_id"].is_unique
assert len(updated) == len(df) + 1

updated.to_csv(OUTPUT, index=False)

print("Updated registry saved:", OUTPUT)
print("Previous events:", len(df))
print("Updated events:", len(updated))
print("\nEvent status counts:")
print(updated["event_status"].value_counts().to_string())
