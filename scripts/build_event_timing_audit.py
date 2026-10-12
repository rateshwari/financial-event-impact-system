"""Create an auditable review template for event timestamps and eligibility."""

from pathlib import Path

import pandas as pd

REGISTRY = Path("data/processed/event_registry_updated.csv")
OUTPUT = Path("data/processed/event_timing_audit.csv")

REQUIRED_COLUMNS = {
    "event_id",
    "event_date",
    "event_category",
    "target_ticker",
    "event_name",
    "event_status",
    "timing_notes",
}

events = pd.read_csv(REGISTRY, dtype={"event_id": str})

missing = REQUIRED_COLUMNS - set(events.columns)
if missing:
    raise ValueError(f"Registry is missing required columns: {sorted(missing)}")

if events["event_id"].isna().any() or events["event_id"].duplicated().any():
    raise ValueError("Event IDs must be present and unique.")

audit = events[
    [
        "event_id",
        "event_date",
        "event_category",
        "target_ticker",
        "event_name",
        "event_status",
        "timing_notes",
    ]
].copy()

# These fields require source verification. Do not infer them from
# market reactions or later syndicated coverage.
audit["first_public_timestamp"] = ""
audit["timestamp_timezone"] = ""
audit["timestamp_source_url"] = ""
audit["timestamp_evidence"] = ""
audit["timestamp_confidence"] = "unresolved"
audit["event_date_basis"] = "unverified"
audit["primary_analysis_eligible"] = "pending_review"
audit["exclusion_reason"] = ""

audit.to_csv(OUTPUT, index=False)

print(f"Events audited: {len(audit)}")
print(f"Output: {OUTPUT}")

print("\nReview template:")
print(
    audit[
        [
            "event_id",
            "event_date",
            "event_name",
            "timestamp_confidence",
            "event_date_basis",
            "primary_analysis_eligible",
        ]
    ].to_string(index=False)
)
