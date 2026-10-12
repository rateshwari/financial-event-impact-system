"""Safely merge preliminary manual labels into a new review dataset."""

from pathlib import Path
import pandas as pd

MASTER_PATH = Path("data/processed/manual_event_review.csv")
BATCH_PATH = Path("data/processed/event_review_batch1.csv")
OUTPUT_PATH = Path("data/processed/manual_event_review_merged.csv")

KEY = "article_url"
LABEL_COLUMNS = [
    "relevance_decision",
    "verified_target_ticker",
    "event_category",
    "event_description",
    "duplicate_cluster_id",
    "timestamp_confidence",
    "review_notes",
]


def is_nonempty(series):
    return series.notna() & series.astype("string").str.strip().ne("")


master = pd.read_csv(MASTER_PATH, dtype={KEY: "string"})
batch = pd.read_csv(BATCH_PATH, dtype={KEY: "string"})

for name, df in [("master", master), ("batch", batch)]:
    if KEY not in df.columns:
        raise ValueError(f"{name} is missing {KEY!r}.")
    if df[KEY].isna().any() or df[KEY].duplicated().any():
        raise ValueError(f"{name} has missing or duplicate article URLs.")

missing_columns = set(LABEL_COLUMNS) - set(batch.columns)
if missing_columns:
    raise ValueError(f"Batch is missing columns: {sorted(missing_columns)}")

master_urls = set(master[KEY])
batch_urls = set(batch[KEY])
unmatched = batch_urls - master_urls
if unmatched:
    raise ValueError(f"{len(unmatched)} batch URLs do not match the master.")

# Treat manual labels as text, including currently blank master fields.
for column in LABEL_COLUMNS:
    master[column] = master[column].astype("string")
    batch[column] = batch[column].astype("string")

master_indexed = master.set_index(KEY)
batch_indexed = batch.set_index(KEY)

existing = master_indexed.loc[list(batch_urls), LABEL_COLUMNS]
for column in LABEL_COLUMNS:
    if is_nonempty(existing[column]).any():
        raise ValueError(
            f"Master already contains values for {column!r}; "
            "review conflicts before merging."
        )

merged = master_indexed.copy()
for column in LABEL_COLUMNS:
    merged.loc[batch_indexed.index, column] = batch_indexed[column]

merged = merged.reset_index()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_PATH, index=False)

saved = pd.read_csv(OUTPUT_PATH, dtype={KEY: "string"})
if len(saved) != len(master) or saved[KEY].nunique() != len(master):
    raise ValueError("Saved output row count or URL uniqueness check failed.")

expected = (
    batch["relevance_decision"]
    .value_counts(dropna=False)
    .sort_index()
    .to_dict()
)

actual = (
    saved.loc[saved[KEY].isin(batch_urls), "relevance_decision"]
    .value_counts(dropna=False)
    .sort_index()
    .to_dict()
)

if expected != actual:
    raise ValueError(
        f"Decision counts differ.\nExpected:\n{expected}\nActual:\n{actual}"
    )

print(f"Master rows preserved: {len(saved)}")
print(f"Articles matched and merged: {len(batch_urls)}")
print(f"Articles not yet reviewed: {len(saved) - len(batch_urls)}")
print("\nMerged decision counts:")
print(saved["relevance_decision"].value_counts(dropna=False).to_string())
print(f"\nSaved new file: {OUTPUT_PATH}")
print("Original master and batch files were not modified.")
