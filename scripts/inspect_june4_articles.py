import pandas as pd
from pathlib import Path

SOURCE = Path("data/processed/manual_event_review.csv")

df = pd.read_csv(SOURCE)

df["publication_date"] = pd.to_datetime(
    df["publication_date"], errors="coerce", utc=True
)

mask = (
    df["publication_date"].dt.strftime("%Y-%m-%d").eq("2024-06-04")
    & df["headline"].fillna("").str.contains(
        r"election|sensex|nifty|market crash|market falls|stocks fall|shares fall",
        case=False,
        regex=True,
    )
)

selected = df.loc[mask].copy()

columns = [
    "publisher",
    "headline",
    "publication_date",
    "publication_date_source",
    "article_url",
    "text_characters",
]

print("June 4 matching articles:", len(selected))
print()

if selected.empty:
    print("No matching articles found in the current review file.")
else:
    print(
        selected[columns]
        .sort_values("publication_date")
        .to_string(index=False)
    )
