# Data dictionary

## Raw prices (`data/raw/prices.csv`)
| Column | Type | Meaning |
|---|---|---|
| date | YYYY-MM-DD | Exchange trading date |
| ticker | string | Yahoo Finance symbol; `^NSEI` is the NIFTY 50 benchmark |
| close | float | Daily `Close` from yfinance with `auto_adjust=True`, therefore adjusted for corporate actions |

## Raw news (`data/raw/news.csv`)
| Column | Meaning |
|---|---|
| news_id | Stable source ID or reproducible hash |
| published_at | ISO-8601 publication timestamp, timezone-aware when possible |
| headline | Original headline text |
| source | Publisher |
| url | Canonical article URL |
| ticker | Company ticker only when supported by the source/mapping evidence |
| event_type | Optional human-verified event label for evaluation |

Do not infer publication timestamps from file dates. Do not invent labels, tickers,
or timestamps. Keep source and collection metadata for reproducibility.
