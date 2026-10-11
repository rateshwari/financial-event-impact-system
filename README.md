# Financial Event Impact Detection System

Research question: Do identifiable financial news events correspond to statistically measurable abnormal stock returns, and does the relationship differ by event category?

## Build sequence
1. Validate and ingest historical news/prices.
2. Use PySpark to clean, transform, and write Parquet.
3. Apply FinBERT sentiment and event/entity extraction.
4. Run market-model event studies.
5. Persist curated results in PostgreSQL.
6. Expose results through FastAPI and visualize with Streamlit.
7. Add Kafka only after the offline pipeline is correct.

This project reports associations, not proof of causality. Keep model confidence, potential exposure, and observed market reaction separate.

## Local setup (macOS Apple Silicon, no Docker required)
Recommended: Python 3.11 and Java 17.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
pytest
```

Install `requirements-spark.txt`, `requirements-nlp.txt`, `requirements-api.txt`, and `requirements-dashboard.txt` only at their corresponding milestones.

## CSV schemas
`data/raw/news.csv`: `news_id,published_at,headline,source,url,ticker,event_type`
Required: `news_id,published_at,headline,source`. Use ISO-8601 timestamps with timezone.

`data/raw/prices.csv`: `date,ticker,close`
Use adjusted prices when possible; document the price convention.

## Baseline event study
- Map publication time to a valid trading session using exchange-calendar rules.
- Estimation window: trading sessions [-120, -21].
- Event window: [0, 1].
- Estimate expected returns using a market model.
- Abnormal return = observed stock return - expected return.
- CAR = sum of abnormal returns over the event window.

This is a baseline specification. Later work must consider overlapping events, confounders, multiple testing, economic significance, and robust inference.
