# Architecture

```text
Historical news + prices
        |
Validation / deduplication
        |
PySpark transformations -----> Parquet data lake
        |
FinBERT + event taxonomy + company mapping
        |
Market model / abnormal returns / CAR
        |
PostgreSQL -> FastAPI -> Streamlit

Later: Kafka -> Spark Structured Streaming -> curated pipeline
```

Keep raw records immutable. Preserve source URLs and publication timestamps. Track model/taxonomy versions. Use stable company identifiers and validate ticker ambiguity. Store observed returns separately from exposure estimates and predictions.
