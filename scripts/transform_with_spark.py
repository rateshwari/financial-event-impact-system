from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    DateType,
    StringType,
    DoubleType,
)


ROOT = Path(__file__).resolve().parents[1]

PRICES_PATH = str(ROOT / "data/processed/prices_clean.csv")
BASELINE_PATH = str(ROOT / "data/processed/daily_returns.csv")

RETURNS_OUTPUT = str(ROOT / "data/processed/daily_returns_parquet")
SUMMARY_OUTPUT = str(ROOT / "data/processed/ticker_return_summary_parquet")

EXPECTED_TICKERS = 6
EXPECTED_PRICE_ROWS_PER_TICKER = 1178
EXPECTED_RETURN_ROWS_PER_TICKER = 1177
RETURN_TOLERANCE = 1e-10


def main():
    spark = (
        SparkSession.builder
        .appName("FinancialEventImpactSystem")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\nSTEP 1: Read and validate cleaned prices")

        price_schema = StructType([
            StructField("date", DateType(), True),
            StructField("ticker", StringType(), True),
            StructField("close", DoubleType(), True),
        ])

        prices = (
            spark.read
            .option("header", True)
            .schema(price_schema)
            .csv(PRICES_PATH)
        )

        if prices.filter(
            F.col("date").isNull()
            | F.col("ticker").isNull()
            | (F.trim(F.col("ticker")) == "")
            | F.col("close").isNull()
            | F.isnan("close")
            | (F.col("close") <= 0)
        ).limit(1).count():
            raise ValueError(
                "Prices contain null, invalid, or non-positive values."
            )

        duplicate_prices = (
            prices.groupBy("date", "ticker")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        if duplicate_prices:
            raise ValueError(
                f"Found {duplicate_prices} duplicate date/ticker pairs."
            )

        tickers = prices.select("ticker").distinct().count()
        if tickers != EXPECTED_TICKERS:
            raise ValueError(
                f"Expected {EXPECTED_TICKERS} tickers, found {tickers}."
            )

        price_counts = prices.groupBy("ticker").agg(
            F.count("*").alias("rows"),
            F.countDistinct("date").alias("unique_dates"),
        )

        price_counts.show(truncate=False)

        if price_counts.filter(
            (F.col("rows") != EXPECTED_PRICE_ROWS_PER_TICKER)
            | (F.col("unique_dates") != EXPECTED_PRICE_ROWS_PER_TICKER)
        ).limit(1).count():
            raise ValueError(
                "Each ticker must have 1,178 unique price dates."
            )

        # Every trading date must contain all six symbols.
        incomplete_dates = (
            prices.groupBy("date")
            .agg(F.countDistinct("ticker").alias("ticker_count"))
            .filter(F.col("ticker_count") != EXPECTED_TICKERS)
            .count()
        )

        if incomplete_dates:
            raise ValueError(
                f"Found {incomplete_dates} dates without all six tickers."
            )

        print("Price validation passed.")

        print("\nSTEP 2: Calculate daily returns with a Spark window")

        window = Window.partitionBy("ticker").orderBy("date")

        returns = (
            prices
            .withColumn(
                "previous_close",
                F.lag("close").over(window),
            )
            .withColumn(
                "daily_return",
                F.col("close") / F.col("previous_close") - F.lit(1.0),
            )
            .filter(F.col("previous_close").isNotNull())
            .select("date", "ticker", "close", "daily_return")
            .cache()
        )

        return_count = returns.count()
        print("Calculated return rows:", return_count)

        return_counts = returns.groupBy("ticker").count()
        return_counts.show(truncate=False)

        if return_count != EXPECTED_TICKERS * EXPECTED_RETURN_ROWS_PER_TICKER:
            raise ValueError(
                f"Expected 7,062 return rows; got {return_count}."
            )

        if return_counts.filter(
            F.col("count") != EXPECTED_RETURN_ROWS_PER_TICKER
        ).limit(1).count():
            raise ValueError(
                "Each ticker must have 1,177 daily returns."
            )

        invalid_returns = returns.filter(
            F.col("daily_return").isNull()
            | F.isnan("daily_return")
            | (F.abs(F.col("daily_return")) == float("inf"))
        ).count()

        if invalid_returns:
            raise ValueError(
                f"Found {invalid_returns} invalid daily returns."
            )

        print("Return calculation passed.")

        print("\nSTEP 3: Compare against the existing CSV baseline")

        baseline_schema = StructType([
            StructField("date", DateType(), True),
            StructField("ticker", StringType(), True),
            StructField("close", DoubleType(), True),
            StructField("daily_return", DoubleType(), True),
        ])

        baseline = (
            spark.read
            .option("header", True)
            .schema(baseline_schema)
            .csv(BASELINE_PATH)
            .select(
                "date",
                "ticker",
                F.col("daily_return").alias("baseline_return"),
            )
        )

        spark_returns = returns.select(
            "date",
            "ticker",
            F.col("daily_return").alias("spark_return"),
        )

        comparison = spark_returns.join(
            baseline,
            on=["date", "ticker"],
            how="full_outer",
        )

        mismatches = comparison.filter(
            F.col("spark_return").isNull()
            | F.col("baseline_return").isNull()
            | (
                F.abs(
                    F.col("spark_return") - F.col("baseline_return")
                ) > RETURN_TOLERANCE
            )
        )

        mismatch_count = mismatches.count()

        print("Baseline rows:", baseline.count())
        print("Mismatched return observations:", mismatch_count)

        if mismatch_count:
            mismatches.show(20, truncate=False)
            raise ValueError(
                "Spark returns differ from the CSV baseline. "
                "Outputs were not written."
            )

        print("Baseline comparison passed.")

        print("\nSTEP 4: Generate per-ticker statistics")

        summary = (
            returns.groupBy("ticker")
            .agg(
                F.count("*").alias("observation_count"),
                F.avg("daily_return").alias("mean_daily_return"),
                F.stddev_samp("daily_return").alias("daily_volatility"),
                F.min("daily_return").alias("minimum_daily_return"),
                F.max("daily_return").alias("maximum_daily_return"),
            )
            .orderBy("ticker")
        )

        summary.show(truncate=False)

        print("\nSTEP 5: Write Parquet outputs")

        (
            returns.write
            .mode("overwrite")
            .parquet(RETURNS_OUTPUT)
        )

        (
            summary.write
            .mode("overwrite")
            .parquet(SUMMARY_OUTPUT)
        )

        # Read the written Parquet back and verify its row count.
        parquet_check = spark.read.parquet(RETURNS_OUTPUT)
        parquet_rows = parquet_check.count()

        if parquet_rows != return_count:
            raise ValueError(
                f"Parquet verification failed: expected {return_count}, "
                f"read back {parquet_rows}."
            )

        summary_rows = spark.read.parquet(SUMMARY_OUTPUT).count()

        if summary_rows != EXPECTED_TICKERS:
            raise ValueError(
                f"Expected {EXPECTED_TICKERS} summary rows; "
                f"found {summary_rows}."
            )

        print("\nSUCCESS: Spark pipeline completed.")
        print("Spark return rows:", return_count)
        print("Parquet return rows:", parquet_rows)
        print("Summary rows:", summary_rows)
        print("Returns output:", RETURNS_OUTPUT)
        print("Summary output:", SUMMARY_OUTPUT)

        returns.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
