"""Baseline demand forecasting"""

import polars as pl


class DemandForecaster:
    """Simple weighted moving average forecast for baseline demand"""

    @staticmethod
    def compute_baseline_demand(inventory_df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute baseline 30-day demand forecast (no promotion)

        Uses weighted moving average based on:
        - Recent sales velocity
        - Days since last sale
        - Historical average
        """

        df = inventory_df.clone()

        # Baseline units for next 30 days
        # Simple approximation: avg_daily_units_30d * 30
        df = df.with_columns([
            (pl.col("avg_daily_units_30d") * 30).alias("baseline_units_30d")
        ])

        # Adjust for non-moving items
        df = df.with_columns([
            pl.when(pl.col("days_since_last_sale") > 60)
              .then(pl.col("baseline_units_30d") * 0.3)  # Reduce forecast
              .when(pl.col("days_since_last_sale") > 30)
              .then(pl.col("baseline_units_30d") * 0.6)
              .otherwise(pl.col("baseline_units_30d"))
              .alias("baseline_units_30d")
        ])

        # Adjust for returned items
        # Returns have lower natural demand (stigma) but higher discount response
        df = df.with_columns([
            pl.when(pl.col("is_returned") == True)
              .then(
                  pl.when(pl.col("return_condition") == "damaged")
                    .then(pl.col("baseline_units_30d") * 0.1)  # Very low natural demand
                  .when(pl.col("return_condition") == "opened")
                    .then(pl.col("baseline_units_30d") * 0.4)  # Reduced demand
                  .otherwise(pl.col("baseline_units_30d") * 0.7)  # Moderate reduction
              )
              .otherwise(pl.col("baseline_units_30d"))
              .alias("baseline_units_30d")
        ])

        # Ensure minimum of 0.1 for division safety
        df = df.with_columns([
            pl.col("baseline_units_30d").clip(0.1, None).alias("baseline_units_30d")
        ])

        return df
