"""Feature engineering for distress scoring"""

import numpy as np
import polars as pl


class FeatureEngine:
    """Compute distress metrics and scores for inventory"""

    @staticmethod
    def compute_distress_features(inventory_df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all distress-related features for inventory analysis

        Features computed:
        - non_moving_score
        - overstock_score
        - aging_score
        - decay_score
        - storage_score
        - demand_weakness_score
        - margin_buffer_score
        - overall distress_score
        """

        df = inventory_df.clone()

        # 1. Non-moving score (based on days since last sale)
        df = df.with_columns(
            [
                (pl.col("days_since_last_sale") / 60.0)
                .clip(0, 1)
                .alias("non_moving_score")
            ]
        )

        # 2. Calculate avg daily units sold (last 30 days approximation)
        df = df.with_columns(
            [
                (
                    pl.col("total_units_sold")
                    / pl.max_horizontal(
                        [
                            (pl.lit(365) - pl.col("days_since_last_sale")).clip(1, 365),
                            pl.lit(30),
                        ]
                    )
                )
                .clip(0.01, None)
                .alias("avg_daily_units_30d")
            ]
        )

        # 3. Days of cover
        df = df.with_columns(
            [
                (pl.col("on_hand_qty") / pl.col("avg_daily_units_30d")).alias(
                    "days_of_cover"
                )
            ]
        )

        # 4. Overstock score
        df = df.with_columns(
            [(pl.col("days_of_cover") / 120.0).clip(0, 1).alias("overstock_score")]
        )

        # 5. Aging score
        df = df.with_columns(
            [(pl.col("inventory_age_days") / 180.0).clip(0, 1).alias("aging_score")]
        )

        # 6. Decay score (aging ratio vs shelf life)
        df = df.with_columns(
            [
                (pl.col("inventory_age_days") / pl.col("shelf_life_days"))
                .clip(0, 1)
                .alias("decay_score")
            ]
        )

        # 7. Storage burden score
        df = df.with_columns(
            [
                (pl.col("package_volume_cm3") * pl.col("on_hand_qty")).alias(
                    "storage_burden"
                )
            ]
        )

        # Normalize storage burden to 0-1 scale
        max_burden = df.select(pl.col("storage_burden").max()).item()
        df = df.with_columns(
            [(pl.col("storage_burden") / max_burden).clip(0, 1).alias("storage_score")]
        )

        # 8. Demand weakness score (recent velocity vs historical)
        # Approximate: if days since last sale is low, velocity is good
        df = df.with_columns(
            [
                pl.when(pl.col("days_since_last_sale") > 30)
                .then(pl.lit(0.8))
                .when(pl.col("days_since_last_sale") > 14)
                .then(pl.lit(0.5))
                .otherwise(pl.lit(0.2))
                .alias("demand_weakness_score")
            ]
        )

        # 9. Gross margin percentage
        df = df.with_columns(
            [
                (
                    (pl.col("list_price") - pl.col("unit_cost")) / pl.col("list_price")
                ).alias("gross_margin_pct")
            ]
        )

        # 10. Margin buffer score
        df = df.with_columns(
            [
                (pl.col("gross_margin_pct") / 0.60)
                .clip(0, 1)
                .alias("margin_buffer_score")
            ]
        )

        # 11. Return liability score
        # Returned items are automatic liability - must move
        df = df.with_columns(
            [
                pl.when(pl.col("is_returned") == True)
                .then(
                    # Base return liability
                    pl.when(pl.col("return_condition") == "damaged")
                    .then(pl.lit(0.95))  # Damaged = urgent
                    .when(pl.col("return_condition") == "opened")
                    .then(pl.lit(0.75))  # Opened = high priority
                    .otherwise(pl.lit(0.60))  # Good condition = moderate priority
                )
                .otherwise(pl.lit(0.0))  # Not returned = no return liability
                .alias("return_liability_score")
            ]
        )

        # 12. Overall Distress Score (weighted combination)
        # Returned items get automatic boost to distress
        df = df.with_columns(
            [
                pl.when(pl.col("is_returned"))
                .then(
                    (
                        # Returned items: return liability dominates
                        0.40
                        * pl.col("return_liability_score")  # Returns = top priority
                        + 0.15 * pl.col("non_moving_score")
                        + 0.10 * pl.col("overstock_score")
                        + 0.10 * pl.col("aging_score")
                        + 0.10 * pl.col("decay_score")
                        + 0.08 * pl.col("storage_score")
                        + 0.07 * pl.col("demand_weakness_score")
                    )
                )
                .otherwise(
                    (
                        # Normal items: standard distress calculation
                        0.25 * pl.col("non_moving_score")
                        + 0.20 * pl.col("overstock_score")
                        + 0.20 * pl.col("aging_score")
                        + 0.15 * pl.col("decay_score")
                        + 0.10 * pl.col("storage_score")
                        + 0.10 * pl.col("demand_weakness_score")
                    )
                )
                .alias("distress_score")
            ]
        )

        # 13. Distress category
        df = df.with_columns(
            [
                pl.when(pl.col("distress_score") < 0.35)
                .then(pl.lit("healthy"))
                .when(pl.col("distress_score") < 0.55)
                .then(pl.lit("watch"))
                .when(pl.col("distress_score") < 0.75)
                .then(pl.lit("at_risk"))
                .otherwise(pl.lit("urgent"))
                .alias("distress_category")
            ]
        )

        return df

    @staticmethod
    def compute_with_custom_weights(
        inventory_df: pl.DataFrame, weights: dict = None
    ) -> pl.DataFrame:
        """
        Compute distress score with custom weights

        Args:
            inventory_df: Inventory dataframe with base features
            weights: Dict with keys matching score types
                    For new items: non_moving, overstock, aging, decay, storage, demand_weakness
                    For returned items: adds return_liability (automatically gets 0.40 weight)
        """
        if weights is None:
            # Default weights for new items
            weights = {
                "non_moving": 0.25,
                "overstock": 0.20,
                "aging": 0.20,
                "decay": 0.15,
                "storage": 0.10,
                "demand_weakness": 0.10,
            }

        df = inventory_df.clone()

        # Recompute distress score with custom weights
        # Different formula for returned vs new items
        df = df.with_columns(
            [
                pl.when(pl.col("is_returned"))
                .then(
                    (
                        # Returned items: return liability dominates
                        weights.get("return_liability", 0.40)
                        * pl.col("return_liability_score")
                        + weights.get("non_moving", 0.15) * pl.col("non_moving_score")
                        + weights.get("overstock", 0.10) * pl.col("overstock_score")
                        + weights.get("aging", 0.10) * pl.col("aging_score")
                        + weights.get("decay", 0.10) * pl.col("decay_score")
                        + weights.get("storage", 0.08) * pl.col("storage_score")
                        + weights.get("demand_weakness", 0.07)
                        * pl.col("demand_weakness_score")
                    )
                )
                .otherwise(
                    (
                        # Normal items: standard weights
                        weights.get("non_moving", 0.25) * pl.col("non_moving_score")
                        + weights.get("overstock", 0.20) * pl.col("overstock_score")
                        + weights.get("aging", 0.20) * pl.col("aging_score")
                        + weights.get("decay", 0.15) * pl.col("decay_score")
                        + weights.get("storage", 0.10) * pl.col("storage_score")
                        + weights.get("demand_weakness", 0.10)
                        * pl.col("demand_weakness_score")
                    )
                )
                .alias("distress_score")
            ]
        )

        # Update distress category
        df = df.with_columns(
            [
                pl.when(pl.col("distress_score") < 0.35)
                .then(pl.lit("healthy"))
                .when(pl.col("distress_score") < 0.55)
                .then(pl.lit("watch"))
                .when(pl.col("distress_score") < 0.75)
                .then(pl.lit("at_risk"))
                .otherwise(pl.lit("urgent"))
                .alias("distress_category")
            ]
        )

        return df
