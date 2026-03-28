"""Promotion recommendation engine"""

import polars as pl
import numpy as np


class PromotionEngine:
    """
    Core engine for markdown recommendation
    Simulates discount ladder and selects optimal action
    """

    # Default promotion ladder
    DISCOUNT_LADDER = [0.0, 0.05, 0.10, 0.15, 0.20]

    # Default uplift factors for each discount level
    DEFAULT_UPLIFT = {
        0.00: 1.00,
        0.05: 1.08,
        0.10: 1.18,
        0.15: 1.32,
        0.20: 1.50,
    }

    def __init__(
        self,
        distress_threshold: float = 0.55,
        min_margin_pct: float = 0.15,
        low_stock_threshold: float = 0.75,
        uplift_factors: dict = None,
        weights: dict = None
    ):
        """
        Initialize promotion engine with configurable parameters

        Args:
            distress_threshold: Minimum distress score to consider promotion
            min_margin_pct: Minimum acceptable gross margin
            low_stock_threshold: Don't promote if inventory < baseline * threshold
            uplift_factors: Custom uplift multipliers for each discount
            weights: Custom weights for distress score calculation
        """
        self.distress_threshold = distress_threshold
        self.min_margin_pct = min_margin_pct
        self.low_stock_threshold = low_stock_threshold
        self.uplift_factors = uplift_factors or self.DEFAULT_UPLIFT
        self.weights = weights

    def simulate_promotions(self, inventory_df: pl.DataFrame) -> pl.DataFrame:
        """
        For each SKU, simulate all discount levels and recommend best action

        Returns DataFrame with recommendation per SKU
        """

        results = []

        for row in inventory_df.iter_rows(named=True):
            recommendation = self._evaluate_sku(row)
            results.append(recommendation)

        results_df = pl.DataFrame(results)
        return results_df

    def _evaluate_sku(self, sku: dict) -> dict:
        """Evaluate single SKU and recommend best action"""

        product_id = sku["product_id"]
        distress_score = sku["distress_score"]
        on_hand = sku["on_hand_qty"]
        baseline_demand = sku["baseline_units_30d"]
        list_price = sku["list_price"]
        unit_cost = sku["unit_cost"]
        holding_cost_daily = sku["holding_cost_per_unit_per_day"]
        decay_score = sku["decay_score"]
        gross_margin_pct = sku["gross_margin_pct"]
        discount_sensitivity = sku["discount_sensitivity"]
        days_of_cover = sku.get("days_of_cover", 999)

        # Apply business guard rails first
        if distress_score < self.distress_threshold:
            return self._create_recommendation(
                product_id, 0.0, 0, sku,
                reason="Below distress threshold"
            )

        if on_hand < baseline_demand * self.low_stock_threshold:
            return self._create_recommendation(
                product_id, 0.0, 0, sku,
                reason="Low stock relative to demand"
            )

        # Simulate each discount level
        best_action = None
        best_score = float('-inf')
        action_details = []

        for discount_pct in self.DISCOUNT_LADDER:
            # Check margin constraint
            discounted_price = list_price * (1 - discount_pct)
            margin_after_discount = (discounted_price - unit_cost) / discounted_price

            if margin_after_discount < self.min_margin_pct and discount_pct > 0:
                continue  # Skip if margin too thin

            # Calculate expected results
            action_result = self._calculate_action_economics(
                discount_pct=discount_pct,
                baseline_demand=baseline_demand,
                on_hand=on_hand,
                list_price=list_price,
                unit_cost=unit_cost,
                holding_cost_daily=holding_cost_daily,
                decay_score=decay_score,
                discount_sensitivity=discount_sensitivity
            )

            action_details.append({
                "discount": discount_pct,
                **action_result
            })

            # Track best action
            if action_result["action_score"] > best_score:
                best_score = action_result["action_score"]
                best_action = (discount_pct, action_result)

        # Select best action
        if best_action:
            discount_pct, economics = best_action
            return self._create_recommendation(
                product_id, discount_pct, best_score, sku,
                economics=economics,
                all_actions=action_details
            )
        else:
            return self._create_recommendation(
                product_id, 0.0, 0, sku,
                reason="No viable action (margin constraints)"
            )

    def _calculate_action_economics(
        self,
        discount_pct: float,
        baseline_demand: float,
        on_hand: int,
        list_price: float,
        unit_cost: float,
        holding_cost_daily: float,
        decay_score: float,
        discount_sensitivity: float
    ) -> dict:
        """Calculate economics for a specific discount action"""

        # Get uplift factor
        uplift = self.uplift_factors.get(discount_pct, 1.0)

        # Adjust uplift by SKU discount sensitivity
        adjusted_uplift = 1.0 + (uplift - 1.0) * discount_sensitivity

        # Expected units sold
        expected_units = baseline_demand * adjusted_uplift
        units_sold = min(expected_units, on_hand)

        # Discounted price
        discounted_price = list_price * (1 - discount_pct)

        # Expected margin
        expected_margin = (discounted_price - unit_cost) * units_sold

        # Holding cost avoided (assume 30 days)
        holding_cost_avoided = units_sold * holding_cost_daily * 30

        # Waste/decay penalty avoided
        waste_penalty_per_unit = unit_cost * 0.5 * decay_score
        waste_avoided = units_sold * waste_penalty_per_unit

        # Leftover penalty
        leftover_units = max(on_hand - units_sold, 0)
        future_risk_cost = holding_cost_daily * 60 + waste_penalty_per_unit
        leftover_penalty = leftover_units * future_risk_cost

        # Total action score
        action_score = (
            expected_margin +
            holding_cost_avoided +
            waste_avoided -
            leftover_penalty
        )

        return {
            "expected_units": expected_units,
            "units_sold": units_sold,
            "expected_margin": expected_margin,
            "holding_cost_avoided": holding_cost_avoided,
            "waste_avoided": waste_avoided,
            "leftover_penalty": leftover_penalty,
            "action_score": action_score,
            "leftover_units": leftover_units
        }

    def _create_recommendation(
        self,
        product_id: str,
        discount_pct: float,
        action_score: float,
        sku_data: dict,
        reason: str = None,
        economics: dict = None,
        all_actions: list = None
    ) -> dict:
        """Create recommendation record"""

        rec = {
            "product_id": product_id,
            "distress_score": sku_data["distress_score"],
            "distress_category": sku_data["distress_category"],
            "on_hand_qty": sku_data["on_hand_qty"],
            "days_since_last_sale": sku_data["days_since_last_sale"],
            "days_of_cover": sku_data.get("days_of_cover", 0),
            "recommended_discount_pct": discount_pct,
            "recommended_action": f"{int(discount_pct * 100)}% off" if discount_pct > 0 else "No action",
            "action_score": action_score,
        }

        if economics:
            rec.update({
                "expected_units_no_action": sku_data["baseline_units_30d"],
                "expected_units_with_action": economics["units_sold"],
                "expected_margin": economics["expected_margin"],
                "inventory_relief": economics["units_sold"],
                "leftover_units": economics["leftover_units"],
            })
        else:
            rec.update({
                "expected_units_no_action": sku_data["baseline_units_30d"],
                "expected_units_with_action": 0,
                "expected_margin": 0,
                "inventory_relief": 0,
                "leftover_units": sku_data["on_hand_qty"],
            })

        if reason:
            rec["reason"] = reason
        else:
            # Generate reason codes
            reasons = []
            if sku_data.get("non_moving_score", 0) > 0.6:
                reasons.append("non-moving")
            if sku_data.get("aging_score", 0) > 0.6:
                reasons.append("aging inventory")
            if sku_data.get("days_of_cover", 0) > 90:
                reasons.append("high days of cover")
            if sku_data.get("storage_score", 0) > 0.7:
                reasons.append("high storage burden")
            if sku_data.get("decay_score", 0) > 0.5:
                reasons.append("elevated decay risk")

            rec["reason"] = ", ".join(reasons) if reasons else "general distress"

        return rec
