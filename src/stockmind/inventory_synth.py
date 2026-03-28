"""Synthetic inventory generation module"""

import duckdb
import polars as pl
import numpy as np
from datetime import datetime, timedelta


class InventorySynthesizer:
    """Generate synthetic inventory data from products and order history"""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.snapshot_date = datetime.now().date()

    def generate_inventory_snapshot(self):
        """Generate synthetic inventory snapshot table"""

        print("Step 1/2: Loading products with sales data (optimized)...")
        # Combined query - much faster!
        combined_query = """
        SELECT
            p.product_id,
            p.product_category_name,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm,
            COALESCE(COUNT(DISTINCT oi.order_id), 0) as total_orders,
            COALESCE(COUNT(oi.order_id), 0) as total_units_sold,
            COALESCE(AVG(oi.price), 50.0) as avg_price,
            MAX(o.order_purchase_timestamp) as last_sale_date,
            MIN(o.order_purchase_timestamp) as first_sale_date
        FROM products p
        LEFT JOIN order_items oi ON p.product_id = oi.product_id
        LEFT JOIN orders o ON oi.order_id = o.order_id AND o.order_status = 'delivered'
        GROUP BY p.product_id, p.product_category_name, p.product_weight_g,
                 p.product_length_cm, p.product_height_cm, p.product_width_cm
        LIMIT 500
        """
        result = pl.read_database(combined_query, self.conn)
        print(f"  Loaded {len(result):,} products with sales data")

        print("Step 2/2: Generating synthetic inventory attributes...")
        # Generate synthetic inventory attributes
        inventory_df = self._generate_synthetic_attributes(result)

        # Save to DuckDB
        print("Saving to database...")
        self.conn.execute("DROP TABLE IF EXISTS inventory_snapshot")
        self.conn.execute("CREATE TABLE inventory_snapshot AS SELECT * FROM inventory_df")

        print(f"✓ Generated inventory snapshot with {len(inventory_df)} SKUs")
        return inventory_df

    def _generate_synthetic_attributes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add synthetic inventory attributes"""

        n = len(df)
        np.random.seed(42)

        # Calculate package volume
        df = df.with_columns([
            (
                pl.col("product_length_cm").fill_null(20) *
                pl.col("product_height_cm").fill_null(10) *
                pl.col("product_width_cm").fill_null(15)
            ).alias("package_volume_cm3")
        ])

        # Generate inventory quantities based on sales velocity
        df = df.with_columns([
            pl.when(pl.col("total_units_sold") == 0)
              .then(pl.lit(np.random.randint(5, 50, n)))  # Non-moving items
              .when(pl.col("total_units_sold") < 5)
              .then(pl.lit(np.random.randint(10, 100, n)))  # Slow-moving
              .otherwise(pl.lit(np.random.randint(50, 300, n)))  # Normal items
              .alias("on_hand_qty")
        ])

        # Calculate list price (add margin to cost)
        df = df.with_columns([
            (pl.col("avg_price") * 1.0).alias("list_price"),
            (pl.col("avg_price") * 0.65).alias("unit_cost"),  # 35% gross margin baseline
        ])

        # Generate inventory age (days since received)
        age_days = np.random.exponential(60, n).astype(int)
        age_days = np.clip(age_days, 1, 365)

        df = df.with_columns([
            pl.lit(age_days).alias("inventory_age_days"),
            pl.lit([
                (self.snapshot_date - timedelta(days=int(age))).isoformat()
                for age in age_days
            ]).alias("received_date")
        ])

        # Assign storage class based on volume
        df = df.with_columns([
            pl.when(pl.col("package_volume_cm3") < 5000)
              .then(pl.lit("small"))
              .when(pl.col("package_volume_cm3") < 20000)
              .then(pl.lit("medium"))
              .when(pl.col("package_volume_cm3") < 50000)
              .then(pl.lit("large"))
              .otherwise(pl.lit("bulky"))
              .alias("storage_class")
        ])

        # Shelf life by category (simplified)
        df = df.with_columns([
            pl.when(pl.col("product_category_name").is_in(["alimentos", "bebidas"]))
              .then(pl.lit(90))  # Perishable
              .when(pl.col("product_category_name").is_in(["perfumaria", "beleza_saude"]))
              .then(pl.lit(365))  # Cosmetics
              .otherwise(pl.lit(730))  # Durable goods
              .alias("shelf_life_days")
        ])

        # Holding cost per unit per day (function of unit cost and storage class)
        df = df.with_columns([
            pl.when(pl.col("storage_class") == "small")
              .then(pl.col("unit_cost") * 0.0003)
              .when(pl.col("storage_class") == "medium")
              .then(pl.col("unit_cost") * 0.0005)
              .when(pl.col("storage_class") == "large")
              .then(pl.col("unit_cost") * 0.0008)
              .otherwise(pl.col("unit_cost") * 0.0012)
              .alias("holding_cost_per_unit_per_day")
        ])

        # Discount sensitivity (how responsive to promotions)
        discount_sensitivity = np.random.uniform(0.5, 1.5, n)
        df = df.with_columns([
            pl.lit(discount_sensitivity).alias("discount_sensitivity")
        ])

        # Seasonality class
        seasonality_classes = np.random.choice(
            ["stable", "seasonal", "trending"],
            size=n,
            p=[0.6, 0.3, 0.1]
        )
        df = df.with_columns([
            pl.lit(seasonality_classes).alias("seasonality_class")
        ])

        # Returns data - 15% of inventory are returned items
        is_returned = np.random.choice([True, False], size=n, p=[0.15, 0.85])

        # Return condition for returned items
        return_conditions = []
        for returned in is_returned:
            if returned:
                # 60% good condition, 30% opened, 10% damaged
                condition = np.random.choice(
                    ["good", "opened", "damaged"],
                    p=[0.60, 0.30, 0.10]
                )
            else:
                condition = "new"
            return_conditions.append(condition)

        # Return processing cost (sunk cost already incurred)
        # Good: $2-5, Opened: $3-8, Damaged: $5-15
        return_processing_costs = []
        for condition in return_conditions:
            if condition == "good":
                cost = np.random.uniform(2, 5)
            elif condition == "opened":
                cost = np.random.uniform(3, 8)
            elif condition == "damaged":
                cost = np.random.uniform(5, 15)
            else:
                cost = 0.0
            return_processing_costs.append(cost)

        # Returned date (only for returned items, within last 90 days)
        returned_dates = []
        for returned in is_returned:
            if returned:
                days_ago = np.random.randint(1, 90)
                date = (self.snapshot_date - timedelta(days=days_ago)).isoformat()
            else:
                date = None
            returned_dates.append(date)

        df = df.with_columns([
            pl.Series("is_returned", is_returned),
            pl.Series("return_condition", return_conditions),
            pl.Series("return_processing_cost", return_processing_costs),
            pl.Series("returned_date", returned_dates)
        ])

        # Calculate days since last sale
        df = df.with_columns([
            pl.when(pl.col("last_sale_date").is_null())
              .then(pl.lit(999))
              .otherwise(
                  ((pl.lit(self.snapshot_date) - pl.col("last_sale_date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").cast(pl.Date)).dt.total_days())
              )
              .alias("days_since_last_sale")
        ])

        # Add snapshot date
        df = df.with_columns([
            pl.lit(self.snapshot_date.isoformat()).alias("snapshot_date")
        ])

        return df


if __name__ == "__main__":
    from ingest import DataIngestor

    # Load data first
    ingestor = DataIngestor()
    conn = ingestor.connect()

    # Generate inventory
    synth = InventorySynthesizer(conn)
    inventory = synth.generate_inventory_snapshot()

    print("\nSample inventory data:")
    print(inventory.head(10))

    ingestor.close()
