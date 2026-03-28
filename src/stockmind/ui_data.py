"""Data preparation utilities for UI"""

import duckdb
import polars as pl


class UIDataProvider:
    """Provides data for UI components"""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def get_orders(self, limit: int = 1000, offset: int = 0) -> pl.DataFrame:
        """Get orders with customer info"""
        query = f"""
        SELECT
            o.order_id,
            o.customer_id,
            c.customer_city,
            c.customer_state,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_delivered_customer_date,
            COUNT(DISTINCT oi.product_id) as total_items,
            SUM(oi.price) as total_amount
        FROM orders o
        LEFT JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN order_items oi ON o.order_id = oi.order_id
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        ORDER BY o.order_purchase_timestamp DESC
        LIMIT {limit} OFFSET {offset}
        """
        return pl.read_database(query, self.conn)

    def get_products(self, limit: int = 1000, offset: int = 0, category: str = None) -> pl.DataFrame:
        """Get products with sales info"""
        where_clause = ""
        if category and category != "All":
            where_clause = f"WHERE p.product_category_name = '{category}'"

        query = f"""
        SELECT
            p.product_id,
            p.product_category_name,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm,
            COUNT(DISTINCT oi.order_id) as total_orders,
            COALESCE(SUM(oi.price), 0) as total_revenue
        FROM products p
        LEFT JOIN order_items oi ON p.product_id = oi.product_id
        {where_clause}
        GROUP BY 1, 2, 3, 4, 5, 6
        ORDER BY total_revenue DESC
        LIMIT {limit} OFFSET {offset}
        """
        return pl.read_database(query, self.conn)

    def get_customers(self, limit: int = 1000, offset: int = 0) -> pl.DataFrame:
        """Get customers with order stats"""
        query = f"""
        SELECT
            c.customer_id,
            c.customer_unique_id,
            c.customer_city,
            c.customer_state,
            COUNT(DISTINCT o.order_id) as total_orders,
            SUM(oi.price) as total_spent
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        LEFT JOIN order_items oi ON o.order_id = oi.order_id
        GROUP BY 1, 2, 3, 4
        ORDER BY total_spent DESC NULLS LAST
        LIMIT {limit} OFFSET {offset}
        """
        return pl.read_database(query, self.conn)

    def get_inventory(self, limit: int = 1000, offset: int = 0) -> pl.DataFrame:
        """Get inventory snapshot"""
        query = f"""
        SELECT * FROM inventory_snapshot
        ORDER BY distress_score DESC
        LIMIT {limit} OFFSET {offset}
        """
        try:
            return pl.read_database(query, self.conn)
        except Exception:
            # Table might not exist yet
            return pl.DataFrame()

    def get_categories(self) -> list:
        """Get list of product categories"""
        query = "SELECT DISTINCT product_category_name FROM products WHERE product_category_name IS NOT NULL ORDER BY 1"
        result = self.conn.execute(query).fetchall()
        return ["All"] + [row[0] for row in result if row[0]]

    def get_inventory_summary_stats(self) -> dict:
        """Get summary statistics for inventory"""
        try:
            query = """
            SELECT
                COUNT(*) as total_skus,
                SUM(on_hand_qty) as total_units,
                SUM(CASE WHEN distress_category = 'urgent' THEN 1 ELSE 0 END) as urgent_count,
                SUM(CASE WHEN distress_category = 'at_risk' THEN 1 ELSE 0 END) as at_risk_count,
                SUM(CASE WHEN distress_category = 'watch' THEN 1 ELSE 0 END) as watch_count,
                SUM(CASE WHEN distress_category = 'healthy' THEN 1 ELSE 0 END) as healthy_count,
                AVG(distress_score) as avg_distress_score
            FROM inventory_snapshot
            """
            result = self.conn.execute(query).fetchone()
            return {
                "total_skus": result[0] or 0,
                "total_units": result[1] or 0,
                "urgent": result[2] or 0,
                "at_risk": result[3] or 0,
                "watch": result[4] or 0,
                "healthy": result[5] or 0,
                "avg_distress": result[6] or 0
            }
        except Exception:
            return {
                "total_skus": 0,
                "total_units": 0,
                "urgent": 0,
                "at_risk": 0,
                "watch": 0,
                "healthy": 0,
                "avg_distress": 0
            }
