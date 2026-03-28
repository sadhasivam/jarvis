"""Data ingestion module - Load CSV data into DuckDB"""

import duckdb
import polars as pl
from pathlib import Path


class DataIngestor:
    """Handles loading Olist CSV data into DuckDB"""

    def __init__(self, db_path: str = "jarvis.db", data_dir: str = "data/raw/olist"):
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.conn = None

    def connect(self):
        """Connect to DuckDB"""
        self.conn = duckdb.connect(self.db_path)
        return self.conn

    def close(self):
        """Close DuckDB connection"""
        if self.conn:
            self.conn.close()

    def load_csv_to_table(self, csv_name: str, table_name: str):
        """Load a CSV file into DuckDB table"""
        csv_path = self.data_dir / csv_name

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            return False

        try:
            # Read CSV with Polars
            df = pl.read_csv(str(csv_path))

            # Register with DuckDB and create table
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

            print(f"Loaded {len(df)} rows into {table_name}")
            return True
        except Exception as e:
            print(f"Error loading {csv_name}: {e}")
            return False

    def load_all_tables(self):
        """Load all Olist tables into DuckDB"""
        tables = {
            "olist_customers_dataset.csv": "customers",
            "olist_orders_dataset.csv": "orders",
            "olist_order_items_dataset.csv": "order_items",
            "olist_order_payments_dataset.csv": "order_payments",
            "olist_products_dataset.csv": "products",
            "olist_sellers_dataset.csv": "sellers",
            "olist_order_reviews_dataset.csv": "order_reviews",
            "product_category_name_translation.csv": "category_translation",
        }

        self.connect()

        for csv_file, table_name in tables.items():
            self.load_csv_to_table(csv_file, table_name)

        print("\nData ingestion complete!")
        return self.conn

    def get_table_info(self):
        """Get information about all tables"""
        self.connect()
        tables = self.conn.execute("SHOW TABLES").fetchall()

        info = {}
        for (table_name,) in tables:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            info[table_name] = count

        return info


if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.load_all_tables()

    # Show table info
    info = ingestor.get_table_info()
    print("\nTable Summary:")
    for table, count in info.items():
        print(f"  {table}: {count:,} rows")

    ingestor.close()
