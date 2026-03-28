#!/usr/bin/env python3
"""
Standalone script to generate synthetic inventory data
Run this BEFORE starting the FastAPI server

Usage:
    python generate_inventory.py
"""

import duckdb
import sys
sys.path.insert(0, 'src')

from stockmind.inventory_synth import InventorySynthesizer
from stockmind.feature_engine import FeatureEngine
from stockmind.forecast import DemandForecaster

print("="*60)
print("StockMind - Inventory Generation Script")
print("Processing 500 products (optimized for 1GB servers)")
print("="*60)

print("\n🔌 Connecting to database...")
conn = duckdb.connect('jarvis.db')
conn.execute("SET arrow_large_buffer_size=true")

print("\n📊 Checking existing tables...")
tables = conn.execute("SHOW TABLES").fetchall()
table_names = [t[0] for t in tables]

if "inventory_snapshot" in table_names:
    count = conn.execute("SELECT COUNT(*) FROM inventory_snapshot").fetchone()[0]
    print(f"⚠️  Found existing inventory_snapshot table with {count:,} rows")
    response = input("   Delete and regenerate? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled. Exiting.")
        conn.close()
        sys.exit(0)
    print("   Dropping existing table...")
    conn.execute("DROP TABLE inventory_snapshot")

# Check required tables
required_tables = ['products', 'orders', 'order_items']
missing = [t for t in required_tables if t not in table_names]
if missing:
    print(f"\n❌ ERROR: Missing required tables: {missing}")
    print("   Please run the app first to load CSV data, or run data ingestion.")
    conn.close()
    sys.exit(1)

print("\n" + "="*60)
print("Starting Inventory Generation")
print("="*60)

try:
    # Step 1: Generate inventory snapshot
    print("\n1️⃣  Generating synthetic inventory...")
    synth = InventorySynthesizer(conn)
    inventory_df = synth.generate_inventory_snapshot()
    print(f"   ✅ Generated {len(inventory_df):,} SKUs")

    # Step 2: Compute distress features
    print("\n2️⃣  Computing distress features...")
    feature_engine = FeatureEngine()
    inventory_df = feature_engine.compute_distress_features(inventory_df)
    print(f"   ✅ Computed distress scores")

    # Step 3: Compute baseline forecast
    print("\n3️⃣  Computing baseline demand forecast...")
    forecaster = DemandForecaster()
    inventory_df = forecaster.compute_baseline_demand(inventory_df)
    print(f"   ✅ Computed 30-day baseline forecast")

    # Step 4: Save to database
    print("\n4️⃣  Saving to database...")
    conn.execute("DROP TABLE IF EXISTS inventory_snapshot")
    conn.execute("CREATE TABLE inventory_snapshot AS SELECT * FROM inventory_df")

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM inventory_snapshot").fetchone()[0]
    print(f"   ✅ Saved {count:,} rows to inventory_snapshot table")

    # Show summary stats
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    stats = conn.execute("""
        SELECT
            COUNT(*) as total_skus,
            SUM(on_hand_qty) as total_units,
            SUM(CASE WHEN distress_category = 'urgent' THEN 1 ELSE 0 END) as urgent,
            SUM(CASE WHEN distress_category = 'at_risk' THEN 1 ELSE 0 END) as at_risk,
            SUM(CASE WHEN distress_category = 'watch' THEN 1 ELSE 0 END) as watch,
            SUM(CASE WHEN distress_category = 'healthy' THEN 1 ELSE 0 END) as healthy,
            ROUND(AVG(distress_score), 3) as avg_distress_score,
            ROUND(AVG(on_hand_qty), 1) as avg_on_hand,
            ROUND(AVG(list_price), 2) as avg_price
        FROM inventory_snapshot
    """).fetchone()

    # Return stats
    return_stats = conn.execute("""
        SELECT
            SUM(CASE WHEN is_returned = true THEN 1 ELSE 0 END) as returned_count,
            SUM(CASE WHEN return_condition = 'good' THEN 1 ELSE 0 END) as good_returns,
            SUM(CASE WHEN return_condition = 'opened' THEN 1 ELSE 0 END) as opened_returns,
            SUM(CASE WHEN return_condition = 'damaged' THEN 1 ELSE 0 END) as damaged_returns,
            ROUND(AVG(CASE WHEN is_returned = true THEN distress_score ELSE NULL END), 3) as avg_return_distress,
            ROUND(AVG(CASE WHEN is_returned = false THEN distress_score ELSE NULL END), 3) as avg_new_distress
        FROM inventory_snapshot
    """).fetchone()

    print(f"""
    Total SKUs:           {stats[0]:,}
    Total Units:          {stats[1]:,}

    Distress Categories:
      - Urgent:           {stats[2]:,}
      - At Risk:          {stats[3]:,}
      - Watch:            {stats[4]:,}
      - Healthy:          {stats[5]:,}

    Return Inventory:
      - Total Returns:    {return_stats[0]:,} ({return_stats[0]/stats[0]*100:.1f}%)
      - Good Condition:   {return_stats[1]:,}
      - Opened:           {return_stats[2]:,}
      - Damaged:          {return_stats[3]:,}
      - Avg Distress (Returns): {return_stats[4]}
      - Avg Distress (New):     {return_stats[5]}

    Averages:
      - Distress Score:   {stats[6]}
      - On Hand Qty:      {stats[7]}
      - List Price:       R$ {stats[8]}
    """)

    # Show top 5 distressed SKUs
    print("="*60)
    print("Top 5 Most Distressed SKUs")
    print("="*60)

    top_distressed = conn.execute("""
        SELECT
            product_id,
            ROUND(distress_score, 3) as distress,
            distress_category,
            on_hand_qty,
            days_since_last_sale,
            ROUND(list_price, 2) as price,
            return_condition
        FROM inventory_snapshot
        ORDER BY distress_score DESC
        LIMIT 5
    """).fetchall()

    for i, row in enumerate(top_distressed, 1):
        return_tag = f"[{row[6].upper()}]" if row[6] != "new" else ""
        print(f"{i}. {row[0][:20]:20} {return_tag:10} | Score: {row[1]} | {row[2]:8} | Qty: {row[3]:4} | Days: {row[4]:3} | Price: R${row[5]}")

    print("\n" + "="*60)
    print("✅ SUCCESS! Inventory generation complete.")
    print("="*60)
    print("\nYou can now start the FastAPI server:")
    print("  ./run_server.sh")
    print()

except Exception as e:
    print(f"\n❌ ERROR during inventory generation:")
    print(f"   {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    conn.close()
