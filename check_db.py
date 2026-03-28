#!/usr/bin/env python3
"""Quick database diagnostic script"""

import duckdb

print("="*60)
print("Database Diagnostic Check")
print("="*60)

try:
    conn = duckdb.connect('jarvis.db')

    # List all tables
    print("\n📊 Tables in database:")
    tables = conn.execute("SHOW TABLES").fetchall()

    if not tables:
        print("  ❌ No tables found!")
        print("\n  Run this to load data:")
        print("  python -c 'from src.stockmind.ingest import DataIngestor; DataIngestor().load_all_tables()'")
    else:
        for (table_name,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  ✓ {table_name:30} {count:>10,} rows")

    # Check specific tables needed by app
    print("\n🔍 Required tables check:")
    required = ['orders', 'products', 'customers', 'order_items', 'inventory_snapshot']
    table_names = [t[0] for t in tables]

    for req in required:
        if req in table_names:
            print(f"  ✅ {req}")
        else:
            print(f"  ❌ {req} - MISSING!")

    print("\n" + "="*60)

    conn.close()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
