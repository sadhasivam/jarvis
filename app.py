"""
StockMind - Inventory Markdown Recommendation Engine
Streamlit Application
"""

import streamlit as st
import duckdb
import polars as pl
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stockmind.ingest import DataIngestor
from stockmind.feature_engine import FeatureEngine
from stockmind.forecast import DemandForecaster
from stockmind.engine import PromotionEngine
from stockmind.ui_data import UIDataProvider


# Page config
st.set_page_config(
    page_title="StockMind - Inventory Markdown Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom CSS - Clean and simple
st.markdown("""
<style>
    /* Just increase base font size */
    html, body, [class*="css"] {
        font-size: 17px;
    }

    /* Clean headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #141413;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #87867f;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_connection():
    """Get cached database connection"""
    conn = duckdb.connect("jarvis.db", read_only=False)
    # Disable Arrow conversion to avoid segfault with Python 3.14
    # conn.execute("SET arrow_large_buffer_size=true")
    return conn


@st.cache_data
def load_data_if_needed():
    """Load CSV data into DB if tables don't exist"""
    conn = get_db_connection()
    try:
        # Check if tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]

        if "products" not in table_names or "orders" not in table_names:
            st.info("Loading data for the first time...")
            ingestor = DataIngestor()
            ingestor.load_all_tables()
            return True
        return False
    except Exception as e:
        st.error(f"Error checking tables: {e}")
        return False


def check_inventory_exists():
    """Check if inventory exists (don't auto-generate)"""
    conn = get_db_connection()
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        return "inventory_snapshot" in table_names
    except Exception:
        return False


def tab_orders():
    """Orders List Screen"""
    st.markdown("### 📦 Orders")

    conn = get_db_connection()
    ui_data = UIDataProvider(conn)

    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        status_filter = st.selectbox(
            "Order Status",
            ["All", "delivered", "shipped", "processing", "canceled"]
        )
    with col2:
        limit = st.number_input("Rows to display", min_value=10, max_value=10000, value=100, step=10)

    # Fetch data
    try:
        df = ui_data.get_orders(limit=limit)

        if status_filter != "All":
            df = df.filter(pl.col("order_status") == status_filter)

        st.write(f"**Total Orders: {len(df)}**")
        st.dataframe(df, width="stretch", height=500)

    except Exception as e:
        st.error(f"Error loading orders: {e}")


def tab_products():
    """Products List Screen"""
    st.markdown("### 📱 Products")

    conn = get_db_connection()
    ui_data = UIDataProvider(conn)

    # Filters
    col1, col2 = st.columns([2, 1])
    with col1:
        categories = ui_data.get_categories()
        category_filter = st.selectbox("Category", categories)
    with col2:
        limit = st.number_input("Rows to display", min_value=10, max_value=10000, value=100, step=10, key="prod_limit")

    # Fetch data
    try:
        df = ui_data.get_products(limit=limit, category=category_filter)

        st.write(f"**Total Products: {len(df)}**")
        st.dataframe(df, width="stretch", height=500)

        # Simple stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Products", f"{len(df):,}")
        with col2:
            total_revenue = df.select(pl.col("total_revenue").sum()).item()
            st.metric("Total Revenue", f"R$ {total_revenue:,.2f}")
        with col3:
            total_orders = df.select(pl.col("total_orders").sum()).item()
            st.metric("Total Orders", f"{total_orders:,}")

    except Exception as e:
        st.error(f"Error loading products: {e}")


def tab_customers():
    """Customers List Screen"""
    st.markdown("### 👥 Customers")

    conn = get_db_connection()
    ui_data = UIDataProvider(conn)

    # Controls
    col1, col2 = st.columns([2, 1])
    with col2:
        limit = st.number_input("Rows to display", min_value=10, max_value=10000, value=100, step=10, key="cust_limit")

    # Fetch data
    try:
        df = ui_data.get_customers(limit=limit)

        st.write(f"**Total Customers: {len(df)}**")
        st.dataframe(df, width="stretch", height=500)

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            total_spent = df.select(pl.col("total_spent").sum()).item() or 0
            st.metric("Total Spent", f"R$ {total_spent:,.2f}")
        with col3:
            total_orders = df.select(pl.col("total_orders").sum()).item() or 0
            st.metric("Total Orders", f"{total_orders:,}")

    except Exception as e:
        st.error(f"Error loading customers: {e}")


def tab_inventory():
    """Inventory CRUD Screen"""
    st.markdown("### 📊 Inventory Management")

    conn = get_db_connection()
    ui_data = UIDataProvider(conn)

    # Summary stats
    stats = ui_data.get_inventory_summary_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total SKUs", f"{stats['total_skus']:,}")
    with col2:
        st.metric("Total Units", f"{stats['total_units']:,}")
    with col3:
        st.metric("Avg Distress Score", f"{stats['avg_distress']:.3f}")
    with col4:
        st.metric("Urgent SKUs", f"{stats['urgent']:,}", delta=f"{stats['at_risk']:,} at risk")

    st.markdown("---")

    # CRUD Operations
    tab_list, tab_create, tab_update, tab_delete = st.tabs(["📋 View", "➕ Create", "✏️ Update", "🗑️ Delete"])

    with tab_list:
        limit = st.number_input("Rows to display", min_value=10, max_value=5000, value=100, step=10, key="inv_limit")

        try:
            df = ui_data.get_inventory(limit=limit)

            if len(df) > 0:
                st.dataframe(
                    df.select([
                        "product_id", "product_category_name", "on_hand_qty",
                        "distress_score", "distress_category", "days_since_last_sale",
                        "days_of_cover", "list_price", "unit_cost"
                    ]),
                    width="stretch",
                    height=400
                )
            else:
                st.warning("No inventory data. Please generate inventory first.")

        except Exception as e:
            st.error(f"Error loading inventory: {e}")

    with tab_create:
        st.info("Manual inventory creation - Add new SKU to inventory")

        with st.form("create_inventory"):
            col1, col2 = st.columns(2)
            with col1:
                product_id = st.text_input("Product ID")
                on_hand_qty = st.number_input("On Hand Quantity", min_value=0, value=100)
                list_price = st.number_input("List Price (R$)", min_value=0.0, value=100.0)
            with col2:
                unit_cost = st.number_input("Unit Cost (R$)", min_value=0.0, value=65.0)
                inventory_age_days = st.number_input("Inventory Age (days)", min_value=0, value=30)

            submitted = st.form_submit_button("Create Inventory Record")

            if submitted and product_id:
                try:
                    # Insert record
                    conn.execute(f"""
                    INSERT INTO inventory_snapshot (
                        product_id, on_hand_qty, list_price, unit_cost, inventory_age_days, snapshot_date
                    ) VALUES (
                        '{product_id}', {on_hand_qty}, {list_price}, {unit_cost}, {inventory_age_days}, '{datetime.now().date()}'
                    )
                    """)
                    st.success(f"Created inventory record for {product_id}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating record: {e}")

    with tab_update:
        st.info("Update existing inventory record")

        product_id_update = st.text_input("Product ID to Update")

        if product_id_update:
            try:
                existing = conn.execute(f"""
                SELECT * FROM inventory_snapshot WHERE product_id = '{product_id_update}' LIMIT 1
                """).fetchone()

                if existing:
                    with st.form("update_inventory"):
                        on_hand_qty = st.number_input("On Hand Quantity", value=int(existing[3] or 0))
                        list_price = st.number_input("List Price", value=float(existing[10] or 0))

                        submitted = st.form_submit_button("Update Record")

                        if submitted:
                            conn.execute(f"""
                            UPDATE inventory_snapshot
                            SET on_hand_qty = {on_hand_qty}, list_price = {list_price}
                            WHERE product_id = '{product_id_update}'
                            """)
                            st.success(f"Updated {product_id_update}")
                            st.rerun()
                else:
                    st.warning("Product ID not found")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab_delete:
        st.warning("Delete inventory record")

        product_id_delete = st.text_input("Product ID to Delete", key="delete_pid")

        if st.button("Delete Record", type="secondary"):
            if product_id_delete:
                try:
                    conn.execute(f"DELETE FROM inventory_snapshot WHERE product_id = '{product_id_delete}'")
                    st.success(f"Deleted {product_id_delete}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


def tab_promotion_engine():
    """Promotion Engine with configurable rules"""
    st.markdown("### 🎯 Promotion Recommendation Engine")

    conn = get_db_connection()

    # Configuration Section
    st.markdown("#### ⚙️ Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Distress Weights**")
        w_non_moving = st.slider("Non-Moving", 0.0, 1.0, 0.25, 0.05)
        w_overstock = st.slider("Overstock", 0.0, 1.0, 0.20, 0.05)
        w_aging = st.slider("Aging", 0.0, 1.0, 0.20, 0.05)

    with col2:
        st.markdown("**Distress Weights (cont.)**")
        w_decay = st.slider("Decay", 0.0, 1.0, 0.15, 0.05)
        w_storage = st.slider("Storage", 0.0, 1.0, 0.10, 0.05)
        w_demand = st.slider("Demand Weakness", 0.0, 1.0, 0.10, 0.05)

    with col3:
        st.markdown("**Thresholds**")
        distress_threshold = st.slider("Min Distress for Promo", 0.0, 1.0, 0.55, 0.05)
        min_margin = st.slider("Min Margin %", 0.0, 0.5, 0.15, 0.05)
        low_stock_threshold = st.slider("Low Stock Threshold", 0.0, 2.0, 0.75, 0.05)

    st.markdown("---")

    # Check if inventory exists
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        inventory_exists = "inventory_snapshot" in table_names
    except:
        inventory_exists = False

    if not inventory_exists:
        st.warning("⚠️ Inventory not initialized yet!")
        st.info("Click the '⚡ Generate Inventory' button in the sidebar to create synthetic inventory data first.")
        return

    # Run Analysis Button
    if st.button("🚀 Run Promotion Analysis", type="primary", width="stretch"):
        with st.spinner("Analyzing inventory and generating recommendations..."):
            try:
                # Load inventory (avoid .pl() due to Python 3.14 Arrow issues)
                inventory_df = pl.read_database("SELECT * FROM inventory_snapshot LIMIT 5000", conn)

                if len(inventory_df) == 0:
                    st.error("No inventory data found. Please regenerate inventory.")
                    return

                # Recompute features with custom weights
                weights = {
                    "non_moving": w_non_moving,
                    "overstock": w_overstock,
                    "aging": w_aging,
                    "decay": w_decay,
                    "storage": w_storage,
                    "demand_weakness": w_demand
                }

                feature_engine = FeatureEngine()
                inventory_df = feature_engine.compute_distress_features(inventory_df)
                inventory_df = feature_engine.compute_with_custom_weights(inventory_df, weights)

                # Forecast baseline demand
                forecaster = DemandForecaster()
                inventory_df = forecaster.compute_baseline_demand(inventory_df)

                # Run promotion engine
                engine = PromotionEngine(
                    distress_threshold=distress_threshold,
                    min_margin_pct=min_margin,
                    low_stock_threshold=low_stock_threshold
                )

                recommendations_df = engine.simulate_promotions(inventory_df)

                # Store in session state
                st.session_state['recommendations'] = recommendations_df
                st.session_state['inventory_analyzed'] = inventory_df

                st.success(f"✅ Analysis complete! Generated recommendations for {len(recommendations_df)} SKUs")

            except Exception as e:
                st.error(f"Error running analysis: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Display Results
    if 'recommendations' in st.session_state:
        st.markdown("---")
        st.markdown("#### 📊 Results")

        recommendations_df = st.session_state['recommendations']

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        promo_count = len(recommendations_df.filter(pl.col("recommended_discount_pct") > 0))
        total_relief = recommendations_df.select(pl.col("inventory_relief").sum()).item()
        total_margin = recommendations_df.select(pl.col("expected_margin").sum()).item()
        avg_discount = recommendations_df.filter(
            pl.col("recommended_discount_pct") > 0
        ).select(pl.col("recommended_discount_pct").mean()).item() if promo_count > 0 else 0

        with col1:
            st.metric("SKUs to Promote", f"{promo_count:,}")
        with col2:
            st.metric("Expected Inventory Relief", f"{int(total_relief):,} units")
        with col3:
            st.metric("Expected Total Margin", f"R$ {total_margin:,.2f}")
        with col4:
            st.metric("Avg Discount", f"{avg_discount*100:.1f}%")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Promotion distribution
            promo_dist = recommendations_df.group_by("recommended_action").agg(
                pl.len().alias("count")
            ).sort("recommended_action")

            fig = px.bar(
                promo_dist.to_pandas(),
                x="recommended_action",
                y="count",
                title="Promotion Action Distribution",
                labels={"recommended_action": "Action", "count": "SKU Count"}
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            # Distress category distribution
            distress_dist = recommendations_df.group_by("distress_category").agg(
                pl.len().alias("count")
            ).sort("distress_category")

            fig = px.pie(
                distress_dist.to_pandas(),
                names="distress_category",
                values="count",
                title="Distress Category Distribution"
            )
            st.plotly_chart(fig, width="stretch")

        # Top recommendations table
        st.markdown("#### 🔝 Top Recommendations")

        top_recs = recommendations_df.filter(
            pl.col("recommended_discount_pct") > 0
        ).sort("action_score", descending=True).head(20)

        st.dataframe(
            top_recs.select([
                "product_id", "distress_score", "distress_category",
                "on_hand_qty", "recommended_action", "expected_margin",
                "inventory_relief", "reason"
            ]),
            width="stretch",
            height=400
        )

        # Download button
        csv = recommendations_df.to_pandas().to_csv(index=False)
        st.download_button(
            label="📥 Download Full Recommendations (CSV)",
            data=csv,
            file_name=f"stockmind_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def main():
    """Main application"""

    # Header
    st.markdown('<p class="main-header">StockMind</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Inventory Markdown Recommendation Engine</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # Show inventory status
        inventory_exists = check_inventory_exists()
        if inventory_exists:
            st.success("✅ Inventory Ready")
            try:
                conn = get_db_connection()
                count = conn.execute("SELECT COUNT(*) FROM inventory_snapshot").fetchone()[0]
                st.caption(f"{count:,} SKUs loaded")
            except:
                pass
        else:
            st.error("❌ No Inventory")
            st.caption("Run: python generate_inventory.py")

        st.markdown("---")

        st.markdown("### About")
        st.markdown("""
        **StockMind** identifies distressed inventory and recommends
        optimal markdown actions to maximize margin while clearing stock.
        """)

    # Initialize data
    load_data_if_needed()

    # Check if inventory exists (show banner if not)
    inventory_exists = check_inventory_exists()
    if not inventory_exists:
        st.warning("⚠️ **Inventory not yet created!** Click '⚡ Generate Inventory' in the sidebar to get started.")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 Orders",
        "📱 Products",
        "👥 Customers",
        "📊 Inventory",
        "🎯 Promotion Engine"
    ])

    with tab1:
        tab_orders()

    with tab2:
        tab_products()

    with tab3:
        tab_customers()

    with tab4:
        tab_inventory()

    with tab5:
        tab_promotion_engine()


if __name__ == "__main__":
    main()
