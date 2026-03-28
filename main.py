#!/usr/bin/env python3
"""
StockMind - FastAPI + HTMX Application
Inventory Markdown Recommendation Engine
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import duckdb
import polars as pl
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from stockmind.feature_engine import FeatureEngine
from stockmind.forecast import DemandForecaster
from stockmind.engine import PromotionEngine

# Initialize FastAPI - Disable auto docs to use /about instead
app = FastAPI(
    title="StockMind",
    version="0.1.0",
    docs_url=None,  # Disable /docs
    redoc_url=None  # Disable /redoc
)

# Mount static files only if directory exists and has files
static_dir = Path("static")
if static_dir.exists() and any(static_dir.rglob("*")):
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates - Disable cache for Python 3.14 compatibility
# Create a no-op cache to avoid Python 3.14 dict hashing issues with Jinja2
class NoOpCache(dict):
    def __setitem__(self, key, value):  # noqa: ARG002
        pass
    def __getitem__(self, key):
        raise KeyError(key)
    def get(self, key, default=None):
        return default

templates = Jinja2Templates(directory="templates")
templates.env.cache = NoOpCache()
templates.env.auto_reload = True

# Database connection
DB_PATH = "jarvis.db"


def get_db_connection():
    """Get DuckDB connection"""
    conn = duckdb.connect(DB_PATH, read_only=False)
    return conn


def get_inventory_count():
    """Get total SKU count from inventory"""
    try:
        conn = get_db_connection()
        result = conn.execute("SELECT COUNT(*) FROM inventory_snapshot").fetchone()
        conn.close()
        return result[0] if result else 0
    except:
        return 0


# =============================================================================
# PAGE ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def orders_page(request: Request):
    """Orders tab page"""
    return templates.TemplateResponse(
        request=request,
        name="orders.html",
        context={
            "active_tab": "orders",
            "inventory_count": get_inventory_count()
        }
    )


@app.get("/products", response_class=HTMLResponse)
async def products_page(request: Request):
    """Products tab page"""
    return templates.TemplateResponse(
        request=request,
        name="products.html",
        context={
            "active_tab": "products",
            "inventory_count": get_inventory_count()
        }
    )


@app.get("/customers", response_class=HTMLResponse)
async def customers_page(request: Request):
    """Customers tab page"""
    return templates.TemplateResponse(
        request=request,
        name="customers.html",
        context={
            "active_tab": "customers",
            "inventory_count": get_inventory_count()
        }
    )


@app.get("/inventory", response_class=HTMLResponse)
async def inventory_page(request: Request):
    """Inventory CRUD tab page"""
    return templates.TemplateResponse(
        request=request,
        name="inventory.html",
        context={
            "active_tab": "inventory",
            "inventory_count": get_inventory_count()
        }
    )


@app.get("/promotion-engine", response_class=HTMLResponse)
async def promotion_engine_page(request: Request):
    """Promotion Engine tab page"""
    return templates.TemplateResponse(
        request=request,
        name="promotion_engine.html",
        context={
            "active_tab": "promotion",
            "inventory_count": get_inventory_count()
        }
    )


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About tab page"""
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "active_tab": "about",
            "inventory_count": get_inventory_count()
        }
    )


# =============================================================================
# API ENDPOINTS (HTMX Partials)
# =============================================================================

@app.get("/api/orders", response_class=HTMLResponse)
async def get_orders(request: Request, limit: int = 50):
    """Get orders data for HTMX update"""
    try:
        conn = get_db_connection()

        # Query orders with items
        query = """
            SELECT
                o.order_id,
                o.customer_id,
                o.order_status,
                o.order_purchase_timestamp::VARCHAR as order_date,
                COUNT(oi.order_item_id) as items_count,
                ROUND(SUM(oi.price), 2) as total_value
            FROM orders o
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            GROUP BY o.order_id, o.customer_id, o.order_status, o.order_purchase_timestamp
            ORDER BY o.order_purchase_timestamp DESC
            LIMIT ?
        """

        orders_df = pl.read_database(query, conn, execute_options={'parameters': [limit]})
        conn.close()

        # Convert to list of dicts
        orders = orders_df.to_dicts() if len(orders_df) > 0 else []

        return templates.TemplateResponse(
            request=request,
            name="partials/orders_table.html",
            context={"orders": orders}
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={"error": str(e)}
        )


@app.get("/api/products", response_class=HTMLResponse)
async def get_products(request: Request, limit: int = 50):
    """Get products data for HTMX update"""
    try:
        conn = get_db_connection()

        query = """
            SELECT
                p.product_id,
                p.product_category_name,
                ROUND(p.product_weight_g / 1000.0, 2) as weight_kg,
                p.product_length_cm * p.product_height_cm * p.product_width_cm as volume_cm3,
                COUNT(DISTINCT oi.order_id) as orders_count,
                ROUND(SUM(oi.price), 2) as total_revenue
            FROM products p
            LEFT JOIN order_items oi ON p.product_id = oi.product_id
            GROUP BY p.product_id, p.product_category_name, p.product_weight_g,
                     p.product_length_cm, p.product_height_cm, p.product_width_cm
            ORDER BY total_revenue DESC
            LIMIT ?
        """

        products_df = pl.read_database(query, conn, execute_options={'parameters': [limit]})
        conn.close()

        products = products_df.to_dicts() if len(products_df) > 0 else []

        return templates.TemplateResponse(
            request=request,
            name="partials/products_table.html",
            context={"products": products}
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={"error": str(e)}
        )


@app.get("/api/customers", response_class=HTMLResponse)
async def get_customers(request: Request, limit: int = 50):
    """Get customers data for HTMX update"""
    try:
        conn = get_db_connection()

        query = """
            SELECT
                c.customer_id,
                c.customer_unique_id,
                c.customer_city,
                c.customer_state,
                COUNT(DISTINCT o.order_id) as orders_count,
                ROUND(SUM(oi.price), 2) as total_spent
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            GROUP BY c.customer_id, c.customer_unique_id, c.customer_city, c.customer_state
            ORDER BY total_spent DESC
            LIMIT ?
        """

        customers_df = pl.read_database(query, conn, execute_options={'parameters': [limit]})
        conn.close()

        customers = customers_df.to_dicts() if len(customers_df) > 0 else []

        return templates.TemplateResponse(
            request=request,
            name="partials/customers_table.html",
            context={"customers": customers}
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={"error": str(e)}
        )


@app.get("/api/inventory", response_class=HTMLResponse)
async def get_inventory(request: Request, limit: int = 50, offset: int = 0):
    """Get inventory data for HTMX update"""
    try:
        conn = get_db_connection()

        query = """
            SELECT
                product_id,
                on_hand_qty,
                list_price,
                days_since_last_sale,
                days_of_cover,
                distress_score,
                distress_category,
                inventory_age_days
            FROM inventory_snapshot
            ORDER BY distress_score DESC
            LIMIT ? OFFSET ?
        """

        inventory_df = pl.read_database(
            query,
            conn,
            execute_options={'parameters': [limit, offset]}
        )
        conn.close()

        inventory = inventory_df.to_dicts() if len(inventory_df) > 0 else []

        return templates.TemplateResponse(
            request=request,
            name="partials/inventory_table.html",
            context={"inventory": inventory}
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={"error": str(e)}
        )


@app.post("/api/analyze", response_class=HTMLResponse)
async def run_promotion_analysis(
    request: Request,
    distress_threshold: float = Form(0.55),
    min_margin_pct: float = Form(0.15),
    low_stock_threshold: float = Form(0.75),
    w_non_moving: float = Form(0.25),
    w_overstock: float = Form(0.20),
    w_aging: float = Form(0.20),
    w_decay: float = Form(0.15),
    w_storage: float = Form(0.10),
    w_demand_weak: float = Form(0.10),
):
    """Run promotion engine analysis"""
    try:
        conn = get_db_connection()

        # Load inventory (limit to 200 for 1GB server)
        inventory_df = pl.read_database(
            "SELECT * FROM inventory_snapshot LIMIT 200",
            conn
        )
        conn.close()

        if len(inventory_df) == 0:
            raise ValueError("No inventory data found. Please run generate_inventory.py first.")

        # Recompute distress scores with custom weights
        feature_engine = FeatureEngine()
        inventory_df = feature_engine.compute_with_custom_weights(
            inventory_df,
            weights={
                'non_moving': w_non_moving,
                'overstock': w_overstock,
                'aging': w_aging,
                'decay': w_decay,
                'storage': w_storage,
                'demand_weakness': w_demand_weak,
            }
        )

        # Recompute forecast
        forecaster = DemandForecaster()
        inventory_df = forecaster.compute_baseline_demand(inventory_df)

        # Run promotion engine
        engine = PromotionEngine(
            distress_threshold=distress_threshold,
            min_margin_pct=min_margin_pct,
            low_stock_threshold=low_stock_threshold
        )

        recommendations = engine.simulate_promotions(inventory_df)

        # Convert to list of dicts for template
        results = recommendations.to_dicts() if len(recommendations) > 0 else []

        # Calculate summary stats
        summary = {
            "total_skus": len(results),
            "urgent": sum(1 for r in results if r.get('distress_category') == 'urgent'),
            "at_risk": sum(1 for r in results if r.get('distress_category') == 'at_risk'),
            "promo_0": sum(1 for r in results if r.get('recommended_discount_pct') == 0.0),
            "promo_5": sum(1 for r in results if r.get('recommended_discount_pct') == 0.05),
            "promo_10": sum(1 for r in results if r.get('recommended_discount_pct') == 0.10),
            "promo_15": sum(1 for r in results if r.get('recommended_discount_pct') == 0.15),
            "promo_20": sum(1 for r in results if r.get('recommended_discount_pct') == 0.20),
            "expected_total_margin": sum(r.get('expected_margin', 0) for r in results),
            "expected_inventory_relief": sum(r.get('expected_units_with_action', 0) for r in results),
        }

        return templates.TemplateResponse(
            request=request,
            name="partials/analysis_results.html",
            context={
                "results": results[:50],  # Show top 50
                "summary": summary
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={"error": str(e)}
        )


# =============================================================================
# INVENTORY CRUD OPERATIONS
# =============================================================================

@app.post("/api/inventory/update", response_class=HTMLResponse)
async def update_inventory_item(
    request: Request,
    product_id: str = Form(...),
    on_hand_qty: int = Form(...),
    list_price: float = Form(...)
):
    """Update inventory item"""
    try:
        conn = get_db_connection()

        conn.execute("""
            UPDATE inventory_snapshot
            SET on_hand_qty = ?,
                list_price = ?
            WHERE product_id = ?
        """, [on_hand_qty, list_price, product_id])

        conn.close()

        return templates.TemplateResponse(
            request=request,
            name="partials/success.html",
            context={"message": f"Updated {product_id}"}
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={"error": str(e)}
        )


@app.delete("/api/inventory/{product_id}", response_class=HTMLResponse)
async def delete_inventory_item(request: Request, product_id: str):
    """Delete inventory item"""
    try:
        conn = get_db_connection()

        conn.execute(
            "DELETE FROM inventory_snapshot WHERE product_id = ?",
            [product_id]
        )

        conn.close()

        return templates.TemplateResponse(
            request=request,
            name="partials/success.html",
            context={"message": f"Deleted {product_id}"}
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={"error": str(e)}
        )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "StockMind"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999, reload=True)
