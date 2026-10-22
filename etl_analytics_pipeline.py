"""
E-Commerce Sales & Customer Retention Analytics Pipeline
Author: Van Trieu Phan
Description:
    Extracts and cleans multi-table transactional data using Python (Pandas)
    and executes complex SQL queries (CTEs, Window Functions, Multi-table Joins)
    to evaluate cohort retention, CLV, churn indicators, and seasonal revenue trends.
"""

import os
import sqlite3
import pandas as pd


def load_and_clean_data(data_dir: str) -> dict:
    """Loads raw CSV datasets and performs initial data cleaning using Pandas."""
    print("Loading raw datasets...")
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    translations = pd.read_csv(os.path.join(data_dir, "product_category_name_translation.csv"))

    print("Cleaning and standardizing data in Pandas...")
    # 1. Standardize timestamp columns and drop records with missing purchase dates
    orders = orders.dropna(subset=["order_purchase_timestamp"]).copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_date"] = orders["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")
    orders["order_month"] = orders["order_purchase_timestamp"].dt.strftime("%Y-%m-01")

    # 2. Clean missing product categories and merge English translations
    products["product_category_name"] = products["product_category_name"].fillna("unknown")
    products_clean = pd.merge(
        products[["product_id", "product_category_name"]],
        translations,
        on="product_category_name",
        how="left"
    )
    products_clean["product_category_name_english"] = (
        products_clean["product_category_name_english"]
        .fillna("other_uncategorized")
        .str.replace("_", " ")
        .str.title()
    )

    # 3. Filter out invalid payment values
    payments = payments[payments["payment_value"] > 0].copy()

    return {
        "orders": orders,
        "items": items,
        "customers": customers,
        "payments": payments,
        "products": products_clean,
    }


def build_sqlite_database(datasets: dict, db_path: str) -> sqlite3.Connection:
    """Loads cleaned Pandas DataFrames into a local SQLite relational database."""
    print(f"Building SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    for table_name, df in datasets.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    return conn


def run_sql_analytics(conn: sqlite3.Connection, output_dir: str) -> None:
    """Executes complex SQL queries with CTEs, Joins, and Window Functions."""
    print("Executing SQL analytical queries...")

    # Query 1: Master Transactional & Customer KPI Dataset (CTEs + Window Functions + Joins)
    kpi_query = """
    WITH OrderPayments AS (
        -- Aggregate total payment value per order to prevent fan-out during joins
        SELECT 
            order_id,
            ROUND(SUM(payment_value), 2) AS total_order_payment
        FROM payments
        GROUP BY order_id
    ),
    OrderItemDetails AS (
        -- Join order items with translated English product categories
        SELECT 
            i.order_id,
            p.product_category_name_english AS product_category,
            COUNT(i.order_item_id) AS item_count,
            ROUND(SUM(i.price), 2) AS items_subtotal,
            ROUND(SUM(i.freight_value), 2) AS freight_total
        FROM items i
        LEFT JOIN products p ON i.product_id = p.product_id
        GROUP BY i.order_id, p.product_category_name_english
    ),
    CustomerWindowMetrics AS (
        -- Use SQL Window Functions to compute Cohort Month, Order Sequence, and CLV per customer
        SELECT 
            o.order_id,
            c.customer_unique_id,
            c.customer_state,
            o.order_date,
            o.order_month,
            MIN(o.order_month) OVER (
                PARTITION BY c.customer_unique_id
            ) AS cohort_month,
            ROW_NUMBER() OVER (
                PARTITION BY c.customer_unique_id 
                ORDER BY o.order_date
            ) AS customer_order_seq,
            COUNT(o.order_id) OVER (
                PARTITION BY c.customer_unique_id
            ) AS total_customer_orders,
            ROUND(SUM(op.total_order_payment) OVER (
                PARTITION BY c.customer_unique_id
            ), 2) AS customer_lifetime_value
        FROM orders o
        INNER JOIN customers c ON o.customer_id = c.customer_id
        INNER JOIN OrderPayments op ON o.order_id = op.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_month >= '2017-01-01'
          AND o.order_month <= '2018-08-01'
    )
    SELECT 
        cw.order_id,
        cw.customer_unique_id,
        cw.customer_state,
        cw.order_date,
        cw.order_month,
        cw.cohort_month,
        -- Calculate Cohort Index (months elapsed since first purchase)
        (CAST(STRFTIME('%Y', cw.order_month) AS INTEGER) - CAST(STRFTIME('%Y', cw.cohort_month) AS INTEGER)) * 12 +
        (CAST(STRFTIME('%m', cw.order_month) AS INTEGER) - CAST(STRFTIME('%m', cw.cohort_month) AS INTEGER)) AS cohort_index,
        COALESCE(oid.product_category, 'Other Uncategorized') AS product_category,
        COALESCE(oid.item_count, 1) AS item_count,
        COALESCE(oid.items_subtotal, 0.0) AS product_revenue,
        COALESCE(oid.freight_total, 0.0) AS freight_value,
        op.total_order_payment AS total_payment_value,
        cw.customer_order_seq,
        cw.total_customer_orders,
        cw.customer_lifetime_value,
        CASE 
            WHEN cw.total_customer_orders > 1 THEN 'Repeat Customer'
            ELSE 'One-Time / Churned'
        END AS retention_status
    FROM CustomerWindowMetrics cw
    INNER JOIN OrderPayments op ON cw.order_id = op.order_id
    LEFT JOIN OrderItemDetails oid ON cw.order_id = oid.order_id
    ORDER BY cw.order_date;
    """

    master_df = pd.read_sql_query(kpi_query, conn)
    master_output_path = os.path.join(output_dir, "tableau_ecommerce_kpis.csv")
    master_df.to_csv(master_output_path, index=False)
    print(f"Exported master KPI dataset ({len(master_df):,} rows) to {master_output_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    DB_PATH = os.path.join(BASE_DIR, "ecommerce.db")

    cleaned_data = load_and_clean_data(DATA_DIR)
    db_conn = build_sqlite_database(cleaned_data, DB_PATH)
    run_sql_analytics(db_conn, OUTPUT_DIR)
    db_conn.close()
    print("ETL & SQL Analytics Pipeline completed successfully.")