import pandas as pd
import streamlit as st
import os

def load_data(data_dir='data'):
    """
    Handles the interactive loading of data (file uploader or local directory).
    This function is NOT cached because it contains a widget.
    """
    files_needed = [
        'olist_orders_dataset.csv', 'olist_order_items_dataset.csv',
        'olist_products_dataset.csv', 'olist_customers_dataset.csv',
        'olist_order_reviews_dataset.csv', 'olist_order_payments_dataset.csv',
        'olist_sellers_dataset.csv', 'olist_product_category_name_translation.csv'
    ]
    
    dfs = {}
    local_data_exists = os.path.exists(data_dir) and all(os.path.exists(os.path.join(data_dir, f)) for f in files_needed)
    
    if local_data_exists:
        for f in files_needed:
            dfs[f] = pd.read_csv(os.path.join(data_dir, f))
    else:
        # Fallback for deployment environments where /data is missing
        st.info("📦 **Deployment Mode**: Local `/data` directory not detected. Please upload the required Olist CSV files.")
        uploaded_files = st.file_uploader("Upload Olist CSV Datasets", accept_multiple_files=True, type=['csv'])
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                dfs[uploaded_file.name] = pd.read_csv(uploaded_file)
            
            # Check if all necessary files have been uploaded
            missing_files = [f for f in files_needed if f not in dfs]
            if missing_files:
                st.warning(f"Pending uploads: {', '.join(missing_files)}")
                return None
        else:
            return None

    # Call the cached transformation function
    return transform_data(dfs)

@st.cache_data
def transform_data(dfs):
    """
    Heavily cached data transformation and feature engineering.
    """
    orders = dfs['olist_orders_dataset.csv']
    order_items = dfs['olist_order_items_dataset.csv']
    products = dfs['olist_products_dataset.csv']
    customers = dfs['olist_customers_dataset.csv']
    reviews = dfs['olist_order_reviews_dataset.csv']
    payments = dfs['olist_order_payments_dataset.csv']
    sellers = dfs['olist_sellers_dataset.csv']
    translation = dfs['olist_product_category_name_translation.csv']
    
    # Date parsing
    date_cols_orders = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols_orders: orders[col] = pd.to_datetime(orders[col])
    order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'])
    for col in ['review_creation_date', 'review_answer_timestamp']: reviews[col] = pd.to_datetime(reviews[col])

    # Merging
    master_df = orders.merge(customers, on='customer_id', how='left')
    master_df = master_df.merge(order_items, on='order_id', how='left')
    master_df = master_df.merge(products, on='product_id', how='left')
    master_df = master_df.merge(translation, on='product_category_name', how='left')
    master_df = master_df.merge(sellers, on='seller_id', how='left')
    master_df = master_df.merge(reviews, on='order_id', how='left')
    master_df = master_df.merge(payments, on='order_id', how='left')
    
    # Feature engineering
    master_df['order_month'] = master_df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    master_df['order_year'] = master_df['order_purchase_timestamp'].dt.year
    
    # Delivery metrics (will be NaN for undelivered orders)
    master_df['delivery_days'] = (master_df['order_delivered_customer_date'] - master_df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
    
    # is_late: 1 if late, 0 if on-time. NaN for undelivered.
    master_df['is_late'] = (master_df['order_delivered_customer_date'] > master_df['order_estimated_delivery_date']).astype(float)
    master_df.loc[master_df['order_delivered_customer_date'].isna(), 'is_late'] = float('nan')
    
    return master_df
