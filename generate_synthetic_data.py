import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os
import uuid
from datetime import datetime, timedelta

def generate_data(n_orders=5000, data_dir='data'):
    print(f"Generating synthetic dataset with {n_orders} orders in '{data_dir}'...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    np.random.seed(42)
    
    # 1. Products
    categories = ['bed_bath_table', 'health_beauty', 'sports_leisure', 'furniture_decor', 'computers_accessories', 'housewares', 'watches_gifts', 'telephony', 'garden_tools', 'auto']
    n_products = 500
    products = pd.DataFrame({
        'product_id': [str(uuid.uuid4())[:8] for _ in range(n_products)],
        'product_category_name': np.random.choice(categories, n_products),
        'product_name_lenght': np.random.randint(10, 60, n_products),
        'product_description_lenght': np.random.randint(100, 1000, n_products),
        'product_photos_qty': np.random.randint(1, 10, n_products),
        'product_weight_g': np.random.randint(100, 10000, n_products),
        'product_length_cm': np.random.randint(10, 50, n_products),
        'product_height_cm': np.random.randint(10, 50, n_products),
        'product_width_cm': np.random.randint(10, 50, n_products)
    })
    products.to_csv(os.path.join(data_dir, 'olist_products_dataset.csv'), index=False)
    
    # 2. Category Translation
    translation = pd.DataFrame({
        'product_category_name': categories,
        'product_category_name_english': categories
    })
    translation.to_csv(os.path.join(data_dir, 'olist_product_category_name_translation.csv'), index=False)
    
    # 3. Sellers
    n_sellers = 50
    sellers = pd.DataFrame({
        'seller_id': [str(uuid.uuid4())[:8] for _ in range(n_sellers)],
        'seller_zip_code_prefix': np.random.randint(1000, 99999, n_sellers),
        'seller_city': ['sao paulo', 'rio de janeiro', 'belo horizonte', 'curitiba', 'porto alegre'] * 10,
        'seller_state': ['SP', 'RJ', 'MG', 'PR', 'RS'] * 10
    })
    sellers.to_csv(os.path.join(data_dir, 'olist_sellers_dataset.csv'), index=False)
    
    # 4. Customers
    n_customers = 4000
    customers = pd.DataFrame({
        'customer_id': [str(uuid.uuid4())[:8] for _ in range(n_customers)],
        'customer_unique_id': [str(uuid.uuid4())[:8] for _ in range(n_customers)],
        'customer_zip_code_prefix': np.random.randint(1000, 99999, n_customers),
        'customer_city': np.random.choice(['sao paulo', 'rio de janeiro', 'belo horizonte', 'curitiba', 'porto alegre'], n_customers),
        'customer_state': np.random.choice(['SP', 'RJ', 'MG', 'PR', 'RS', 'SC', 'ES', 'GO', 'BA', 'PE'], n_customers)
    })
    customers.to_csv(os.path.join(data_dir, 'olist_customers_dataset.csv'), index=False)
    
    # 5. Orders
    start_date = datetime(2017, 1, 1)
    orders = []
    for i in range(n_orders):
        purchase_time = start_date + timedelta(days=np.random.randint(0, 600), hours=np.random.randint(0, 24))
        status = np.random.choice(['delivered', 'shipped', 'canceled', 'processing', 'invoiced'], p=[0.95, 0.02, 0.01, 0.01, 0.01])
        
        delivered_time = purchase_time + timedelta(days=np.random.randint(3, 15)) if status == 'delivered' else pd.NaT
        estimated_time = purchase_time + timedelta(days=np.random.randint(10, 25))
        
        orders.append({
            'order_id': str(uuid.uuid4())[:8],
            'customer_id': np.random.choice(customers['customer_id']),
            'order_status': status,
            'order_purchase_timestamp': purchase_time,
            'order_approved_at': purchase_time + timedelta(hours=np.random.randint(0, 5)),
            'order_delivered_carrier_date': purchase_time + timedelta(days=np.random.randint(1, 3)),
            'order_delivered_customer_date': delivered_time,
            'order_estimated_delivery_date': estimated_time
        })
    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(os.path.join(data_dir, 'olist_orders_dataset.csv'), index=False)
    
    # 6. Order Items
    order_items = []
    for order_id in orders_df['order_id']:
        n_items = np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
        for j in range(n_items):
            order_items.append({
                'order_id': order_id,
                'order_item_id': j + 1,
                'product_id': np.random.choice(products['product_id']),
                'seller_id': np.random.choice(sellers['seller_id']),
                'shipping_limit_date': datetime.now() + timedelta(days=7),
                'price': np.random.uniform(20, 500),
                'freight_value': np.random.uniform(5, 50)
            })
    items_df = pd.DataFrame(order_items)
    items_df.to_csv(os.path.join(data_dir, 'olist_order_items_dataset.csv'), index=False)
    
    # 7. Payments
    payments = []
    for order_id in orders_df['order_id']:
        items_total = items_df[items_df['order_id'] == order_id]
        total_val = items_total['price'].sum() + items_total['freight_value'].sum()
        payments.append({
            'order_id': order_id,
            'payment_sequential': 1,
            'payment_type': np.random.choice(['credit_card', 'boleto', 'voucher', 'debit_card'], p=[0.75, 0.15, 0.05, 0.05]),
            'payment_installments': np.random.randint(1, 12),
            'payment_value': total_val
        })
    payments_df = pd.DataFrame(payments)
    payments_df.to_csv(os.path.join(data_dir, 'olist_order_payments_dataset.csv'), index=False)
    
    # 8. Reviews
    reviews = []
    messages = [
        "Great product!", "Fast delivery.", "Excellent service.", "Very satisfied.", 
        "Arrived late.", "Product damaged.", "Bad quality.", "Never arrived.", 
        "Okay product.", "Recommend."
    ]
    for order_id in orders_df['order_id']:
        if np.random.random() < 0.9:
            score = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.05, 0.1, 0.2, 0.55])
            reviews.append({
                'review_id': str(uuid.uuid4())[:8],
                'order_id': order_id,
                'review_score': score,
                'review_comment_title': "Review",
                'review_comment_message': np.random.choice(messages) if score < 3 or np.random.random() < 0.2 else "",
                'review_creation_date': datetime.now(),
                'review_answer_timestamp': datetime.now()
            })
    reviews_df = pd.DataFrame(reviews)
    reviews_df.to_csv(os.path.join(data_dir, 'olist_order_reviews_dataset.csv'), index=False)
    print("Generation complete.")

if __name__ == "__main__":
    generate_data()
