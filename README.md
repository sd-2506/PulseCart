# PulseCart 🛒 | E-commerce Intelligence Platform

PulseCart is a high-performance analytics dashboard designed to transform raw e-commerce data into actionable business intelligence. Developed using **Streamlit**, **Pandas**, and **Plotly**, it processes the Brazilian Olist dataset to provide strategic insights across four key pillars: Overview, Products, Customers, and Operations.

---

## 🌟 Key Features

### 1. Executive Overview
- **Core KPIs**: Tracking Total Orders, Revenue (R$), Average Order Value (AOV), and Satisfaction.
- **Trend Analysis**: Monthly revenue growth monitoring with 3-month rolling averages.
- **Revenue Forecasting**: 30 to 180-day projections powered by **Facebook Prophet**.
- **Business Alerts**: Automated detection of revenue drops, satisfaction dips, and logistics anomalies.

### 2. Product Intelligence
- **BCG Matrix Analysis**: Categorizes products (Stars, Cash Cows, Question Marks, Dogs) based on volume and satisfaction.
- **Price Sensitivity**: Correlation analysis between pricing and review scores.
- **Category Lookup**: Instant performance summaries for specific product categories.

### 3. Customer Intelligence
- **RFM Segmentation**: Behavioral classification (Champions, Loyal, At Risk, Lost) using Recency, Frequency, and Monetary metrics.
- **Geographic Insights**: Market penetration heatmaps across Brazil's states.
- **CLV Analysis**: Identifies high-value customer segments driving revenue growth.

### 4. Operational Health
- **Logistics Performance**: On-time delivery rate tracking and trend analysis.
- **Delivery Distribution**: Speed benchmarks using median and 90th percentile delivery times.
- **Risk Monitoring**: Identification of sellers with high delay rates or low ratings.

### 5. Reporting & Exports
- **Executive Summaries**: Natural-language business briefings generated on demand.
- **Professional Exports**: Multi-sheet, stylized Excel reports and raw CSV data downloads.

---

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **Analysis**: Pandas, NumPy, Scikit-Learn
- **Visualization**: Plotly
- **Forecasting**: Facebook Prophet

---

## ⚙️ Setup & Installation

### 1. Dataset Preparation
**Option A: Synthetic Data (Fast Testing)**
Run the generation script to create a sample dataset of 5,000 orders:
```bash
python3 generate_synthetic_data.py
```

**Option B: Production Data**
1. Download the [Olist Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. Place all CSV files into a `/data` folder in the project root.

### 2. Run Application
```bash
# Install dependencies
pip install -r requirements.txt

# Start dashboard
streamlit run app.py
```

---
Developed by **Shibabrata Dey**
