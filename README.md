# PulseCart 🛒 | E-commerce Intelligence Platform

PulseCart is a high-performance, multi-page analytics dashboard designed to transform raw e-commerce data into actionable business intelligence. Built using **Streamlit**, **Pandas**, and **Plotly**, it processes the Brazilian Olist dataset to provide insights across four key strategic pillars: Overview, Products, Customers, and Operations.

---

## 🌟 Key Features

### 1. Executive Overview
- **Real-time KPIs**: Instant visibility into Total Orders, Revenue (R$), Average Order Value (AOV), and Customer Satisfaction.
- **Dynamic Trend Analysis**: Monthly revenue growth charts with 3-month rolling averages.
- **Predictive Forecasting**: 30 to 180-day revenue projections powered by **Facebook Prophet** (Time Series Analysis).
- **Smart Alerts**: Automated detection of critical anomalies (e.g., >20% revenue drop, satisfaction dips, or logistics spikes).

### 2. Product Intelligence
- **Strategic Positioning (BCG Matrix)**: Categorizes products into Stars, Cash Cows, Question Marks, and Dogs based on market volume and satisfaction.
- **Price vs. Satisfaction**: Correlation analysis with OLS trendlines to identify pricing power.
- **Search Intelligence**: Instant lookup for specific category performance metrics.

### 3. Customer Intelligence
- **RFM Segmentation**: Behavioral segmentation (Champions, Loyal, At Risk, Lost) using Recency, Frequency, and Monetary scores.
- **Geographic Distribution**: Heatmaps showing market penetration across Brazil's states.
- **CLV & Pareto Analysis**: Identifies the "Top 20%" of customers generating the bulk of revenue.

### 4. Operational Health
- **Delivery Reliability**: On-time rate tracking with monthly performance trends.
- **Logistics Speed**: Histograms of delivery times with median and 90th percentile benchmarks.
- **Seller Risk Monitoring**: Automated flagging of sellers with high delay rates (>30%) or poor ratings.
- **Cancellation Insights**: Keyword frequency analysis from review comments to identify root causes of order cancellations.

### 5. Enterprise Reporting
- **Automated Summary**: Generates a natural-language executive briefing on demand.
- **Multi-Sheet Excel Reports**: Stylized, branded Excel downloads with formatted data and auto-fit columns.
- **CSV Data Export**: One-click export for deeper offline analysis.

---

## 🛠️ Technology Stack
- **Frontend**: Streamlit (with Custom CSS & Theming)
- **Data Engine**: Pandas, NumPy
- **Visuals**: Plotly Express, Plotly Graph Objects
- **Forecasting**: Facebook Prophet
- **Reporting**: XlsxWriter

---

## 📂 Project Structure
```text
pulsecart/
├── .streamlit/
│   └── config.toml      # Custom branding & theme
├── data/                # Local storage for CSVs
├── data_loader.py       # Core data merging & feature engineering
├── app.py               # Main dashboard logic & UI
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## ⚙️ Setup & Installation

### 1. Dataset Preparation
You have two options to load data:

**Option A: Synthetic Data (Recommended for Testing)**
Run the included generation script to create a sample dataset of 5,000 orders:
```bash
python3 generate_synthetic_data.py
```

**Option B: Real Olist Dataset**
1. Download the [Olist Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. Create a folder named `data/` in the root directory.
3. Extract all CSV files into the `data/` folder.

### 2. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```

### ☁️ Deployment Note
PulseCart is **cloud-ready**. If deployed to environments where local storage is restricted (like Streamlit Cloud), the application will automatically activate **Deployment Mode**, providing a secure file uploader to initialize the dashboard with your CSV files.

---
Built with ❤️ by **Shibabrata Dey**.
