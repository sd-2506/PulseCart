import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import io
from data_loader import load_data

# Shared Plotly configuration to hide toolbar except download button
plotly_config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 
        'autoScale', 'resetScale2d', 'hoverClosestCartesian', 
        'hoverCompareCartesian', 'toggleSpikelines'
    ]
}

# 1. Page Config
st.set_page_config(
    page_title="PulseCart",
    page_icon="🛒",
    layout="wide"
)

# 7. Custom CSS to hide default elements and style metrics
st.markdown("""
<style>
    /* Hide Streamlit default header, footer, and menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Sidebar glassmorphism styling */
    [data-testid="stSidebar"] {
        background: rgba(17, 17, 17, 0.7) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        background-color: #000000;
    }
    
    /* Custom styling for KPI metrics */
    div[data-testid="metric-container"] {
        background-color: #111111;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom: 3px solid #9c27b0 !important;
        padding: 20px;
        border-radius: 15px;
        transition: transform 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-bottom: 3px solid #FFEB3B !important;
    }

    /* Improved dataframe and text */
    [data-testid="stDataFrame"] {
        font-size: 0.9rem;
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    .stMarkdown {
        color: #CCCCCC;
    }
</style>
""", unsafe_allow_html=True)

# Function to safely load data
def get_data():
    try:
        return load_data('data')
    except Exception as e:
        return None

def generate_alerts(df):
    """
    Analyzes the dataframe for business anomalies and returns a list of alert dictionaries.
    """
    alerts = []
    if df.empty:
        return alerts
        
    # Reference date for 'recent' logic (historical dataset)
    now = df['order_purchase_timestamp'].max()
    thirty_days_ago = now - pd.Timedelta(days=30)
    last_30_days = df[df['order_purchase_timestamp'] >= thirty_days_ago]
    
    # Alert 1 — Revenue drop
    rev_trend = df.drop_duplicates(subset=['order_id', 'payment_sequential']).groupby('order_month')['payment_value'].sum().sort_index()
    if len(rev_trend) >= 4:
        last_month_rev = rev_trend.iloc[-1]
        rolling_avg = rev_trend.iloc[-4:-1].mean()
        if rolling_avg > 0:
            decline = (rolling_avg - last_month_rev) / rolling_avg
            if decline > 0.20:
                alerts.append({
                    "level": "critical",
                    "message": f"Revenue for {rev_trend.index[-1]} is significantly below the 3-month average.",
                    "metric_value": last_month_rev,
                    "threshold": rolling_avg * 0.8
                })

    # Alert 2 — Review score decline
    if not last_30_days.empty:
        avg_review = last_30_days.drop_duplicates(subset=['order_id', 'review_id'])['review_score'].mean()
        if avg_review < 3.5:
            alerts.append({
                "level": "warning",
                "message": "Average customer satisfaction (Review Score) has dropped below target in the last 30 days.",
                "metric_value": avg_review,
                "threshold": 3.5
            })

    # Alert 3 — Late delivery spike
    if not last_30_days.empty:
        late_rate = last_30_days['is_late'].mean() * 100
        if late_rate > 25:
            alerts.append({
                "level": "critical",
                "message": "Late delivery rate has exceeded 25% in the last 30 days.",
                "metric_value": late_rate,
                "threshold": 25.0
            })

    # Alert 4 — Category sales cliff
    if len(rev_trend) >= 2:
        m1, m2 = rev_trend.index[-2:]
        top_5_cats = df.groupby('product_category_name_english')['payment_value'].sum().nlargest(5).index
        cat_rev_m = df[df['order_month'].isin([m1, m2])].groupby(['product_category_name_english', 'order_month'])['payment_value'].sum().unstack()
        
        for cat in top_5_cats:
            if cat in cat_rev_m.index and m1 in cat_rev_m.columns and m2 in cat_rev_m.columns:
                r1, r2 = cat_rev_m.loc[cat, m1], cat_rev_m.loc[cat, m2]
                if r1 > 0:
                    decline = (r1 - r2) / r1
                    if decline > 0.30:
                        alerts.append({
                            "level": "warning",
                            "message": f"Top category '{cat}' shows a {decline:.1%} revenue decline MoM.",
                            "metric_value": decline * 100,
                            "threshold": 30.0
                        })

    # Alert 5 — Cancellation surge
    if not last_30_days.empty:
        recent_total = last_30_days['order_id'].nunique()
        recent_cancelled = last_30_days[last_30_days['order_status'] == 'canceled']['order_id'].nunique()
        cancel_rate = (recent_cancelled / recent_total * 100) if recent_total > 0 else 0
        if cancel_rate > 10:
            alerts.append({
                "level": "critical",
                "message": "Order cancellation rate is abnormally high (>10%) in the last 30 days.",
                "metric_value": cancel_rate,
                "threshold": 10.0
            })
            
    return alerts

@st.cache_data
def forecast_revenue(df, periods):
    """
    Fits a Prophet model to daily revenue data and predicts the next 'periods' days.
    """
    # Aggregate to daily total revenue
    daily_rev = df.drop_duplicates(subset=['order_id', 'payment_sequential']).groupby(df['order_purchase_timestamp'].dt.date)['payment_value'].sum().reset_index()
    daily_rev.columns = ['ds', 'y']
    daily_rev['ds'] = pd.to_datetime(daily_rev['ds'])
    
    # Initialize and fit Prophet model
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(daily_rev)
    
    # Create future dataframe and predict
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast, daily_rev

@st.cache_data
def generate_excel_report(df):
    """
    Generates a stylized multi-sheet Excel report using xlsxwriter.
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Styles
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#534AB7', 'font_color': 'white', 'border': 1})
        alt_fmt = workbook.add_format({'bg_color': '#F8F7FF'})
        money_fmt = workbook.add_format({'num_format': 'R$ #,##0.00'})
        
        def write_styled_sheet(name, data, formats=None):
            data.to_excel(writer, sheet_name=name, index=False)
            worksheet = writer.sheets[name]
            worksheet.freeze_panes(1, 0)
            
            # Format header
            for col_num, value in enumerate(data.columns.values):
                worksheet.write(0, col_num, value, header_fmt)
            
            # Format rows (alternating) and auto-fit
            for row_num in range(1, len(data) + 1):
                if row_num % 2 == 0:
                    worksheet.set_row(row_num, None, alt_fmt)
            
            for i, col in enumerate(data.columns):
                max_len = max(data[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, min(max_len, 50))

        # Sheet 1: KPI Summary
        # Note: using the same logic as the main dashboard
        rev = df.drop_duplicates(subset=['order_id', 'order_item_id'])['price'].sum()
        orders = df['order_id'].nunique()
        aov = rev / orders if orders > 0 else 0
        review = df['review_score'].mean()
        ot_rate = (df['is_late'] == 0).mean()
        top_cat_rev = df.groupby('product_category_name_english')['price'].sum()
        top_cat = top_cat_rev.idxmax() if not top_cat_rev.empty else "N/A"
        
        kpi_df = pd.DataFrame({
            'Metric': ['Total Revenue', 'Total Orders', 'Avg Order Value', 'Avg Review Score', 'On-Time Rate', 'Top Category'],
            'Value': [f"R$ {rev:,.2f}", f"{orders:,}", f"R$ {aov:,.2f}", f"{review:.2f}", f"{ot_rate:.1%}", top_cat]
        })
        write_styled_sheet('KPI Summary', kpi_df)

        # Sheet 2: Monthly Revenue
        monthly_rev = df.drop_duplicates(subset=['order_id', 'payment_sequential']).groupby('order_month')['payment_value'].sum().reset_index()
        monthly_rev.columns = ['Month', 'Revenue (R$)']
        write_styled_sheet('Monthly Revenue', monthly_rev)

        # Sheet 3: Top 20 Categories
        top_cats = df.groupby('product_category_name_english')['price'].sum().sort_values(ascending=False).head(20).reset_index()
        top_cats.columns = ['Category', 'Revenue (R$)']
        write_styled_sheet('Top 20 Categories', top_cats)

        # Sheet 4: RFM Segment Summary
        # Use simplified version for report
        snapshot = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
        rfm_data = df.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': lambda x: (snapshot - x.max()).days,
            'order_id': 'nunique',
            'payment_value': 'sum'
        }).reset_index()
        rfm_data.columns = ['ID', 'Recency', 'Frequency', 'Monetary']
        # For brevity, just sum stats in the sheet
        rfm_summary = pd.DataFrame({
            'Statistic': ['Avg Recency', 'Avg Frequency', 'Avg Monetary'],
            'Value': [f"{rfm_data['Recency'].mean():.1f}d", f"{rfm_data['Frequency'].mean():.2f}", f"R$ {rfm_data['Monetary'].mean():.2f}"]
        })
        write_styled_sheet('Customer Stats', rfm_summary)

        # Sheet 5: Seller Performance
        seller_perf = df.groupby('seller_id').agg({
            'order_id': 'nunique',
            'is_late': 'mean',
            'review_score': 'mean'
        }).sort_values('order_id', ascending=False).head(50).reset_index()
        seller_perf.columns = ['Seller ID', 'Orders', 'Late Rate', 'Avg Review']
        write_styled_sheet('Top Sellers', seller_perf)

    return output.getvalue()

# Step 3 — Loading UX
with st.spinner("Analyzing PulseCart datasets..."):
    df = get_data()

if df is not None and not df.empty:
    num_orders = df['order_id'].nunique()
    num_months = df['order_month'].nunique()
    st.toast(f"Ready — {num_orders:,} orders loaded across {num_months} months.", icon="✅")
elif df is None:
    st.info("Awaiting dataset upload or discovery in /data folder.")
    st.stop()
else:
    st.error("Dataset appears to be empty or malformed.")
    st.stop()

# 2. Header
st.title("🛒 PulseCart")
st.markdown("### E-commerce Intelligence Platform")
st.markdown("---")

# Pre-compute filter options
min_date = df['order_purchase_timestamp'].min().date()
max_date = df['order_purchase_timestamp'].max().date()

top_categories = df['product_category_name_english'].value_counts().nlargest(20).index.tolist()
all_statuses = df['order_status'].dropna().unique().tolist()

# Reset filter callback
def reset_filters():
    st.session_state.date_range = (min_date, max_date)
    st.session_state.category_filter = []
    st.session_state.status_filter = []

# Initialize session state for filters to enable resetting via widget keys
if 'date_range' not in st.session_state:
    st.session_state.date_range = (min_date, max_date)
if 'category_filter' not in st.session_state:
    st.session_state.category_filter = []
if 'status_filter' not in st.session_state:
    st.session_state.status_filter = []

# 3. Sidebar
with st.sidebar:
    st.header("Global Filters")
    
    st.button("Reset Filters", on_click=reset_filters, use_container_width=True)
    
    date_range = st.date_input(
        "Date Range",
        min_value=min_date,
        max_value=max_date,
        key="date_range"
    )
    
    category_filter = st.multiselect(
        "Product Category (Top 20)",
        options=top_categories,
        key="category_filter"
    )
    
    status_filter = st.multiselect(
        "Order Status",
        options=all_statuses,
        key="status_filter"
    )
    
    st.markdown("---")
    
    # 5. Sidebar Navigation (Pages)
    page = st.radio(
        "Navigation",
        options=["Overview", "Product Intelligence", "Customer Intelligence", "Operational Health"]
    )
    
    # 6. Export & Reports Panel
    st.markdown("---")
    st.subheader("📊 Export & Reports")
    
    # Since df_filtered is defined AFTER this block in the current flow, 
    # we need to make sure we use a dummy or handle the order.
    # No-op, panel added below after df_filtered is computed

# 4. Apply Filters
df_filtered = df.copy()

# Date Range Filter
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1, microseconds=-1)
    df_filtered = df_filtered[
        (df_filtered['order_purchase_timestamp'] >= start_dt) & 
        (df_filtered['order_purchase_timestamp'] <= end_dt)
    ]
elif isinstance(date_range, tuple) and len(date_range) == 1:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = start_dt + pd.Timedelta(days=1, microseconds=-1)
    df_filtered = df_filtered[
        (df_filtered['order_purchase_timestamp'] >= start_dt) & 
        (df_filtered['order_purchase_timestamp'] <= end_dt)
    ]

# Category Filter
if category_filter:
    df_filtered = df_filtered[df_filtered['product_category_name_english'].isin(category_filter)]

# Status Filter
if status_filter:
    df_filtered = df_filtered[df_filtered['order_status'].isin(status_filter)]

# 5. Export & Reports Panel (Sidebar)
with st.sidebar:
    # CSV Export
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Export",
        data=csv_data,
        file_name=f"pulsecart_export_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Excel Export
    excel_report = generate_excel_report(df_filtered)
    st.download_button(
        label="📊 Download Excel Report (.xlsx)",
        data=excel_report,
        file_name=f"pulsecart_report_{datetime.date.today()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    
    # Executive Summary
    with st.expander("📝 Generate executive summary"):
        if st.button("Compute Summary"):
            s_rev = df_filtered.drop_duplicates(subset=['order_id', 'order_item_id'])['price'].sum()
            s_orders = df_filtered['order_id'].nunique()
            cat_rev_all = df_filtered.groupby('product_category_name_english')['price'].sum()
            s_top_cat = cat_rev_all.idxmax() if not cat_rev_all.empty else "N/A"
            s_top_share = (cat_rev_all.max() / s_rev * 100) if s_rev > 0 else 0
            s_review = df_filtered['review_score'].mean()
            s_ot = (1 - df_filtered['is_late'].mean()) * 100
            
            s_risk = df_filtered.groupby('seller_id').agg({'is_late': 'mean', 'review_score': 'mean'})
            s_flagged = ((s_risk['is_late'] > 0.3) & (s_risk['review_score'] < 3)).sum()
            
            st.write(f"""
            In the selected period, PulseCart recorded **{s_orders:,}** orders totalling **R$ {s_rev:,.2f}** in revenue. 
            The top-performing category was **{s_top_cat}** contributing **{s_top_share:.1f}%** of total revenue. 
            Average customer review score was **{s_review:.2f}/5**. 
            On-time delivery rate stood at **{s_ot:.1f}%**. 
            **{s_flagged}** seller accounts were flagged for risk based on late delivery and low review scores.
            """)


# 6. KPI Ribbon
# Use unique subset checks to avoid inflating metrics due to multiple joins
total_orders = df_filtered['order_id'].nunique()

# Revenue = sum of unique order items' prices
revenue = df_filtered.drop_duplicates(subset=['order_id', 'order_item_id'])['price'].sum()

# Avg Order Value
aov = revenue / total_orders if total_orders > 0 else 0.0

# Avg Review Score
avg_review = df_filtered.dropna(subset=['review_score']).drop_duplicates(subset=['order_id', 'review_id'])['review_score'].mean()

# Display KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", f"{total_orders:,}")
with col2:
    st.metric("Total Revenue", f"R$ {revenue:,.2f}")
with col3:
    st.metric("Avg Order Value", f"R$ {aov:,.2f}")
with col4:
    st.metric("Avg Review Score", f"{avg_review:.2f}" if pd.notna(avg_review) else "N/A")

st.markdown("---")

# Render the active page
st.subheader(page)
if page == "Overview":
    # Smart Alerts Section
    with st.expander("🔔 Smart alerts — click to expand", expanded=False):
        alerts = generate_alerts(df_filtered)
        if alerts:
            for alert in alerts:
                icon = "🚨" if alert['level'] == "critical" else "⚠️"
                msg = f"**{icon} {alert['message']}**"
                detail = f"Current: {alert['metric_value']:.2f} | Threshold: {alert['threshold']:.2f}"
                
                if alert['level'] == "critical":
                    st.error(f"{msg}  \n{detail}")
                else:
                    st.warning(f"{msg}  \n{detail}")
        else:
            st.success("✅ All systems healthy — no anomalies detected")
        
        st.divider()
        col_t1, col_t2 = st.columns([3, 1])
        col_t1.write(f"Last checked: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if col_t2.button("Refresh Alerts"):
            st.rerun()

    st.markdown("---")

    # 1. Revenue over time
    rev_trend = df_filtered.drop_duplicates(subset=['order_id', 'payment_sequential']).groupby('order_month')['payment_value'].sum().reset_index()
    rev_trend['rolling_avg'] = rev_trend['payment_value'].rolling(window=3).mean()
    
    fig1 = px.line(rev_trend, x='order_month', y='payment_value', title='Monthly Revenue Trend', template='plotly_dark')
    fig1.add_scatter(x=rev_trend['order_month'], y=rev_trend['rolling_avg'], name='3-Month Rolling Avg', line=dict(dash='dash', color='#FFEB3B'))
    fig1.update_traces(line_color='#9c27b0', selector=dict(type='scatter', name='payment_value'))
    fig1.update_layout(xaxis_title="Month", yaxis_title="Revenue (R$)", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=True)
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(showgrid=False)
    st.plotly_chart(fig1, use_container_width=True, config=plotly_config)
    st.caption("Analyst Insight: Revenue shows consistent growth with a notable spike during holiday seasons.")

    col_a, col_b = st.columns(2)
    
    with col_a:
        # 2. Order volume vs revenue
        vol_rev = df_filtered.groupby('order_month').agg({'order_id': 'nunique'}).reset_index()
        vol_rev = vol_rev.merge(rev_trend, on='order_month')
        
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=vol_rev['order_month'], y=vol_rev['order_id'], name="Order Volume", marker_color='#1D9E75'), secondary_y=False)
        fig2.add_trace(go.Scatter(x=vol_rev['order_month'], y=vol_rev['payment_value'], name="Revenue", line=dict(color='#9c27b0')), secondary_y=True)
        fig2.update_layout(title_text="Order Volume vs Revenue", template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=False, secondary_y=False)
        fig2.update_yaxes(showgrid=False, secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True, config=plotly_config)
        st.caption("Analyst Insight: Order count and revenue are highly correlated, indicating stable average order values.")

    with col_b:
        # 3. Revenue by payment type
        target_payments = ['credit_card', 'boleto', 'voucher', 'debit_card']
        pay_type = df_filtered[df_filtered['payment_type'].isin(target_payments)].drop_duplicates(subset=['order_id', 'payment_sequential']).groupby('payment_type')['payment_value'].sum().sort_values(ascending=False).reset_index()
        
        fig3 = px.bar(pay_type, x='payment_value', y='payment_type', orientation='h', title='Revenue by Payment Type', color_discrete_sequence=['#9c27b0'], template='plotly_dark')
        fig3.update_layout(xaxis_title="Revenue (R$)", yaxis_title="Payment Type", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'})
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=False)
        st.plotly_chart(fig3, use_container_width=True, config=plotly_config)
        st.caption("Analyst Insight: Credit card is the dominant payment method, contributing to the bulk of the revenue.")

    col_c, col_d = st.columns(2)
    
    with col_c:
        # 4. Top 10 product categories by revenue
        cat_rev = df_filtered.drop_duplicates(subset=['order_id', 'order_item_id']).groupby('product_category_name_english')['price'].sum().sort_values(ascending=False).head(10).reset_index()
        fig4 = px.bar(cat_rev, x='price', y='product_category_name_english', orientation='h', title='Top 10 Categories by Revenue', color='price', color_continuous_scale=['#9c27b0', '#FFEB3B'], template='plotly_dark')
        fig4.update_layout(xaxis_title="Revenue (R$)", yaxis_title="Category", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
        fig4.update_xaxes(showgrid=False)
        fig4.update_yaxes(showgrid=False)
        st.plotly_chart(fig4, use_container_width=True, config=plotly_config)
        st.caption("Analyst Insight: Health & Beauty and Bed Bath Table are the top revenue-generating categories.")

    with col_d:
        # 5. Orders by day of week
        df_dow = df_filtered.copy()
        df_dow['day_of_week'] = df_dow['order_purchase_timestamp'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_orders = df_dow.groupby('day_of_week')['order_id'].nunique().reindex(day_order).reset_index()
        
        fig5 = px.bar(dow_orders, x='day_of_week', y='order_id', title='Orders by Day of Week', color_discrete_sequence=['#1D9E75'])
        fig5.update_layout(xaxis_title="Day of Week", yaxis_title="Order Count", plot_bgcolor='rgba(0,0,0,0)')
        fig5.update_xaxes(showgrid=False)
        fig5.update_yaxes(showgrid=False)
        st.plotly_chart(fig5, use_container_width=True, config=plotly_config)
        st.caption("Analyst Insight: Order activity peaks during weekdays, particularly on Mondays.")

    # Forecasting Section
    st.markdown("---")
    st.subheader("🔮 Predictive Revenue Forecasting")
    
    # Check for sufficient data (Prophet needs at least two cycles for seasonality usually, but 60 days is minimum)
    if df_filtered['order_purchase_timestamp'].nunique() < 60:
        st.warning("Select a wider date range to enable forecasting — minimum 60 days required.")
    else:
        horizon = st.slider("Forecast horizon (days)", 30, 180, 90)
        
        with st.spinner("Generating statistical forecast using Prophet..."):
            try:
                forecast_df, actual_daily = forecast_revenue(df_filtered, horizon)
                
                # Split forecast into historical and future
                max_actual_date = actual_daily['ds'].max()
                pred_only = forecast_df[forecast_df['ds'] > max_actual_date]
                
                fig_fc = go.Figure()
                
                # Shaded Confidence Interval
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
                    y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(83, 74, 183, 0.1)',
                    line_color='rgba(255,255,255,0)',
                    name='Confidence Interval'
                ))
                
                # Actual Data
                fig_fc.add_trace(go.Scatter(
                    x=actual_daily['ds'], 
                    y=actual_daily['y'], 
                    name='Historical Actuals', 
                    line=dict(color='#9c27b0', width=2)
                ))
                
                # Predicted Data
                fig_fc.add_trace(go.Scatter(
                    x=pred_only['ds'], 
                    y=pred_only['yhat'], 
                    name='Future Forecast', 
                    line=dict(color='#FFEB3B', width=2, dash='dash')
                ))
                
                fig_fc.update_layout(
                    title=f'{horizon}-Day Revenue Forecast',
                    xaxis_title='Date',
                    yaxis_title='Revenue (R$)',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_fc, use_container_width=True, config=plotly_config)
                st.caption("This is a statistical forecast based on historical patterns using Facebook Prophet. Actual results may vary.")
                
                # Forecast metrics
                fm1, fm2, fm3 = st.columns(3)
                
                # 1. Total revenue next 30 days
                next_30_rev = pred_only[pred_only['ds'] <= max_actual_date + pd.Timedelta(days=30)]['yhat'].sum()
                fm1.metric("Predicted Rev (Next 30d)", f"R$ {next_30_rev:,.0f}")
                
                # 2. Predicted peak day
                peak_day = pred_only.loc[pred_only['yhat'].idxmax()]
                fm2.metric("Predicted Peak Day", peak_day['ds'].strftime('%Y-%m-%d'))
                
                # 3. % Change vs same period last year
                ly_start = max_actual_date - pd.Timedelta(days=365)
                ly_end = ly_start + pd.Timedelta(days=horizon)
                ly_data = df_filtered[(df_filtered['order_purchase_timestamp'] >= ly_start) & (df_filtered['order_purchase_timestamp'] <= ly_end)]
                ly_rev = ly_data.drop_duplicates(subset=['order_id', 'payment_sequential'])['payment_value'].sum()
                
                total_fc_rev = pred_only['yhat'].sum()
                if ly_rev > 0:
                    yoy_chg = (total_fc_rev - ly_rev) / ly_rev * 100
                    fm3.metric("YoY Forecast Change", f"{yoy_chg:+.1f}%")
                else:
                    fm3.metric("YoY Forecast Change", "N/A")
                    
            except Exception as e:
                st.error(f"Could not generate forecast: {e}")
elif page == "Product Intelligence":
    # Section 1 — BCG-style scatter plot
    cat_stats = df_filtered.groupby('product_category_name_english').agg({
        'order_id': 'nunique',
        'review_score': 'mean',
        'price': 'sum'
    }).rename(columns={'order_id': 'order_count', 'review_score': 'avg_review', 'price': 'total_revenue'}).reset_index()

    if not cat_stats.empty:
        median_vol = cat_stats['order_count'].median()
        median_review = cat_stats['avg_review'].median()

        fig_bcg = px.scatter(
            cat_stats, 
            x='order_count', 
            y='avg_review', 
            size='total_revenue', 
            hover_name='product_category_name_english',
            title='Category Strategic Positioning (BCG-style Matrix)',
            labels={'order_count': 'Order Volume', 'avg_review': 'Avg Review Score'},
            color_discrete_sequence=['#9c27b0']
        )
        
        # Add quadrant lines
        fig_bcg.add_vline(x=median_vol, line_dash="dash", line_color="gray")
        fig_bcg.add_hline(y=median_review, line_dash="dash", line_color="gray")
        
        # Annotate quadrants
        fig_bcg.add_annotation(x=cat_stats['order_count'].max(), y=cat_stats['avg_review'].max(), text="<b>Stars</b>", showarrow=False, font=dict(color="#1D9E75", size=14), xanchor="right")
        fig_bcg.add_annotation(x=cat_stats['order_count'].min(), y=cat_stats['avg_review'].max(), text="<b>Question Marks</b>", showarrow=False, font=dict(color="#D85A30", size=14), xanchor="left")
        fig_bcg.add_annotation(x=cat_stats['order_count'].max(), y=cat_stats['avg_review'].min(), text="<b>Cash Cows</b>", showarrow=False, font=dict(color="#534AB7", size=14), xanchor="right")
        fig_bcg.add_annotation(x=cat_stats['order_count'].min(), y=cat_stats['avg_review'].min(), text="<b>Dogs</b>", showarrow=False, font=dict(color="gray", size=14), xanchor="left")

        fig_bcg.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bcg, use_container_width=True, config=plotly_config)
        st.caption("BCG Matrix: Categorizes products by volume (market penetration) and satisfaction (perceived quality).")

    # Section 2 — Price vs review score scatter
    st.markdown("---")
    st.subheader("Price vs. Satisfaction Analysis")
    
    highlight_cats = st.multiselect(
        "Highlight Specific Categories (Select up to 3)",
        options=top_categories,
        max_selections=3
    )
    
    # Prepare product-level data
    prod_data = df_filtered.dropna(subset=['price', 'review_score']).copy()
    if highlight_cats:
        prod_data = prod_data[prod_data['product_category_name_english'].isin(highlight_cats)]
    
    # Limit data points for scatter to maintain performance if no filters
    if not highlight_cats and len(prod_data) > 5000:
        prod_data = prod_data.sample(5000)

    fig_price_review = px.scatter(
        prod_data,
        x='price',
        y='review_score',
        color='product_category_name_english',
        trendline="ols",
        title='Price vs. Review Score Correlation',
        labels={'price': 'Price (R$)', 'review_score': 'Review Score'},
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Prism,
        template='plotly_dark'
    )
    fig_price_review.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_price_review, use_container_width=True, config=plotly_config)
    st.caption("Price vs. Review: The trendline illustrates whether premium pricing impacts overall customer satisfaction scores.")

    # Section 3 — Category performance table
    st.markdown("---")
    st.subheader("Category Performance Deep Dive")
    
    df_temp = df_filtered.copy()
    df_temp['is_low'] = df_temp['review_score'] <= 2
    cat_perf = df_temp.groupby('product_category_name_english').agg({
        'price': ['sum', 'mean'],
        'order_id': 'nunique',
        'review_score': 'mean',
        'is_low': 'mean'
    }).reset_index()
    
    cat_perf.columns = ['Category', 'Total Revenue', 'Avg Price', 'Order Count', 'Avg Review', 'Low Rating %']
    cat_perf['Low Rating %'] = (cat_perf['Low Rating %'] * 100).round(2)
    cat_perf = cat_perf.sort_values('Total Revenue', ascending=False)
    
    overall_avg_review = df_filtered['review_score'].mean()
    
    def color_review(val):
        color = '#00E676' if val >= overall_avg_review else '#FF5252'
        return f'color: {color}'

    st.dataframe(
        cat_perf.style.format({
            'Total Revenue': 'R$ {:,.2f}',
            'Avg Price': 'R$ {:,.2f}',
            'Order Count': '{:,}',
            'Avg Review': '{:.2f}',
            'Low Rating %': '{:.2f}%'
        }).applymap(color_review, subset=['Avg Review']),
        use_container_width=True,
        height=400
    )

    # Section 4 — Product search
    st.markdown("---")
    st.subheader("Category Search Intelligence")
    search_query = st.text_input("Search category (e.g., 'telephony')", placeholder="Type here...")
    
    if search_query:
        search_results = df_filtered[df_filtered['product_category_name_english'].str.contains(search_query, case=False, na=False)]
        if not search_results.empty:
            res_units = len(search_results)
            res_price = search_results['price'].mean()
            res_rating = search_results['review_score'].mean()
            
            # Find top customer city
            top_city_series = search_results['customer_city'].mode()
            res_city = top_city_series[0] if not top_city_series.empty else "N/A"
            
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Items Sold", f"{res_units:,}")
            s_col2.metric("Avg Price", f"R$ {res_price:.2f}")
            s_col3.metric("Avg Rating", f"{res_rating:.2f} ⭐")
            s_col4.metric("Top Customer City", res_city)
        else:
            st.warning("No data found for this category search.")
elif page == "Customer Intelligence":
    # Section 1 — RFM Segmentation
    st.subheader("Customer Strategic Segmentation (RFM Analysis)")
    
    # Calculate RFM metrics at customer_unique_id level
    # Use a snapshot date of max purchase date + 1 day
    snapshot_date = df_filtered['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    
    # First, get unique orders and their spend
    order_spend = df_filtered.drop_duplicates(subset=['order_id', 'payment_sequential']).groupby(['customer_unique_id', 'order_id', 'order_purchase_timestamp'])['payment_value'].sum().reset_index()
    
    rfm = order_spend.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'payment_value': 'sum'
    }).reset_index()
    rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

    # Scoring 1-5 (5 is best)
    # Recency: lower days is better (closer to 5)
    rfm['R'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    # Frequency: higher is better. Many are 1, so use rank-based qcut
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    # Monetary: higher is better
    rfm['M'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Calculate average RFM score
    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].astype(int).mean(axis=1)

    # Segment Mapping
    def assign_segment(score):
        if score >= 4.0: return 'Champions'
        if score >= 3.0: return 'Loyal Customers'
        if score >= 1.5: return 'At Risk'
        return 'Lost'

    rfm['Segment'] = rfm['RFM_Score'].apply(assign_segment)

    # Visualization: Treemap
    segment_stats = rfm.groupby('Segment').agg({
        'customer_unique_id': 'count',
        'Monetary': 'mean',
        'Recency': 'mean',
        'Frequency': 'mean'
    }).reset_index().rename(columns={'customer_unique_id': 'Customer Count', 'Monetary': 'Avg Monetary'})
    
    fig_rfm = px.treemap(
        segment_stats, 
        path=['Segment'], 
        values='Customer Count',
        color='Avg Monetary',
        color_continuous_scale=['#111111', '#9c27b0'],
        template='plotly_dark',
        title='Customer Segment Distribution & Value'
    )
    fig_rfm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_rfm, use_container_width=True, config=plotly_config)
    
    # Summary Table
    st.dataframe(
        segment_stats.style.format({
            'Customer Count': '{:,}',
            'Avg Monetary': 'R$ {:,.2f}',
            'Recency': '{:.1f} days',
            'Frequency': '{:.2f} orders'
        }),
        use_container_width=True
    )

    # Section 2 — Geographic distribution
    st.markdown("---")
    st.subheader("Market Penetration by Geography")
    
    state_data = df_filtered.groupby('customer_state')['order_id'].nunique().sort_values(ascending=False).reset_index()
    fig_state = px.bar(
        state_data, 
        x='customer_state', 
        y='order_id', 
        title='Orders by State (Brazil)',
        labels={'customer_state': 'State', 'order_id': 'Total Orders'},
        color_discrete_sequence=['#9c27b0'],
        template='plotly_dark'
    )
    fig_state.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_state, use_container_width=True, config=plotly_config)
    st.caption("Geographic insight: Customer base is heavily concentrated in the Southeast region (SP, RJ, MG).")

    # Section 3 — Customer Lifetime Value (CLV)
    st.markdown("---")
    st.subheader("Customer Spend & Pareto Analysis")
    
    p80 = rfm['Monetary'].quantile(0.8)
    total_revenue = rfm['Monetary'].sum()
    top_20_pct_revenue = (rfm[rfm['Monetary'] >= p80]['Monetary'].sum() / total_revenue * 100) if total_revenue > 0 else 0
    
    fig_clv = px.histogram(
        rfm, 
        x='Monetary', 
        nbins=100, 
        title='Distribution of Total Spend per Customer',
        color_discrete_sequence=['#9c27b0'],
        template='plotly_dark'
    )
    fig_clv.add_vline(x=p80, line_dash="dash", line_color="#FFEB3B", annotation_text="80th Percentile")
    fig_clv.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Lifetime Spend (R$)", yaxis_title="Customer Count")
    st.plotly_chart(fig_clv, use_container_width=True, config=plotly_config)
    st.info(f"**Pareto Insight**: The top 20% of your customers generate **{top_20_pct_revenue:.1f}%** of your total revenue.")

    # Section 4 — Review score breakdown
    st.markdown("---")
    st.subheader("Customer Sentiment Breakdown")
    
    # Drop duplicates to count one review per order
    sentiment_data = df_filtered.drop_duplicates(subset=['order_id', 'review_id'])['review_score'].value_counts().sort_index().reset_index()
    sentiment_data.columns = ['Score', 'Count']
    
    # Custom color mapping: 5=green, 4=teal, 3=amber, 2=orange, 1=red
    color_map = {5: '#1D9E75', 4: '#008080', 3: '#FFBF00', 2: '#FF8C00', 1: '#D85A30'}
    
    fig_sentiment = px.pie(
        sentiment_data, 
        values='Count', 
        names='Score', 
        hole=0.6,
        title='Review Score Distribution',
        color='Score',
        color_discrete_map=color_map,
        template='plotly_dark'
    )
    fig_sentiment.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_sentiment, use_container_width=True, config=plotly_config)

    # Section 5 — Repeat buyer rate
    st.markdown("---")
    st.subheader("Retention Analysis")
    
    repeat_data = rfm['Frequency'].apply(lambda x: 'Repeat Buyer' if x > 1 else 'One-time Buyer').value_counts().reset_index()
    repeat_data.columns = ['Status', 'Count']
    repeat_rate = (rfm['Frequency'] > 1).mean() * 100
    
    ret_col1, ret_col2 = st.columns([1, 2])
    with ret_col1:
        st.metric("Repeat Buyer Rate", f"{repeat_rate:.1f}%")
        st.write("A healthy repeat buyer rate is crucial for long-term sustainability.")
    with ret_col2:
        fig_retention = px.pie(
            repeat_data, 
            values='Count', 
            names='Status', 
            title='Repeat vs. One-time Buyers',
            color_discrete_sequence=['#9c27b0', '#FFEB3B'],
            template='plotly_dark'
        )
        fig_retention.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_retention, use_container_width=True, config=plotly_config)
elif page == "Operational Health":
    # Section 1 — On-time delivery rate
    st.subheader("Delivery Execution Performance")
    
    # Calculate global on-time rate (skipping undelivered/NaN)
    on_time_rate = (1 - df_filtered['is_late'].mean()) * 100
    
    # Color logic for the large metric
    if on_time_rate > 90:
        rate_color = "#1D9E75" # Green
    elif on_time_rate > 70:
        rate_color = "#FFBF00" # Amber
    else:
        rate_color = "#D85A30" # Red
        
    st.markdown(f"""
        <div style="background-color: {rate_color}; padding: 30px; border-radius: 12px; text-align: center; color: white; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="margin: 0; font-size: 3.5rem; font-weight: 700;">{on_time_rate:.1f}%</h1>
            <p style="margin: 0; font-size: 1.3rem; opacity: 0.9;">Total Orders Delivered On Time</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Monthly trend line of on-time rate
    monthly_ot = df_filtered.groupby('order_month')['is_late'].apply(lambda x: (1 - x.mean()) * 100).reset_index()
    fig_ot_trend = px.line(monthly_ot, x='order_month', y='is_late', title='Monthly On-Time Delivery Trend', template='plotly_dark')
    fig_ot_trend.update_traces(line_color='#9c27b0', line_width=3)
    fig_ot_trend.update_layout(yaxis_title="On-Time Rate (%)", xaxis_title="Month", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_range=[0, 105])
    st.plotly_chart(fig_ot_trend, use_container_width=True, config=plotly_config)

    # Section 2 — Delivery time distribution
    st.markdown("---")
    st.subheader("Logistics Speed & Reliability")
    
    del_data = df_filtered.dropna(subset=['delivery_days'])
    median_del = del_data['delivery_days'].median()
    p90_del = del_data['delivery_days'].quantile(0.9)
    avg_est = (df_filtered['order_estimated_delivery_date'] - df_filtered['order_purchase_timestamp']).dt.total_seconds().mean() / (24*3600)
    
    fig_del_dist = px.histogram(del_data, x='delivery_days', title='Distribution of Days to Delivery', color_discrete_sequence=['#9c27b0'], template='plotly_dark')
    fig_del_dist.add_vline(x=median_del, line_dash="dash", line_color="#FFEB3B", annotation_text=f"Median: {median_del:.1f}d")
    fig_del_dist.add_vline(x=p90_del, line_dash="dash", line_color="#FF5252", annotation_text=f"90th Pct: {p90_del:.1f}d")
    fig_del_dist.add_vline(x=avg_est, line_dash="dash", line_color="gray", annotation_text=f"Avg Estimated: {avg_est:.1f}d")
    fig_del_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Actual Days to Delivery", yaxis_title="Order Volume")
    st.plotly_chart(fig_del_dist, use_container_width=True, config=plotly_config)

    # Section 3 — Late orders by category
    st.markdown("---")
    st.subheader("Category Logistics Bottlenecks")
    
    late_cat = df_filtered.groupby('product_category_name_english')['is_late'].mean().sort_values(ascending=False).head(15).reset_index()
    late_cat['Late %'] = late_cat['is_late'] * 100
    
    fig_late_cat = px.bar(late_cat, x='Late %', y='product_category_name_english', orientation='h', title='Top 15 Most Delayed Categories (%)', color_discrete_sequence=['#FF5252'], template='plotly_dark')
    fig_late_cat.update_layout(xaxis_title="Late Delivery Rate (%)", yaxis_title="Category", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_late_cat, use_container_width=True, config=plotly_config)
    st.caption("Supply Chain Insight: High delay rates in specific categories often point to bulkier items or overseas logistics bottlenecks.")

    # Section 4 — Seller performance table
    st.markdown("---")
    st.subheader("Seller Health & Risk Monitoring")
    
    seller_perf = df_filtered.groupby('seller_id').agg({
        'order_id': 'nunique',
        'delivery_days': 'mean',
        'is_late': 'mean',
        'review_score': 'mean'
    }).reset_index()
    
    seller_perf.columns = ['Seller ID', 'Total Orders', 'Avg Delivery Days', 'Late Rate', 'Avg Review']
    seller_perf['Risk Flag'] = (seller_perf['Late Rate'] > 0.30) & (seller_perf['Avg Review'] < 3.0)
    
    top_sellers = seller_perf.sort_values('Total Orders', ascending=False).head(20)
    
    def style_risk_sellers(row):
        styles = [''] * len(row)
        # Highlight Late Rate > 30% or Review < 3 in Red
        if row['Late Rate'] > 0.30:
            styles[3] = 'background-color: #fce4e4; color: #cc0000; font-weight: bold'
        if row['Avg Review'] < 3.0:
            styles[4] = 'background-color: #fce4e4; color: #cc0000; font-weight: bold'
        return styles

    st.dataframe(
        top_sellers.style.format({
            'Total Orders': '{:,}',
            'Avg Delivery Days': '{:.1f}d',
            'Late Rate': '{:.1%}',
            'Avg Review': '{:.2f}'
        }).apply(style_risk_sellers, axis=1),
        use_container_width=True
    )

    # Section 5 — Order status funnel
    st.markdown("---")
    st.subheader("Operational Lifecycle Funnel")
    
    status_order = ['created', 'approved', 'invoiced', 'shipped', 'delivered']
    funnel_counts = df_filtered['order_status'].value_counts().reindex(status_order).fillna(0).reset_index()
    funnel_counts.columns = ['Stage', 'Order Count']
    funnel_counts['Stage'] = funnel_counts['Stage'].str.title()
    
    fig_funnel = px.funnel(funnel_counts, x='Order Count', y='Stage', title='Order Progression Stages', color_discrete_sequence=['#9c27b0'], template='plotly_dark')
    fig_funnel.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_funnel, use_container_width=True, config=plotly_config)

    # Section 6 — Cancellation analysis
    st.markdown("---")
    st.subheader("Cancellation Depth & Root Causes")
    
    cancelled_df = df_filtered[df_filtered['order_status'] == 'canceled']
    monthly_cancel = cancelled_df.groupby('order_month')['order_id'].nunique().reset_index()
    
    fig_cancel = px.bar(monthly_cancel, x='order_month', y='order_id', title='Monthly Cancellations Volume', color_discrete_sequence=['#FF5252'], template='plotly_dark')
    fig_cancel.update_layout(xaxis_title="Month", yaxis_title="Cancelled Orders", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cancel, use_container_width=True, config=plotly_config)
    
    # Word frequency for cancellations
    import re
    from collections import Counter
    
    st.markdown("#### Why are they cancelling? (Top Keywords in Reviews)")
    cancel_comments = cancelled_df['review_comment_message'].dropna().str.lower()
    if not cancel_comments.empty:
        all_words = ' '.join(cancel_comments)
        # Extract words longer than 4 chars
        words = re.findall(r'\b\w{5,}\b', all_words)
        # Filter Portuguese stopwords and common generic words
        stops = {'produto', 'entrega', 'ainda', 'recebi', 'pedido', 'compra', 'chegou', 'muito', 'pelo', 'pela', 'fazer', 'estou'}
        filtered_words = [w for w in words if w not in stops]
        
        top_words = Counter(filtered_words).most_common(10)
        word_freq_df = pd.DataFrame(top_words, columns=['Root Cause Keyword', 'Occurrences'])
        st.table(word_freq_df)
    else:
        st.info("Insufficient review data for cancelled orders to perform keyword analysis.")
