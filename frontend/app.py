import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from backend.predict_lstm import predict_future_sales
from backend.insights import top_products, shop_performance, category_demand, monthly_sales_trend

st.set_page_config(page_title="Brake Sales Dashboard", layout="wide")

st.title("🛠️ Vasu's Brake Sales Prediction Dashboard (LSTM)")

df = pd.read_csv(r"backend/data/brake_sales_clean.csv")

product = st.sidebar.selectbox("Select Brake Product", df["item_name"].unique())
months = st.sidebar.slider("Forecast months", 1, 24, 6)

# -------------------------
# Forecasting
# -------------------------
st.header(f"📈 Sales Forecast for {product} ({months} months)")
prediction = predict_future_sales(product, months)

fig_forecast = px.line(
    y=prediction,
    title=f"{product} LSTM Forecast ({months} months)"
)
st.plotly_chart(fig_forecast)

# -------------------------
# Product Insights
# -------------------------
st.header("📊 Product Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Selling Brake Products")
    st.bar_chart(top_products())

with col2:
    st.subheader("Demand by Category")
    st.bar_chart(category_demand())

# -------------------------
# Shop Insights
# -------------------------
st.header("🏬 Shop Performance")
st.bar_chart(shop_performance())

# -------------------------
# Trends
# -------------------------
st.header(f"📉 Monthly Trend for {product}")
trend = monthly_sales_trend(product)
st.line_chart(trend)
