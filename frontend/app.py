import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.predict_models import predict_future_sales_general
from backend.insights import top_products, shop_performance, category_demand, monthly_sales_trend

st.set_page_config(page_title="Brake Sales Dashboard", layout="wide")
st.title("🛠️ Vasu's Brake Sales Prediction Dashboard")

df = pd.read_csv(r"backend/data/brake_sales_clean.csv")

# -------------------------
# Sidebar Tabs
# -------------------------

with st.sidebar:
    st.title("🔧 Controls")

    tab = st.radio(
        "Select View",
        ["AI Forecasting", "Insights"],
        index=0
    )

# ============================================================
# ====================== AI FORECASTING TAB ==================
# ============================================================

if tab == "AI Forecasting":

    st.header("🤖 AI Sequence Model Forecast (Neural Networks)")

    colA, colB = st.columns(2)

    with colA:
        product_seq = st.selectbox("Select Product", df["item_name"].unique())

    with colB:
        months_seq = st.slider("Forecast Months", 1, 36, 12)

    model_choice_seq = st.selectbox(
        "Choose Sequence Model",
        ["lstm", "gru", "cnn", "rnn"]
    )

    if st.button("Run AI Forecast"):
        try:
            seq_pred = predict_future_sales_general(
                product_name=product_seq,
                months=months_seq,
                model=model_choice_seq
            )

            st.subheader(f"{model_choice_seq.upper()} Forecast for {product_seq}")

            fig_ai = px.line(
                y=seq_pred,
                title=f"{product_seq} - {model_choice_seq.upper()} Forecast ({months_seq} months)"
            )
            st.plotly_chart(fig_ai)

        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# ========================= INSIGHTS TAB =====================
# ============================================================

elif tab == "Insights":

    st.header("📊 Product Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Selling Brake Products")
        st.bar_chart(top_products())

    with col2:
        st.subheader("Demand by Category")
        st.bar_chart(category_demand())

    st.header("🏬 Shop Performance")
    st.bar_chart(shop_performance())

    # Trend section
    product = st.selectbox("Select Product for Trend", df["item_name"].unique())
    trend = monthly_sales_trend(product)

    st.header(f"📉 Monthly Trend for {product}")
    st.line_chart(trend)
