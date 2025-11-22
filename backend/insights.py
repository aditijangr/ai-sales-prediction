import pandas as pd
import os

# Correct dynamic path to CSV
df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data", "brake_sales_clean.csv")
)

def top_products(n=5):
    return (
        df.groupby("item_name")["item_cnt_day"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )

def shop_performance():
    return (
        df.groupby("shop_name")["item_cnt_day"]
        .sum()
        .sort_values(ascending=False)
    )

def category_demand():
    return (
        df.groupby("item_category_name")["item_cnt_day"]
        .sum()
        .sort_values(ascending=False)
    )

def monthly_sales_trend(product_name):
    return (
        df[df["item_name"] == product_name]
        .groupby("date_block_num")["item_cnt_day"]
        .sum()
    )
