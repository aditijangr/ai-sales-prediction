import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Load data
data_path = r"backend/data/brake_sales_clean.csv"
df = pd.read_csv(data_path)

df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")

# Monthly aggregated sales
monthly = df.groupby(["date_block_num", "item_name"])["item_cnt_day"].sum().reset_index()
monthly.rename(columns={"item_cnt_day": "sales"}, inplace=True)

# Model Save Directory
model_dir = r"backend/models"
os.makedirs(model_dir, exist_ok=True)


def build_lstm_model():
    model = Sequential()
    model.add(LSTM(64, activation="relu", return_sequences=True, input_shape=(12, 1)))
    model.add(LSTM(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


products = monthly["item_name"].unique()
print("Training LSTM models for brake products...\n")

for product in products:
    print(f"➡ Training model for: {product}")

    temp = monthly[monthly["item_name"] == product]
    sales = temp["sales"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)

    X, y = [], []
    seq_len = 12  # using 12 months history

    for i in range(len(sales_scaled) - seq_len):
        X.append(sales_scaled[i:i+seq_len])
        y.append(sales_scaled[i+seq_len])

    X, y = np.array(X), np.array(y)

    model = build_lstm_model()

    if len(X) == 0:
        print(f"⚠ Not enough data to train {product}. Skipping.")
        continue

    model.fit(X, y, epochs=30, batch_size=8, verbose=0)

    safe_name = product.replace(" ", "_")

    model.save(f"{model_dir}/{safe_name}.h5")
    np.save(f"{model_dir}/{safe_name}_scaler.npy", scaler.data_min_)

print("\n✔ All LSTM models trained and saved!")
