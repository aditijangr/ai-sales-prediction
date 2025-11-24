import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

# Base paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load dataset globally
df = pd.read_csv(os.path.join(BASE_DIR, "data", "brake_sales_clean.csv"))
df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")


def predict_future_sales_general(product_name, months=12, model="lstm"):
    """
    model options:
    - "lstm"
    - "gru"
    - "cnn"
    - "rnn"
    """

    safe_name = product_name.replace(" ", "_")

    # Filter historical sales by product
    temp = df[df["item_name"] == product_name].groupby("date_block_num")["item_cnt_day"].sum()
    sales = temp.values.reshape(-1, 1)

    # Check length
    if len(sales) < 12:
        raise ValueError("Not enough data to forecast. Need at least 12 months.")

    # Load scaler
    scaler_min_path = os.path.join(MODEL_DIR, f"{safe_name}_scaler.npy")
    scaler_min = np.load(scaler_min_path)

    scaler = MinMaxScaler()
    scaler.fit(sales)

    # Load correct model
    model_path = os.path.join(MODEL_DIR, f"{safe_name}_{model}.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    nn_model = tf.keras.models.load_model(model_path, compile=False)

    # Prepare last 12 months
    last_12 = sales[-12:].reshape(1, 12, 1)
    last_12_scaled = scaler.transform(last_12.reshape(-1, 1)).reshape(1, 12, 1)

    predictions = []

    # Auto-regressive forecasting loop
    for _ in range(months):
        pred = nn_model.predict(last_12_scaled, verbose=0)[0][0]
        predictions.append(pred)

        # Roll the window
        last_12_scaled = np.append(last_12_scaled[:, 1:, :], [[[pred]]], axis=1)

    # Inverse scale predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions.flatten()
