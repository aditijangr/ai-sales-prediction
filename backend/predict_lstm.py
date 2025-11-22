import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

model_dir = os.path.join(os.path.dirname(__file__), "models")

# load CSV correctly regardless of working directory
df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data", "brake_sales_clean.csv")
)

df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")


def predict_future_sales(product_name, months=6):
    temp = df[df["item_name"] == product_name].groupby("date_block_num")["item_cnt_day"].sum()

    sales = temp.values.reshape(-1, 1)

    scaler_path = os.path.join(
        model_dir,
        f"{product_name.replace(' ', '_')}_scaler.npy"
    )

    scaler_min = np.load(scaler_path)

    scaler = MinMaxScaler()
    scaler.fit(sales)

    model_path = os.path.join(
        model_dir,
        f"{product_name.replace(' ', '_')}.h5"
    )

    model = tf.keras.models.load_model(model_path, compile=False)


    last_12 = sales[-12:].reshape(1, 12, 1)
    last_12_scaled = scaler.transform(last_12.reshape(-1, 1)).reshape(1, 12, 1)

    predictions = []

    for _ in range(months):
        pred = model.predict(last_12_scaled, verbose=0)[0][0]
        predictions.append(pred)

        last_12_scaled = np.append(last_12_scaled[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions.flatten()
