import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    LSTM, GRU, Dense, Conv1D, Flatten,
    Input, LayerNormalization, MultiHeadAttention, Dropout
)
from sklearn.preprocessing import MinMaxScaler
import os

# Load clean aggregated dataset
data_path = r"data/brake_sales_clean.csv"
df = pd.read_csv(data_path)

df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")

# Monthly aggregated sales
monthly = df.groupby(["date_block_num", "item_name"])["item_cnt_day"].sum().reset_index()
monthly.rename(columns={"item_cnt_day": "sales"}, inplace=True)

# Save directory
model_dir = r"backend/models"
os.makedirs(model_dir, exist_ok=True)

SEQ_LEN = 12   # last 12 months as input


# ----------------------------------------------------
# MODEL BUILDERS
# ----------------------------------------------------

def build_lstm():
    model = Sequential([
        LSTM(64, activation="relu", return_sequences=True, input_shape=(SEQ_LEN, 1)),
        LSTM(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru():
    model = Sequential([
        GRU(64, activation="relu", return_sequences=True, input_shape=(SEQ_LEN, 1)),
        GRU(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_cnn():
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(SEQ_LEN, 1)),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_rnn():
    model = Sequential([
        tf.keras.layers.SimpleRNN(64, activation="relu", return_sequences=True, input_shape=(SEQ_LEN, 1)),
        tf.keras.layers.SimpleRNN(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_transformer():
    inputs = Input(shape=(SEQ_LEN, 1))

    # Project input to the transformer dimension
    x = Dense(64)(inputs)

    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attn_output = Dropout(0.2)(attn_output)
    out1 = LayerNormalization()(x + attn_output)

    # Feed-forward network
    ffn = Sequential([
        Dense(128, activation="relu"),
        Dense(64)
    ])

    ffn_output = ffn(out1)
    ffn_output = Dropout(0.2)(ffn_output)
    out2 = LayerNormalization()(out1 + ffn_output)

    # Output neuron
    output = Dense(1)(out2[:, -1, :])

    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    return model

# ----------------------------------------------------
# TRAINING LOOP (for each product)
# ----------------------------------------------------

products = monthly["item_name"].unique()
print("\nTraining sequence models for brake products...\n")

for product in products:
    print(f"➡ {product}")

    temp = monthly[monthly["item_name"] == product]
    sales = temp["sales"].values.reshape(-1, 1)

    # Skip products with insufficient data
    if len(sales) < SEQ_LEN + 1:
        print(f"⚠ Not enough data for {product}. Skipping...")
        continue

    # Scale data
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)

    # Prepare sequences
    X, y = [], []
    for i in range(len(sales_scaled) - SEQ_LEN):
        X.append(sales_scaled[i:i+SEQ_LEN])
        y.append(sales_scaled[i+SEQ_LEN])

    X, y = np.array(X), np.array(y)

    safe_name = product.replace(" ", "_")

    # ---- Train & Save LSTM ----
    # lstm = build_lstm()
    # lstm.fit(X, y, epochs=30, batch_size=8, verbose=0)
    # lstm.save(f"{model_dir}/{safe_name}_lstm.h5")

    # # ---- Train & Save GRU ----
    # gru = build_gru()
    # gru.fit(X, y, epochs=30, batch_size=8, verbose=0)
    # gru.save(f"{model_dir}/{safe_name}_gru.h5")

    # # ---- Train & Save CNN ----
    # cnn = build_cnn()
    # cnn.fit(X, y, epochs=30, batch_size=8, verbose=0)
    # cnn.save(f"{model_dir}/{safe_name}_cnn.h5")

     # ---- Train & Save RNN ----
    rnn = build_rnn()
    rnn.fit(X, y, epochs=30, batch_size=8, verbose=0)
    rnn.save(f"{model_dir}/{safe_name}_rnn.h5")

    # ---- Train & Save Transformer ----
    # transformer = build_transformer()
    # transformer.fit(X, y, epochs=30, batch_size=8, verbose=0)
    # transformer.save(f"{model_dir}/{safe_name}_transformer.h5")

    # Save scaler
    np.save(f"{model_dir}/{safe_name}_scaler.npy", scaler.data_min_)

print("\n✔ All models trained and saved successfully!")
