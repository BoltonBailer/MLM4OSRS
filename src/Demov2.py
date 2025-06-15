import pandas as pd
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

API_URL = "https://prices.runescape.wiki/api/v1/osrs"
HEADERS = {"User-Agent": "OSRS-Predictor-Demo"}

# item fetch
def fetch_item_id(item_name):
    response = requests.get(f"{API_URL}/mapping", headers=HEADERS)
    items = response.json()
    for item in items:
        if item['name'].lower() == item_name.lower():
            return item['id']
    return None

# timed data
def fetch_time_series(item_id):
    url = f"{API_URL}/timeseries?timestep=1h&id={item_id}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return None
    data = response.json().get('data', [])
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['time_num'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    return df

# ternsor regression model
def run_regression(df):
    X = df[['time_num']].values
    y = df[['avgHighPrice']].values

    # scale the x
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # karas stuff
    model = Sequential([
        Input(shape=(1,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])


    model.fit(X_train, y_train, epochs=20, verbose=0)

    # Predict for all timestamps
    df['scaled_time'] = scaler.transform(X)
    df['predicted'] = model.predict(df['scaled_time']).flatten()

    # val
    y_pred_test = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred_test)

    return df, model, scaler, mae

# graph stuff
def plot_prediction(df, item_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['avgHighPrice'], label='Actual Price')
    plt.plot(df['timestamp'], df['predicted'], label='Predicted Price', linestyle='--')
    plt.title(f"{item_name} Price Prediction (Keras)")
    plt.xlabel("Time")
    plt.ylabel("High Price (GP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    item_name = input("Type item name (case sensitive): ")
    item_id = fetch_item_id(item_name)

    if not item_id:
        print("Item not found.")
        exit()

    df = fetch_time_series(item_id)
    if df is None or df.empty:
        print("Price data unavailable.")
        exit()

    df, model, scaler, mae = run_regression(df)
    print(f"Prediction MAE: {int(mae)} GP")

    # Predict future price
    future_time = df['time_num'].max() + 3600  # Next hour
    future_time_scaled = scaler.transform([[future_time]])
    future_price = model.predict(future_time_scaled)[0][0]
    print(f"Predicted next-hour price: {int(future_price)} GP")

    plot_prediction(df, item_name)
