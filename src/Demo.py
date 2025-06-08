import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

API_URL = "https://prices.runescape.wiki/api/v1/osrs"
HEADERS = {"User-Agent": "OSRS-Predictor-Demo"}

#TODO grab the data rom API
def fetch_item_id(item_name):
    response = requests.get(f"{API_URL}/mapping", headers=HEADERS)

    items = response.json()
    #look for the item
    for item in items:
        if item['name'].lower() == item_name.lower():
            return item['id']
    #if not found return none
    return None

#TODO then I gotta run the ID through  to gets its hourly data
def fetch_time_series(item_id):
    url = f"{API_URL}/timeseries?timestep=1h&id={item_id}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None
    data = response.json().get('data', [])

    #put into df
    df = pd.DataFrame(data)

    #create columns for into
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['time_num'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    return df

#use linear regressions to rpedict price
def run_regression(df):
    #make model
    model = LinearRegression()
    #X and Y
    X = df[['time_num']]
    y = df[['avgHighPrice']]
    #fit model
    model.fit(X, y)

    df['predicted'] = model.predict(X)
    #test MAE
    mae = mean_absolute_error(y, df['predicted'])
    return df, model, mae


    #graph stuff ----------------
def plot_prediction(df, item_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['avgHighPrice'], label='Actual Price')
    plt.plot(df['timestamp'], df['predicted'], label='Predicted Price', linestyle='--')
    plt.title(f"{item_name} Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("High Price (GP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    item_name = input("Type item Name(case sensative): ")
    item_id = fetch_item_id(item_name)

    if not item_id:
        print("Item not found.")
        exit()

    df = fetch_time_series(item_id)
    if df is None or df.empty:
        print("Price data unavailable.")
        exit()

    df, model, mae = run_regression(df)
    print(f"Prediction MAE: {int(mae)} GP")
    future_time = df['time_num'].max() + 3600
    future_price = model.predict([[future_time]])[0][0]
    print(f"Predicted next-hour price: {int(future_price)} GP")

    plot_prediction(df, item_name)
