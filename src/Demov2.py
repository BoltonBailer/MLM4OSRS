import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates

# --- API Setup ---
API_URL = "https://prices.runescape.wiki/api/v1/osrs"
HEADERS = {"User-Agent": "OSRS-Predictor-Demo"}

# Light pastel theme
BG_COLOR = "#f2f2f7"
TEXT_COLOR = "#333333"
PRIMARY = "#a0c4ff"
SECONDARY = "#bdb2ff"
ACCENT = "#ffc6ff"
FONT_MAIN = ("Segoe UI", 10)
FONT_HEADER = ("Segoe UI", 12, "bold")

def fetch_item_id(item_name):
    response = requests.get(f"{API_URL}/mapping", headers=HEADERS)
    items = response.json()
    for item in items:
        if item['name'].lower() == item_name.lower():
            return item['id']
    return None

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

def run_regression(df):
    X = df[['time_num']].values
    y = df[['avgHighPrice']].values

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = Sequential([
        Input(shape=(1,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, verbose=0)

    df['scaled_time'] = X_scaled
    predicted_scaled = model.predict(X_scaled)
    df['predicted'] = y_scaler.inverse_transform(predicted_scaled).flatten()

    y_pred_test = y_scaler.inverse_transform(model.predict(X_test))
    y_test_orig = y_scaler.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_orig, y_pred_test)

    return df, model, x_scaler, y_scaler, mae

class OSRSPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OSRS Price Predictor")
        self.geometry("850x650")
        self.configure(bg=BG_COLOR)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=FONT_HEADER)
        style.configure("TButton", background=PRIMARY, foreground=TEXT_COLOR, font=FONT_MAIN, padding=6)
        style.configure("TEntry", font=FONT_MAIN)

        self.label = ttk.Label(self, text="Enter Item Name:")
        self.label.pack(pady=10)

        self.item_entry = ttk.Entry(self, width=30, font=FONT_MAIN)
        self.item_entry.pack()

        self.predict_button = ttk.Button(self, text="Predict", command=self.predict_price)
        self.predict_button.pack(pady=10)

        self.output_text = tk.Text(self, height=10, width=80, bg="white", fg=TEXT_COLOR, font=FONT_MAIN, insertbackground=TEXT_COLOR)
        self.output_text.pack(pady=10)

        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def predict_price(self):
        item_name = self.item_entry.get().strip()
        item_id = fetch_item_id(item_name)

        if not item_id:
            messagebox.showerror("Error", "Item not found.")
            return

        df = fetch_time_series(item_id)
        if df is None or df.empty:
            messagebox.showerror("Error", "Price data unavailable.")
            return

        df, model, x_scaler, y_scaler, mae = run_regression(df)

        future_forecasts = {}
        for h in range(1, 4):
            future_time = df['time_num'].max() + 3600 * h
            scaled_time = x_scaler.transform([[future_time]])
            scaled_prediction = model.predict(scaled_time)
            predicted_price = y_scaler.inverse_transform(scaled_prediction)[0][0]
            future_forecasts[h] = int(predicted_price)

        live_url = "https://prices.runescape.wiki/api/v1/osrs/latest"
        try:
            live_response = requests.get(live_url, headers=HEADERS)
            live_data = live_response.json().get("data", {}).get(str(item_id), {})
            buy_price = live_data.get("high")
            sell_price = live_data.get("low")
        except:
            buy_price = sell_price = None

        self.output_text.delete(1.0, tk.END)

        self.output_text.insert(tk.END, f"model's average error: ~{int(mae)} gp\n")
        self.output_text.insert(tk.END, f"next hour price guess: {future_forecasts[1]} gp\n\n")

        if buy_price and sell_price:
            est_profit = future_forecasts[1] - buy_price

            self.output_text.insert(tk.END, f"buy now: {buy_price} gp\n")
            self.output_text.insert(tk.END, f"sell later: {future_forecasts[1]} gp\n")
            self.output_text.insert(tk.END, f"estimated profit: {est_profit} gp\n")

            if est_profit > 100:
                self.output_text.insert(tk.END, "yo this flip looks juicy 💸\n")
            else:
                self.output_text.insert(tk.END, "meh, not the best margins bro\n")
        else:
            self.output_text.insert(tk.END, "couldn’t get live data for buy/sell price\n")

        self.output_text.insert(tk.END, "\nshort-term forecast:\n")
        for hour, price in future_forecasts.items():
            self.output_text.insert(tk.END, f"in {hour}h → ~{price} gp\n")

        self.plot_data(df, item_name)


    def plot_data(self, df, item_name):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df['timestamp'], df['avgHighPrice'], label='Actual')
        ax.plot(df['timestamp'], df['predicted'], label='Predicted', linestyle='--')
        ax.set_title(f"{item_name} Price Prediction", color=TEXT_COLOR)
        ax.set_xlabel("Time", color=TEXT_COLOR)
        ax.set_ylabel("High Price", color=TEXT_COLOR)
        ax.tick_params(axis='x', colors=TEXT_COLOR)
        ax.tick_params(axis='y', colors=TEXT_COLOR)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        fig.autofmt_xdate(rotation=30)

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = OSRSPredictorApp()
    app.mainloop()
