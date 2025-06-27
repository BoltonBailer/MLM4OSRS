import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tabulate import tabulate
import json
import os
from PIL import Image, ImageTk
import pygame


BG_COLOR = "#ffffff"
TEXT_COLOR = "#333333"
BTN_COLOR = "#9c8fc9"
SECONDARY = "#bdb2ff"
FONT_MAIN = ("Segoe UI", 10)
FONT_HEADER = ("Segoe UI", 12, "bold")

#api stuff
API_URL = "https://prices.runescape.wiki/api/v1/osrs"
HEADERS = {"User-Agent": "OSRS-Predictor-Demo"}

#fetch items
def fetch_item_id(item_name):
    response = requests.get(f"{API_URL}/mapping", headers=HEADERS)
    items = response.json()
    for item in items:
        if item['name'].lower() == item_name.lower():
            return item['id']
    return None
#timed data
def fetch_time_series(item_id):
    url = f"{API_URL}/timeseries?timestep=1h&id={item_id}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return None
    data = response.json().get('data', [])
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['time_num'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
 
    df = df.dropna(subset=['avgHighPrice'])
    return df
#model
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
#top items
def fetch_top_items():
    prices_url = f"{API_URL}/latest"
    volume_url = f"{API_URL}/volumes"
    map_url = f"{API_URL}/mapping"

    prices_data = requests.get(prices_url, headers=HEADERS).json()["data"]
    volumes_data = requests.get(volume_url, headers=HEADERS).json()["data"]
    mapping_data = requests.get(map_url, headers=HEADERS).json()

    item_list = []
    for item in mapping_data:
        item_id = str(item["id"])
        if item_id in prices_data and item_id in volumes_data:
            low = prices_data[item_id].get("low", 0)
            high = prices_data[item_id].get("high", 0)
            margin = high - low

            volume = volumes_data.get(item_id, 0)  # just an int now!
            profit = margin * volume

            if margin > 0 and volume > 0:
                item_list.append({
                    "Name": item["name"],
                    "Buy": low,
                    "Sell": high,
                    "Margin": margin,
                    "Volume": volume,
                    "Profit": profit
                })

    df = pd.DataFrame(item_list)
    df = df.sort_values(by="Profit", ascending=False).head(15)

    # Format large numbers with commas
    df["Buy"] = df["Buy"].map("{:,}".format)
    df["Sell"] = df["Sell"].map("{:,}".format)
    df["Margin"] = df["Margin"].map("{:,}".format)
    df["Volume"] = df["Volume"].map("{:,}".format)
    df["Profit"] = df["Profit"].map("{:,}".format)

    return df



#steam deck
class OSRSDeckApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.title("OSRS Deck")
        self.geometry("900x700")
        self.configure(bg=BG_COLOR)
        self.all_item_names = []
        self.fetch_item_names()
        #savefiles
        self.fav_file = "favorites.json"
        self.trade_file = "trades.json"
        self.favorite_items = self.load_favorites()
        self.trade_history = self.load_trades()
        #webhook stuff
        self.webhook_settings_file = "webhook_settings.json"
        self.webhook_url, self.alerts_on = self.load_webhook_settings()
        #json config
        with open("config.json", "r") as f:
            self.config = json.load(f)

        #background pic
        bg_image = Image.open(self.config["background_image"])
        bg_image = bg_image.resize((900, 700), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        self.main_frame = tk.Frame(self, width=900, height=700)
        self.main_frame.pack(fill="both", expand=True)
        self.bg_label = tk.Label(self.main_frame, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.content_frame = tk.Frame(self, bg=BG_COLOR)

        pygame.mixer.init()
        pygame.mixer.music.load(self.config["background_music"])
        pygame.mixer.music.play()


        #bk_music.mp3


        buttons = [
            ("Market predict", self.market_predict),
            ("Top Grossing", self.top_gross_list),
            ("Favorites", self.favorites),
            ("Trade Diary", self.diary_trades),
            ("Discord hook", self.disc_hook),
            ("Settings", self.settings_tile),
        ]

        grid_wrapper = tk.Frame(self.main_frame, bg="", highlightthickness=0)
        grid_wrapper.place(relx=0.5, rely=0.5, anchor="center")


        for i, (label, command) in enumerate(buttons):
            wrapper = tk.Frame(grid_wrapper, bg="#d0cbe9", bd=0)
            wrapper.grid(row=i // 3, column=i % 3, padx=15, pady=15)

            btn = tk.Button(
                wrapper,
                text=label,
                width=18,
                height=3,
                bg=BTN_COLOR,
                fg="white",
                font=FONT_MAIN,
                bd=0,
                relief="flat",
                highlightthickness=0,
                activebackground="#a99bde",
                cursor="hand2",
                command=command
            )
            btn.pack(padx=2, pady=2)


    def add_favorite_item(self):
        item = self.fav_entry.get().strip()
        if not item:
            return

        item_id = fetch_item_id(item)
        if not item_id:
            messagebox.showerror("Item Not Found", f"'{item}' is not a valid OSRS item.")
            return

        if item not in self.favorite_items:
            self.favorite_items.append(item)
            self.save_favorites()
            self.render_favorite_items()
        self.fav_entry.delete(0, tk.END)

    def fetch_item_names(self):
        try:
            response = requests.get(f"{API_URL}/mapping", headers=HEADERS)
            self.all_item_names = sorted([item["name"] for item in response.json()])
        except:
            self.all_item_names = []

    def render_favorite_items(self):
        for widget in self.fav_list_frame.winfo_children():
            widget.destroy()

        for item in self.favorite_items:
            container = tk.Frame(self.fav_list_frame, bg=BG_COLOR)
            container.pack(fill="x", padx=5, pady=3)

            name_btn = tk.Button(
                container,
                text=item,
                font=FONT_MAIN,
                bg=SECONDARY,
                fg="white",
                relief="groove",
                command=lambda i=item: self.launch_prediction_from_fav(i)
            )
            name_btn.pack(side="left", fill="x", expand=True)

            remove_btn = tk.Button(
                container,
                text="❌",
                font=("Segoe UI", 10, "bold"),
                bg="#ff6b6b",
                fg="white",
                relief="flat",
                width=3,
                command=lambda i=item: self.remove_favorite_item(i)
            )
            remove_btn.pack(side="right", padx=4)

    
    def remove_favorite_item(self, item):
        if item in self.favorite_items:
            self.favorite_items.remove(item)
            self.save_favorites()
            self.render_favorite_items()
                
    def launch_prediction_from_fav(self, item_name):
            self.market_predict()
            self.item_entry.delete(0, tk.END)
            self.item_entry.insert(0, item_name)
            self.predict_price()

    def show_main_menu(self):
        self.content_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)
    #bttn 1
    def top_gross_list(self):
        df = fetch_top_items()

        self.main_frame.pack_forget()
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.content_frame.pack(fill="both", expand=True)

        label = tk.Label(self.content_frame, text="Top 15 Items", font=FONT_HEADER, bg=BG_COLOR)
        label.pack(pady=10)

        back_btn = tk.Button(
            self.content_frame, text="← Back", bg=SECONDARY, fg="white", font=FONT_MAIN,
            command=self.show_main_menu
        )
        back_btn.pack(pady=5)

        text = tk.Text(self.content_frame, wrap="none", font=("Courier", 9), bg="white", fg="black")
        text.pack(expand=True, fill="both", padx=10, pady=10)
        pretty_table = tabulate(df, headers="keys", tablefmt="plain", showindex=False)
        text.insert("end", pretty_table)
    #button 2
    def market_predict(self):

        self.main_frame.pack_forget()
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.content_frame.pack(fill="both", expand=True)

        tk.Label(self.content_frame, text="Item Price Predictor", font=FONT_HEADER, bg=BG_COLOR).pack(pady=10)

        self.item_entry_var = tk.StringVar()
        self.item_entry = ttk.Entry(self.content_frame, width=30, font=FONT_MAIN, textvariable=self.item_entry_var)
        self.item_entry.pack()
        self.item_entry.bind("<KeyRelease>", self.update_suggestions)

        self.suggestion_box = tk.Listbox(self.content_frame, width=30, height=5, font=FONT_MAIN)
        self.suggestion_box.pack()
        self.suggestion_box.bind("<<ListboxSelect>>", self.select_suggestion)

        predict_btn = tk.Button(
            self.content_frame,
            text="Predict",
            bg=BTN_COLOR,
            fg="white",
            font=FONT_MAIN,
            command=self.predict_price
        )
        predict_btn.pack(pady=5)

        back_btn = tk.Button(
            self.content_frame, text="← Back", bg=SECONDARY, fg="white", font=FONT_MAIN,
            command=self.show_main_menu
        )
        back_btn.pack()
        self.discord_btn = tk.Button(
            self.content_frame,
            text="Send to Discord",
            bg=BTN_COLOR,
            fg="white",
            font=FONT_MAIN,
            command=self.send_to_discord
        )
        self.discord_btn.pack(pady=5)
        
        self.output_text = tk.Text(self.content_frame, height=10, width=80, bg="white", fg="black", font=FONT_MAIN)
        self.output_text.pack(pady=10)

        self.canvas_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        self.canvas_frame.pack(fill="both", expand=True)

    def predict_price(self):
        item_name = self.item_entry.get().strip()
        item_id = fetch_item_id(item_name)

        if not item_id:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "❌ Item not found.\n")
            return

        df = fetch_time_series(item_id)
        if df is None or df.empty:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "❌ Price data unavailable.\n")
            return

        df, model, x_scaler, y_scaler, mae = run_regression(df)

        future_forecasts = {}
        for h in range(1, 4):
            future_time = df['time_num'].max() + 3600 * h
            scaled_time = x_scaler.transform([[future_time]])
            scaled_prediction = model.predict(scaled_time)
            predicted_price = y_scaler.inverse_transform(scaled_prediction)[0][0]
            future_forecasts[h] = int(predicted_price)

        try:
            live_url = f"{API_URL}/latest"
            live_response = requests.get(live_url, headers=HEADERS)
            live_data = live_response.json().get("data", {}).get(str(item_id), {})
            buy_price = live_data.get("high")
            sell_price = live_data.get("low")
        except:
            buy_price = sell_price = None

        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Model MAE: ~{int(mae):,} gp\n")
        self.output_text.insert(tk.END, f"Next hour price guess: {future_forecasts[1]:,} gp\n\n")

        if buy_price and sell_price:
            est_profit = future_forecasts[1] - buy_price
            self.output_text.insert(tk.END, f"Buy now: {buy_price:,} gp\n")
            self.output_text.insert(tk.END, f"Sell later: {future_forecasts[1]:,} gp\n")
            self.output_text.insert(tk.END, f"Est. profit: {est_profit:,} gp\n")

            if est_profit > 100:
                self.output_text.insert(tk.END, "\nThis flip looks juicy!\n")
            else:
                self.output_text.insert(tk.END, "\nNot the best margins.\n")
        else:
            self.output_text.insert(tk.END, "\n⚠️ Couldn’t fetch live buy/sell data.\n")

        self.output_text.insert(tk.END, "\nShort-term forecast:\n")
        for hour, price in future_forecasts.items():
            self.output_text.insert(tk.END, f"⏱ In {hour}h → ~{price} gp\n")

        self.plot_data(df, item_name)

        self.last_prediction_data = {
            "item": item_name,
            "buy": buy_price,
            "sell": future_forecasts[1],
            "profit": est_profit if buy_price else None
        }

    def plot_data(self, df, item_name):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df['timestamp'], df['avgHighPrice'], label='Actual', linewidth=1.5)
        ax.plot(df['timestamp'], df['predicted'], label='Predicted', linestyle='--', linewidth=1.5)
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
    #bttn 3
    def diary_trades(self):
        self.main_frame.pack_forget()
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.content_frame.pack(fill="both", expand=True)

        self.trade_history = getattr(self, "trade_history", [])

        label = tk.Label(self.content_frame, text="Recent Trades", font=FONT_HEADER, bg=BG_COLOR)
        label.pack(pady=10)

        # Input Fields
        entry_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        entry_frame.pack(pady=5)

        tk.Label(entry_frame, text="Item Name:", font=FONT_MAIN, bg=BG_COLOR).grid(row=0, column=0)
        item_entry = tk.Entry(entry_frame, font=FONT_MAIN)
        item_entry.grid(row=0, column=1)

        tk.Label(entry_frame, text="Buy Price:", font=FONT_MAIN, bg=BG_COLOR).grid(row=1, column=0)
        buy_entry = tk.Entry(entry_frame, font=FONT_MAIN)
        buy_entry.grid(row=1, column=1)

        tk.Label(entry_frame, text="Sell Price:", font=FONT_MAIN, bg=BG_COLOR).grid(row=2, column=0)
        sell_entry = tk.Entry(entry_frame, font=FONT_MAIN)
        sell_entry.grid(row=2, column=1)

        def add_trade():
            item = item_entry.get()
            try:
                buy = int(buy_entry.get())
                sell = int(sell_entry.get())
                profit = sell - buy
                self.trade_history.append((item, buy, sell, profit))
                self.save_trades()
                update_output()
                item_entry.delete(0, tk.END)
                buy_entry.delete(0, tk.END)
                sell_entry.delete(0, tk.END)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for buy/sell.")

        submit_btn = tk.Button(entry_frame, text="Add Trade", font=FONT_MAIN, bg=BTN_COLOR, command=add_trade)
        submit_btn.grid(row=3, columnspan=2, pady=5)

      
        output = tk.Text(self.content_frame, height=15, width=70, font=FONT_MAIN, bg="white", fg="black")
        output.pack(pady=10, padx=10)
        def update_output():
            output.delete(1.0, tk.END)
            total_profit = sum(trade[3] for trade in self.trade_history)
            for item, buy, sell, profit in self.trade_history:
                output.insert(tk.END, f"{item} | Bought for {buy:,} | Sold for {sell:,} | Profit: {profit:,} gp\n")
            output.insert(tk.END, f"\nTotal Profit: {total_profit:,} gp")
        
        update_output()

        back_btn = tk.Button(
            self.content_frame, text="← Back", bg=SECONDARY, fg="white", font=FONT_MAIN,
            command=self.show_main_menu
        )
        back_btn.pack(pady=5)
    #bttn 4
    def favorites(self):
            self.main_frame.pack_forget()
            for widget in self.content_frame.winfo_children():
                widget.destroy()
            self.content_frame.pack(fill="both", expand=True)

            label = tk.Label(self.content_frame, text="Favorites", font=FONT_HEADER, bg=BG_COLOR)
            label.pack(pady=10)

            back_btn = tk.Button(
                self.content_frame, text="← Back", bg=SECONDARY, fg="white", font=FONT_MAIN,
                command=self.show_main_menu
            )
            back_btn.pack(pady=5)

            #box
            tk.Label(self.content_frame, text="Add Item to Favorites:", font=FONT_MAIN, bg=BG_COLOR).pack(pady=(10, 5))
            self.fav_entry = ttk.Entry(self.content_frame, width=30)
            self.fav_entry.pack()
            self.fav_entry.bind("<Return>", lambda event: self.add_favorite_item())


            add_btn = tk.Button(
                self.content_frame,
                text="Add",
                bg=BTN_COLOR,
                fg="white",
                font=FONT_MAIN,
                command=self.add_favorite_item
            )
            add_btn.pack(pady=5)

            
            self.fav_list_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
            self.fav_list_frame.pack(pady=10)

            #keep items in
            self.render_favorite_items() 
    #bttn 5
    def settings_tile(self):
        self.main_frame.pack_forget()
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.content_frame.pack(fill="both", expand=True)

        tk.Label(self.content_frame, text="Settings", font=FONT_HEADER, bg=BG_COLOR).pack(pady=10)

        back_btn = tk.Button(
            self.content_frame, text="← Back", bg=SECONDARY, fg="white", font=FONT_MAIN,
            command=self.show_main_menu
        )
        back_btn.pack(pady=5)

        # Music toggle state
        self.music_playing = pygame.mixer.music.get_busy()

        def toggle_music():
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                toggle_btn.config(text="▶️ Turn Music On")
            else:
                pygame.mixer.music.unpause()
                toggle_btn.config(text="🔇 Turn Music Off")

        toggle_btn = tk.Button(
            self.content_frame,
            text="🔇 Turn Music Off" if self.music_playing else "▶️ Turn Music On",
            bg=BTN_COLOR,
            fg="white",
            font=FONT_MAIN,
            command=toggle_music
        )
        toggle_btn.pack(pady=20)

    #bttn 6
    def disc_hook(self):

        self.main_frame.pack_forget()
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.content_frame.pack(fill="both", expand=True)

        label = tk.Label(self.content_frame, text="Price Alerts", font=FONT_HEADER, bg=BG_COLOR)
        label.pack(pady=10)

        back_btn = tk.Button(
            self.content_frame, text="← Back", bg=SECONDARY, fg="white", font=FONT_MAIN,
            command=self.show_main_menu
        )
        back_btn.pack(pady=5)

        # webhook bar
        tk.Label(self.content_frame, text="Enter Discord Webhook URL:", bg=BG_COLOR, font=FONT_MAIN).pack(pady=(20, 5))
        self.webhook_var = tk.StringVar(value=self.webhook_url)
        self.webhook_entry = ttk.Entry(self.content_frame, textvariable=self.webhook_var, width=60)
        self.webhook_entry.pack()

        # toggle
        self.alerts_enabled = tk.BooleanVar(value=self.alerts_on)

        def toggle_alerts():
            state = self.alerts_enabled.get()
            if state:
                status_label.config(text=" Alerts Enabled", fg="green")
            else:
                status_label.config(text=" Alerts Disabled", fg="red")
        toggle_btn = tk.Checkbutton(
            self.content_frame,
            text="Enable Alerts",
            variable=self.alerts_enabled,
            onvalue=True,
            offvalue=False,
            font=FONT_MAIN,
            bg=BG_COLOR,
            command=toggle_alerts
        )
        toggle_btn.pack(pady=10)

        status_label = tk.Label(self.content_frame, text=" Alerts Disabled", bg=BG_COLOR, fg="red", font=FONT_MAIN)
        status_label.pack()

        def save_alert_settings():
            url = self.webhook_var.get()
            enabled = self.alerts_enabled.get()
            self.save_webhook_settings(url, enabled)
            self.webhook_url = url
            self.alerts_on = enabled
            print(f"Saved Webhook: {url}")
            print(f"Alerts {'ON' if enabled else 'OFF'}")

        save_btn = tk.Button(
            self.content_frame,
            text="Save Settings",
            bg=BTN_COLOR,
            fg="white",
            font=FONT_MAIN,
            command=save_alert_settings
        )
        save_btn.pack(pady=15)

    def update_suggestions(self, event):
        typed = self.item_entry_var.get().lower()
        matches = [name for name in self.all_item_names if typed in name.lower()][:10]

        self.suggestion_box.delete(0, tk.END)
        for name in matches:
            self.suggestion_box.insert(tk.END, name)

    def select_suggestion(self, event):
        selected = self.suggestion_box.get(tk.ACTIVE)
        self.item_entry_var.set(selected)
        self.suggestion_box.delete(0, tk.END)

    def load_favorites(self):
        if os.path.exists(self.fav_file):
            with open(self.fav_file, "r") as f:
                return json.load(f)
        return []

    def save_favorites(self):
        with open(self.fav_file, "w") as f:
            json.dump(self.favorite_items, f)

    def load_trades(self):
        if os.path.exists(self.trade_file):
            with open(self.trade_file, "r") as f:
                return json.load(f)
        return []

    def save_trades(self):
        with open(self.trade_file, "w") as f:
            json.dump(self.trade_history, f)

    def load_webhook_settings(self):
        if os.path.exists(self.webhook_settings_file):
            with open(self.webhook_settings_file, "r") as f:
                data = json.load(f)
                return data.get("url", ""), data.get("enabled", False)
        return "", False

    def save_webhook_settings(self, url, enabled):
        with open(self.webhook_settings_file, "w") as f:
            json.dump({"url": url, "enabled": enabled}, f)

    def send_to_discord(self):
        if not hasattr(self, "last_prediction_data"):
            messagebox.showwarning("No Prediction", "Please run a prediction first.")
            return

        if not self.alerts_on or not self.webhook_url:
            messagebox.showwarning("Alerts Disabled", "Enable alerts and set webhook URL.")
            return

        data = self.last_prediction_data
        message = {
            "content": f"📊 `{data['item']}` Prediction:\nBuy: {data['buy']:,} gp\nSell: {data['sell']:,} gp\nProfit: {data['profit']:,} gp"
        }

        try:
            response = requests.post(self.webhook_url, json=message)
            if response.status_code == 204:
                messagebox.showinfo("Success", "Prediction sent to Discord.")
            else:
                messagebox.showerror("Error", f"Failed to send (HTTP {response.status_code})")
        except Exception as e:
            messagebox.showerror("Exception", f"Something went wrong:\n{e}")

    


if __name__ == "__main__":
    app = OSRSDeckApp()
    app.mainloop()