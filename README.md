# OSRS Predictor – Final Product Features

The OSRS Predictor is a machine learning powered application designed to forecast Grand Exchange item prices, assess flipping risk, and help Old School RuneScape players maximize profits.

## Core Features

### Item Price Prediction
• Predict short term (1–3 day) price changes for GE items using trained regression models  
• Models include options like Linear Regression, Random Forest, and LSTM (TBD during development)

### Flipping Risk Assessment
• Categorize item volatility into Low, Medium, or High based on price trends and model output variance

### ROI Calculator
• Calculate estimated return on investment after GE tax using price predictions and confidence scores

## Data and Visualization

### Time Series Graphs
• Interactive plots showing historical high/low prices and daily trade volume  
• Overlay predicted trendlines for visual comparison

### Volume Spike Detection
• Automatically flag abnormal volume activity for potential flipping opportunities

### Model Confidence Display
• Show visual indicator of the model’s confidence in each prediction using probability or score

## User Interaction

### Item Search Tool
• Search and load predictions for any tradeable item on the Grand Exchange

### Filter and Sort Options
• Filter items by F2P/P2P status, price range, margin, and trade volume  
• Sort results by highest ROI, lowest risk, or predicted price increase

### Custom Alerts
• Set custom alerts for margin thresholds, ROI, or volume spikes  
• Support optional delivery via in app message, email, or Discord webhook

## Real Time Data Integration

### OSRS Wiki API Integration
• Pull and cache real time and historical price data from the OSRS Wiki Exchange API  
• Refresh and sync item data automatically to stay up to date

## Planned Stack

• Language: Python 3.10+  
• Libraries: scikit learn, pandas, matplotlib/seaborn, requests  
• UI Framework: tkinter or customtkinter  
• Deployment: Desktop application (Windows/Mac/Linux)
