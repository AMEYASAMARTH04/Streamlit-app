# screener.py

import yfinance as yf
import pandas as pd
from indicators import add_indicators
from joblib import load
from stock_list import nifty_50
import datetime
import os
import sys

# Load trained model
MODEL_PATH = "stock_model.joblib"
FEATURES_PATH = "features.txt"

print("🔄 Initializing Stock Screener...")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}. Please train the model first.")
    sys.exit()

model = load(MODEL_PATH)

# Load training feature names
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
    print("✅ Loaded training features from features.txt")
else:
    print("⚠️ features.txt not found. Please retrain the model and save the features.")
    sys.exit()

# Initialize results list
predictions = []

# Loop through Nifty 50 stocks
for stock in nifty_50:
    print(f"\n📥 Downloading data for {stock}...")
    df = yf.download(
        stock,
        start='2022-01-01',
        end=datetime.datetime.today().strftime('%Y-%m-%d'),
        auto_adjust=True
    )

    if df.empty:
        print(f"⚠️ No data for {stock}. Skipping.")
        continue

    # Add indicators
    df = add_indicators(df)

    if df.empty or len(df) < 10:
        print(f"⚠️ Indicator calculation failed for {stock}. Skipping.")
        continue

    # Check if all required features are present
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing features for {stock}: {missing}. Skipping.")
        continue

    # Extract latest row of feature data
    X = df[feature_cols]
    X_latest = X.iloc[-1:]

    # Prediction
    try:
        y_pred = model.predict(X_latest)[0]
        y_proba = model.predict_proba(X_latest)[0][y_pred]
    except Exception as e:
        print(f"❌ Prediction error for {stock}: {e}")
        continue

    sentiment = "Bullish" if y_pred == 1 else "Bearish"
    confidence = round(y_proba * 100, 2)
    advice = "YES" if sentiment == "Bullish" and confidence >= 60 else "NO"

    predictions.append({
        'Stock': stock,
        'Sentiment': sentiment,
        'Confidence': confidence,
        'Advice': advice
    })

# Convert to DataFrame
if predictions:
    results_df = pd.DataFrame(predictions)
    results_df.sort_values(by='Confidence', ascending=False, inplace=True)
    results_df.to_csv("stock_screener_output.csv", index=False)

    print("\n📈 Final Screener Results:")
    print(results_df)
else:
    print("⚠️ No valid predictions were made. Check for errors in data or features.")


