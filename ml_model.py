# --- STOCK MARKET ANALYSIS PIPELINE WITH CONSTANT PREDICTION DATE ---

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import shap
import joblib
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from indicators import add_indicators  # your custom indicators module

# --- Create folders ---
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("powerbi", exist_ok=True)

# --- Stock list ---
STOCK_CATEGORIES = {
    'Nifty 50': {
        'Adani Enterprises': 'ADANIENT.NS',
        'Adani Ports & SEZ': 'ADANIPORTS.NS',
        'Apollo Hospitals': 'APOLLOHOSP.NS',
        'Asian Paints': 'ASIANPAINT.NS',
        'Axis Bank': 'AXISBANK.NS',
        'Bajaj Auto': 'BAJAJ-AUTO.NS',
        'Bajaj Finance': 'BAJFINANCE.NS',
        'Bajaj Finserv': 'BAJAJFINSV.NS',
        'Bharat Electronics': 'BEL.NS',
        'Bharti Airtel': 'BHARTIARTL.NS',
        'Cipla': 'CIPLA.NS',
        'Coal India': 'COALINDIA.NS',
        "Dr. Reddy's Laboratories": 'DRREDDY.NS',
        'Eicher Motors': 'EICHERMOT.NS',
        'Grasim Industries': 'GRASIM.NS',
        'HCLTech': 'HCLTECH.NS',
        'HDFC Bank': 'HDFCBANK.NS',
        'HDFC Life': 'HDFCLIFE.NS',
        'Hero MotoCorp': 'HEROMOTOCO.NS',
        'Hindalco Industries': 'HINDALCO.NS',
        'Hindustan Unilever': 'HINDUNILVR.NS',
        'ICICI Bank': 'ICICIBANK.NS',
        'IndusInd Bank': 'INDUSINDBK.NS',
        'Infosys': 'INFY.NS',
        'ITC': 'ITC.NS',
        'Jio Financial Services': 'JIOFIN.NS',
        'JSW Steel': 'JSWSTEEL.NS',
        'Kotak Mahindra Bank': 'KOTAKBANK.NS',
        'Larsen & Toubro': 'LT.NS',
        'Mahindra & Mahindra': 'M&M.NS',
        'Maruti Suzuki': 'MARUTI.NS',
        'Nestlé India': 'NESTLEIND.NS',
        'NTPC': 'NTPC.NS',
        'Oil and Natural Gas Corporation': 'ONGC.NS',
        'Power Grid': 'POWERGRID.NS',
        'Reliance Industries': 'RELIANCE.NS',
        'SBI Life Insurance Company': 'SBILIFE.NS',
        'Shriram Finance': 'SHRIRAMFIN.NS',
        'State Bank of India': 'SBIN.NS',
        'Sun Pharma': 'SUNPHARMA.NS',
        'Tata Consultancy Services': 'TCS.NS',
        'Tata Consumer Products': 'TATACONSUM.NS',
        'Tata Motors': 'TATAMOTORS.NS',
        'Tata Steel': 'TATASTEEL.NS',
        'Tech Mahindra': 'TECHM.NS',
        'Titan Company': 'TITAN.NS',
        'Trent': 'TRENT.NS',
        'UltraTech Cement': 'ULTRACEMCO.NS',
        'Wipro': 'WIPRO.NS'
    }
}

# --- SET CONSTANT PREDICTION DATE ---
# Eg: prediction for tomorrow
prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
print(f"🔗 Prediction Date set to: {prediction_date}")

all_predictions = []

# --- Loop for each stock ---
for name, symbol in STOCK_CATEGORIES['Nifty 50'].items():
    print(f"\n--- Processing: {name} ({symbol}) ---")
    df = yf.download(symbol, start="2018-01-01", end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=True)
    if df.empty:
        print(f"⚠ No data for {symbol}")
        continue

    df = add_indicators(df)

    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    df['Close_lag_3'] = df['Close'].shift(3)

    df['Target'] = np.where((df['Close'].shift(-1) - df['Close']) / df['Close'] > 0.01, 1,
                     np.where((df['Close'].shift(-1) - df['Close']) / df['Close'] < -0.01, 0, np.nan))
    df.dropna(inplace=True)

    features = [
        'RSI', 'SMA_20', 'SMA_50', 'EMA_20', 'MACD', 'MACD_Signal',
        'Stoch_K', 'Stoch_D', 'Volume_SMA_20', 'Daily_Return', 'Momentum',
        'BB_Upper', 'BB_Lower', 'ATR', 'Close_lag_1', 'Close_lag_2', 'Close_lag_3'
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df['Target']
    if X.empty or y.empty:
        print(f"⚠ Skipping {symbol} due to insufficient data.")
        continue

    # Drop highly correlated
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_list = [col for col in upper.columns if any(upper[col] > 0.9)]
    X.drop(columns=drop_list, inplace=True, errors='ignore')

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Train model
    tscv = TimeSeriesSplit(n_splits=5)
    xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
    grid = GridSearchCV(xgb, {
        'n_estimators': [300],
        'max_depth': [4],
        'learning_rate': [0.05],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'gamma': [0],
        'reg_alpha': [0],
        'reg_lambda': [2]
    }, cv=tscv, scoring='accuracy', n_jobs=-1)

    grid.fit(X_scaled, y)
    best = grid.best_estimator_
    joblib.dump(best, f"models/{symbol}_model.joblib")

    # Make latest prediction
    latest = df.iloc[-1][X.columns]
    latest_scaled = scaler.transform(latest.to_frame().T)
    prob = best.predict_proba(latest_scaled)[0]
    pred = best.predict(latest_scaled)[0]
    bp, bap = prob[1]*100, prob[0]*100
    advice = ("STRONG BUY" if pred==1 and bp>70 else
              "BUY" if pred==1 else
              "STRONG SELL" if pred==0 and bap>70 else
              "SELL" if pred==0 else "HOLD")

    # --- Append with CONSTANT DATE ---
    all_predictions.append({
        'Date': prediction_date,  #  FIXED DATE FOR ALL ROWS!
        'Company': name,
        'Symbol': symbol,
        'Prediction': 'Bullish' if pred==1 else 'Bearish',
        'Bullish_Confidence': round(bp,2),
        'Bearish_Confidence': round(bap,2),
        'Advice': advice
    })

# --- Save final predictions ---
df_result = pd.DataFrame(all_predictions)
df_result.to_csv("powerbi/predictions_today.csv", index=False)
print("\n✅ Saved all predictions to powerbi/predictions_today.csv")