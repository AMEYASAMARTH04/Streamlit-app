from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
from datetime import datetime

app = Flask(__name__)

DEFAULT_MODEL_DIR = "models"

STOCK_CATEGORIES = {
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
    'Wipro': 'WIPRO.NS',
}

def add_indicators(df):
    try:
        import ta
        close  = pd.Series(df['Close'].values.flatten(),  index=df.index)
        high   = pd.Series(df['High'].values.flatten(),   index=df.index)
        low    = pd.Series(df['Low'].values.flatten(),    index=df.index)
        volume = pd.Series(df['Volume'].values.flatten(), index=df.index)
        df['RSI']           = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        df['Stoch_K']       = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14).stoch()
        df['Stoch_D']       = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14).stoch_signal()
        df['Momentum']      = close.pct_change(periods=10)
        df['SMA_20']        = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        df['SMA_50']        = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        df['EMA_20']        = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df['MACD']          = macd.macd()
        df['MACD_Signal']   = macd.macd_signal()
        df['ATR']           = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df['BB_Upper']      = bb.bollinger_hband()
        df['BB_Lower']      = bb.bollinger_lband()
        df['Volume_SMA_20'] = volume.rolling(window=20).mean()
        df['Daily_Return']  = close.pct_change()
        df.dropna(inplace=True)
    except Exception as e:
        print(f"Indicator error: {e}")
    return df

def predict_single_stock(symbol, model_dir):
    model_path = os.path.join(model_dir, f"{symbol}_model.joblib")
    if not os.path.exists(model_path):
        return None, "Model file not found"
    df = yf.download(symbol, start="2022-01-01",
                     end=datetime.now().strftime('%Y-%m-%d'),
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None, "No data"
    df = add_indicators(df)
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    df['Close_lag_3'] = df['Close'].shift(3)
    df.dropna(inplace=True)
    features = ['RSI','SMA_20','SMA_50','EMA_20','MACD','MACD_Signal',
                'Stoch_K','Stoch_D','Volume_SMA_20','Daily_Return','Momentum',
                'BB_Upper','BB_Lower','ATR','Close_lag_1','Close_lag_2','Close_lag_3']
    features = [f for f in features if f in df.columns]
    try:
        from sklearn.preprocessing import StandardScaler
        model = joblib.load(model_path)
        X = df[features]
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_list = [col for col in upper.columns if any(upper[col] > 0.9)]
        X = X.drop(columns=drop_list, errors='ignore')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        latest_scaled = X_scaled[-1].reshape(1, -1)
        prob = model.predict_proba(latest_scaled)[0]
        pred = model.predict(latest_scaled)[0]
        bp, bap = prob[1]*100, prob[0]*100
        advice = ("STRONG BUY"  if pred==1 and bp>70  else
                  "BUY"         if pred==1             else
                  "STRONG SELL" if pred==0 and bap>70  else
                  "SELL"        if pred==0             else "HOLD")
        return {
            'pred': int(pred),
            'bullish_conf': round(bp, 2),
            'bearish_conf': round(bap, 2),
            'advice': advice,
        }, None
    except Exception as e:
        return None, str(e)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON body sent'}), 400
        company = data.get('company')
        if not company:
            return jsonify({'error': 'company field is required'}), 400
        symbol = STOCK_CATEGORIES.get(company)
        if not symbol:
            return jsonify({'error': f'Company "{company}" not found'}), 404
        result, err = predict_single_stock(symbol, DEFAULT_MODEL_DIR)
        if err:
            return jsonify({'error': err}), 500
        return jsonify({
            'company': company,
            'symbol': symbol,
            'advice': result['advice'],
            'bullish_percentage': result['bullish_conf'],
            'bearish_percentage': result['bearish_conf'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting Flask API on port {port}")
    app.run(host='0.0.0.0', port=port)