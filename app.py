import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, jsonify, request
import threading

# ─────────────────────────────────────────────
# FLASK API (runs in background thread)
# ─────────────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        company = data.get('company')
        if not company:
            return jsonify({'error': 'company is required'}), 400

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

@flask_app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

def run_flask():
    flask_app.run(host='0.0.0.0', port=8000)

# Start Flask in background thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 ML Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1rem; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { color: #8b92a5; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .metric-value { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; }

    .strong-buy  { background: linear-gradient(135deg,#064e3b,#065f46); border:1px solid #059669; border-radius:8px; padding:4px 10px; color:#34d399; font-weight:700; font-size:.8rem; }
    .buy         { background: linear-gradient(135deg,#1e3a5f,#1e4976); border:1px solid #3b82f6; border-radius:8px; padding:4px 10px; color:#93c5fd; font-weight:700; font-size:.8rem; }
    .hold        { background: linear-gradient(135deg,#4a3920,#5a4520); border:1px solid #f59e0b; border-radius:8px; padding:4px 10px; color:#fcd34d; font-weight:700; font-size:.8rem; }
    .sell        { background: linear-gradient(135deg,#4a1515,#5a1a1a); border:1px solid #ef4444; border-radius:8px; padding:4px 10px; color:#fca5a5; font-weight:700; font-size:.8rem; }
    .strong-sell { background: linear-gradient(135deg,#450a0a,#500d0d); border:1px solid #dc2626; border-radius:8px; padding:4px 10px; color:#f87171; font-weight:700; font-size:.8rem; }

    .bullish-badge { color:#34d399; font-weight:700; }
    .bearish-badge { color:#f87171; font-weight:700; }

    .section-header {
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        margin: 1rem 0 .5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #c7d2fe;
    }
    div[data-testid="stSidebar"] { background-color: #111827; }
    .stSelectbox > div > div { background-color: #1f2937; }
    div.stButton > button {
        background: linear-gradient(135deg,#4f46e5,#7c3aed);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600; width: 100%;
    }
    div.stButton > button:hover { background: linear-gradient(135deg,#6366f1,#8b5cf6); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
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

DEFAULT_MODEL_DIR = "models"
PREDICTIONS_CSV = os.path.join("powerbi", "predictions_today.csv")


# ─────────────────────────────────────────────
# HELPER: add_indicators
# ─────────────────────────────────────────────
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
        st.error(f"Indicator error: {e}")
    return df


# ─────────────────────────────────────────────
# HELPER: load predictions CSV
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_saved_predictions(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# ─────────────────────────────────────────────
# HELPER: fetch stock data
# ─────────────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_stock_data(symbol, period="6mo"):
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# HELPER: run single stock prediction
# ─────────────────────────────────────────────
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
            'pred': int(pred), 'bullish_conf': round(bp,2),
            'bearish_conf': round(bap,2), 'advice': advice,
            'df': df
        }, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# HELPER: advice badge HTML
# ─────────────────────────────────────────────
def advice_badge(advice):
    mapping = {
        'STRONG BUY':  'strong-buy',
        'BUY':         'buy',
        'HOLD':        'hold',
        'SELL':        'sell',
        'STRONG SELL': 'strong-sell',
    }
    cls = mapping.get(advice, 'hold')
    return f'<span class="{cls}">{advice}</span>'


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_dir = st.text_input("Models Directory", value=DEFAULT_MODEL_DIR,
                              help="Folder containing *_model.joblib files")
    predictions_csv = st.text_input("Predictions CSV", value=PREDICTIONS_CSV,
                                    help="CSV from ml_model.py output")
    st.divider()
    st.markdown("### 📅 Date Range (Charts)")
    chart_period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
    st.divider()
    st.markdown("### 🔍 Screener Filters")
    filter_advice = st.multiselect(
        "Show Advice",
        ["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"],
        default=["STRONG BUY","BUY","SELL","STRONG SELL"]
    )
    min_conf = st.slider("Min Confidence (%)", 0, 100, 50)
    st.divider()
    st.caption("📊 Nifty 50 ML Stock Screener\nBuilt with XGBoost + Streamlit")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a1f35,#252a45);
            border:1px solid #3d4468; border-radius:16px;
            padding:1.5rem 2rem; margin-bottom:1.5rem;">
    <h1 style="margin:0;color:#e2e8f0;font-size:2rem;">
        📈 Nifty 50 ML Stock Screener
    </h1>
    <p style="margin:.3rem 0 0 0;color:#8b92a5;font-size:.95rem;">
        XGBoost-powered predictions for Nifty 50 stocks — Bullish / Bearish signals with confidence scores
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏠 Screener Dashboard", "🔍 Single Stock Analysis", "📊 Charts & Indicators"])


# ════════════════════════════════════════════
# TAB 1 – SCREENER DASHBOARD
# ════════════════════════════════════════════
with tab1:
    df_preds = load_saved_predictions(predictions_csv)

    if df_preds is not None and not df_preds.empty:
        # ── Summary Cards ──
        total   = len(df_preds)
        bullish = len(df_preds[df_preds['Prediction']=='Bullish'])
        bearish = len(df_preds[df_preds['Prediction']=='Bearish'])
        sb      = len(df_preds[df_preds['Advice']=='STRONG BUY'])
        ss      = len(df_preds[df_preds['Advice']=='STRONG SELL'])
        avg_conf = df_preds['Bullish_Confidence'].mean()

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        for col, label, val, clr in [
            (c1,"Total Stocks", total,    "#e2e8f0"),
            (c2,"Bullish 🟢",  bullish,   "#34d399"),
            (c3,"Bearish 🔴",  bearish,   "#f87171"),
            (c4,"Strong Buy",  sb,        "#6ee7b7"),
            (c5,"Strong Sell", ss,        "#fca5a5"),
            (c6,"Avg Bull%",   f"{avg_conf:.1f}%", "#93c5fd"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{clr}">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Apply Filters ──
        filtered = df_preds.copy()
        if filter_advice:
            filtered = filtered[filtered['Advice'].isin(filter_advice)]
        filtered = filtered[filtered['Bullish_Confidence'] >= min_conf]

        # ── Pie Chart ──
        col_pie, col_bar = st.columns([1,2])
        with col_pie:
            st.markdown('<div class="section-header">Prediction Split</div>', unsafe_allow_html=True)
            fig_pie = px.pie(
                df_preds, names='Prediction',
                color='Prediction',
                color_discrete_map={'Bullish':'#34d399','Bearish':'#f87171'},
                hole=0.55
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=280, margin=dict(t=10,b=10,l=0,r=0),
                legend=dict(orientation="h", x=0.2, y=-0.1)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            st.markdown('<div class="section-header">Advice Distribution</div>', unsafe_allow_html=True)
            advice_order = ['STRONG BUY','BUY','HOLD','SELL','STRONG SELL']
            advice_counts = df_preds['Advice'].value_counts().reindex(advice_order, fill_value=0)
            clrs = ['#059669','#3b82f6','#f59e0b','#ef4444','#dc2626']
            fig_bar = go.Figure(go.Bar(
                x=advice_counts.index, y=advice_counts.values,
                marker_color=clrs, text=advice_counts.values, textposition='outside',
                textfont_color='#e2e8f0'
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,33,50,0.5)',
                font_color='#e2e8f0', height=280, margin=dict(t=10,b=30,l=0,r=0),
                xaxis=dict(gridcolor='#2d3748'), yaxis=dict(gridcolor='#2d3748'),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Confidence Scatter ──
        st.markdown('<div class="section-header">Confidence Scatter — All Stocks</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df_preds, x='Bullish_Confidence', y='Bearish_Confidence',
            color='Prediction', text='Symbol',
            color_discrete_map={'Bullish':'#34d399','Bearish':'#f87171'},
            hover_data=['Company','Advice'],
            size_max=18
        )
        fig_scatter.update_traces(textposition='top center', textfont_size=9)
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,33,50,0.5)',
            font_color='#e2e8f0', height=380, margin=dict(t=10,b=30),
            xaxis=dict(title='Bullish Confidence %', gridcolor='#2d3748'),
            yaxis=dict(title='Bearish Confidence %', gridcolor='#2d3748'),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Table ──
        st.markdown(f'<div class="section-header">Filtered Results ({len(filtered)} stocks)</div>', unsafe_allow_html=True)

        if filtered.empty:
            st.info("No stocks match the current filter settings.")
        else:
            filtered_sorted = filtered.sort_values('Bullish_Confidence', ascending=False).reset_index(drop=True)

            rows_html = ""
            for _, row in filtered_sorted.iterrows():
                pred_cls = "bullish-badge" if row['Prediction']=='Bullish' else "bearish-badge"
                pred_arrow = "▲" if row['Prediction']=='Bullish' else "▼"
                bull_bar = f'<div style="background:#374151;border-radius:4px;height:8px;"><div style="background:#34d399;width:{row["Bullish_Confidence"]}%;height:8px;border-radius:4px;"></div></div>'
                bear_bar = f'<div style="background:#374151;border-radius:4px;height:8px;"><div style="background:#f87171;width:{row["Bearish_Confidence"]}%;height:8px;border-radius:4px;"></div></div>'
                rows_html += f"""
                <tr>
                    <td style="padding:8px;color:#e2e8f0;font-weight:600;">{row['Company']}</td>
                    <td style="padding:8px;color:#8b92a5;font-size:.85rem;">{row['Symbol']}</td>
                    <td style="padding:8px;"><span class="{pred_cls}">{pred_arrow} {row['Prediction']}</span></td>
                    <td style="padding:8px;">{advice_badge(row['Advice'])}</td>
                    <td style="padding:8px;min-width:130px;">{bull_bar}<span style="font-size:.75rem;color:#34d399;">{row['Bullish_Confidence']}%</span></td>
                    <td style="padding:8px;min-width:130px;">{bear_bar}<span style="font-size:.75rem;color:#f87171;">{row['Bearish_Confidence']}%</span></td>
                </tr>"""

            table_html = f"""
            <div style="overflow-x:auto;border-radius:12px;border:1px solid #2d3748;">
            <table style="width:100%;border-collapse:collapse;background:#1a1f35;">
                <thead>
                    <tr style="background:#252a45;border-bottom:1px solid #3d4468;">
                        <th style="padding:10px;color:#8b92a5;text-align:left;font-size:.8rem;">COMPANY</th>
                        <th style="padding:10px;color:#8b92a5;text-align:left;font-size:.8rem;">SYMBOL</th>
                        <th style="padding:10px;color:#8b92a5;text-align:left;font-size:.8rem;">PREDICTION</th>
                        <th style="padding:10px;color:#8b92a5;text-align:left;font-size:.8rem;">ADVICE</th>
                        <th style="padding:10px;color:#8b92a5;text-align:left;font-size:.8rem;">BULL CONF</th>
                        <th style="padding:10px;color:#8b92a5;text-align:left;font-size:.8rem;">BEAR CONF</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table></div>"""
            st.markdown(table_html, unsafe_allow_html=True)

            st.markdown("")
            csv_data = filtered_sorted.to_csv(index=False)
            st.download_button("⬇️ Download Filtered Results (CSV)", csv_data,
                               "filtered_predictions.csv", "text/csv")

    else:
        st.warning(f"⚠️ Predictions CSV not found at `{predictions_csv}`.")
        st.info("Please run `ml_model.py` first to generate `powerbi/predictions_today.csv`, then place it in the correct path.")


# ════════════════════════════════════════════
# TAB 2 – SINGLE STOCK ANALYSIS
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Single Stock Deep Dive</div>', unsafe_allow_html=True)

    col_sel1, col_sel2 = st.columns([2,1])
    with col_sel1:
        selected_name = st.selectbox("Select Stock", list(STOCK_CATEGORIES.keys()))
    with col_sel2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_pred = st.button("🔮 Run Prediction")

    selected_symbol = STOCK_CATEGORIES[selected_name]
    st.caption(f"Symbol: `{selected_symbol}`")

    df_preds2 = load_saved_predictions(predictions_csv)
    if df_preds2 is not None and not df_preds2.empty:
        row = df_preds2[df_preds2['Symbol']==selected_symbol]
        if not row.empty:
            r = row.iloc[0]
            st.markdown("#### 📋 Saved Prediction (from last run)")
            m1,m2,m3,m4 = st.columns(4)
            m1.markdown(f"""<div class="metric-card"><div class="metric-label">Prediction</div>
                <div class="metric-value" style="color:{'#34d399' if r['Prediction']=='Bullish' else '#f87171'}">
                {'▲' if r['Prediction']=='Bullish' else '▼'} {r['Prediction']}</div></div>""",
                unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-card"><div class="metric-label">Advice</div>
                <div class="metric-value" style="font-size:1.1rem;padding-top:.4rem">{advice_badge(r['Advice'])}</div></div>""",
                unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-card"><div class="metric-label">Bullish Conf.</div>
                <div class="metric-value" style="color:#34d399">{r['Bullish_Confidence']}%</div></div>""",
                unsafe_allow_html=True)
            m4.markdown(f"""<div class="metric-card"><div class="metric-label">Bearish Conf.</div>
                <div class="metric-value" style="color:#f87171">{r['Bearish_Confidence']}%</div></div>""",
                unsafe_allow_html=True)
            st.markdown("")

    if run_pred:
        with st.spinner(f"Running ML prediction for {selected_name}..."):
            result, err = predict_single_stock(selected_symbol, model_dir)
        if err:
            st.error(f"Error: {err}")
        else:
            st.markdown("#### 🤖 Live Prediction Result")
            m1,m2,m3,m4 = st.columns(4)
            pred_label = "Bullish" if result['pred']==1 else "Bearish"
            m1.markdown(f"""<div class="metric-card"><div class="metric-label">Prediction</div>
                <div class="metric-value" style="color:{'#34d399' if result['pred']==1 else '#f87171'}">
                {'▲' if result['pred']==1 else '▼'} {pred_label}</div></div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-card"><div class="metric-label">Advice</div>
                <div class="metric-value" style="font-size:1.1rem;padding-top:.4rem">{advice_badge(result['advice'])}</div></div>""",
                unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-card"><div class="metric-label">Bullish Conf.</div>
                <div class="metric-value" style="color:#34d399">{result['bullish_conf']}%</div></div>""",
                unsafe_allow_html=True)
            m4.markdown(f"""<div class="metric-card"><div class="metric-label">Bearish Conf.</div>
                <div class="metric-value" style="color:#f87171">{result['bearish_conf']}%</div></div>""",
                unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['bullish_conf'],
                title={'text': "Bullish Confidence %", 'font':{'color':'#e2e8f0'}},
                delta={'reference': 50, 'increasing':{'color':'#34d399'}, 'decreasing':{'color':'#f87171'}},
                gauge={
                    'axis':{'range':[0,100], 'tickcolor':'#8b92a5'},
                    'bar':{'color':'#6366f1'},
                    'steps':[
                        {'range':[0,30],  'color':'#450a0a'},
                        {'range':[30,50], 'color':'#7f1d1d'},
                        {'range':[50,70], 'color':'#1e3a5f'},
                        {'range':[70,100],'color':'#064e3b'},
                    ],
                    'threshold':{'line':{'color':'white','width':3},'thickness':0.75,'value':70}
                },
                number={'font':{'color':'#e2e8f0'}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                height=260, margin=dict(t=30,b=0)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 – CHARTS & INDICATORS
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Price Charts & Technical Indicators</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([2,1])
    with col_s1:
        chart_stock_name = st.selectbox("Select Stock for Chart", list(STOCK_CATEGORIES.keys()), key="chart_stock")
    with col_s2:
        show_vol = st.checkbox("Show Volume", value=True)

    chart_symbol = STOCK_CATEGORIES[chart_stock_name]

    with st.spinner(f"Fetching {chart_stock_name} data..."):
        df_chart = fetch_stock_data(chart_symbol, period=chart_period)

    if df_chart is not None and not df_chart.empty:
        df_ind = df_chart.copy()
        try:
            import ta
            close  = pd.Series(df_ind['Close'].values.flatten(),  index=df_ind.index)
            high   = pd.Series(df_ind['High'].values.flatten(),   index=df_ind.index)
            low    = pd.Series(df_ind['Low'].values.flatten(),    index=df_ind.index)
            volume = pd.Series(df_ind['Volume'].values.flatten(), index=df_ind.index)

            df_ind['SMA_20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
            df_ind['SMA_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
            df_ind['EMA_20'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            df_ind['BB_Upper'] = bb.bollinger_hband()
            df_ind['BB_Lower'] = bb.bollinger_lband()
            macd_i = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            df_ind['MACD']        = macd_i.macd()
            df_ind['MACD_Signal'] = macd_i.macd_signal()
            df_ind['MACD_Hist']   = macd_i.macd_diff()
            df_ind['RSI']         = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            indicators_ok = True
        except:
            indicators_ok = False

        rows_count = 4 if indicators_ok else 2
        row_heights = [0.5,0.15,0.2,0.15] if indicators_ok else [0.7,0.3]
        specs = [[{"secondary_y": True}]] * rows_count

        fig_main = make_subplots(rows=rows_count, cols=1, shared_xaxes=True,
                                 vertical_spacing=0.03,
                                 row_heights=row_heights, specs=specs)

        close_vals = df_ind['Close'].values.flatten()

        fig_main.add_trace(go.Candlestick(
            x=df_ind.index,
            open=df_ind['Open'].values.flatten(),
            high=df_ind['High'].values.flatten(),
            low=df_ind['Low'].values.flatten(),
            close=close_vals,
            name='Price',
            increasing_line_color='#34d399',
            decreasing_line_color='#f87171'
        ), row=1, col=1)

        if indicators_ok:
            fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['SMA_20'], name='SMA 20',
                line=dict(color='#f59e0b', width=1.2), opacity=0.9), row=1, col=1)
            fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['SMA_50'], name='SMA 50',
                line=dict(color='#a78bfa', width=1.2), opacity=0.9), row=1, col=1)
            fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_Upper'], name='BB Upper',
                line=dict(color='#94a3b8', width=0.8, dash='dash'), opacity=0.5), row=1, col=1)
            fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_Lower'], name='BB Lower',
                line=dict(color='#94a3b8', width=0.8, dash='dash'),
                fill='tonexty', fillcolor='rgba(148,163,184,0.05)', opacity=0.5), row=1, col=1)

        if show_vol:
            vol_colors = ['#34d399' if c >= o else '#f87171'
                          for c, o in zip(df_ind['Close'].values.flatten(),
                                          df_ind['Open'].values.flatten())]
            fig_main.add_trace(go.Bar(x=df_ind.index, y=df_ind['Volume'].values.flatten(),
                name='Volume', marker_color=vol_colors, opacity=0.6), row=2, col=1)

        if indicators_ok:
            hist_colors = ['#34d399' if v >= 0 else '#f87171' for v in df_ind['MACD_Hist'].fillna(0)]
            fig_main.add_trace(go.Bar(x=df_ind.index, y=df_ind['MACD_Hist'],
                name='MACD Hist', marker_color=hist_colors, opacity=0.7), row=3, col=1)
            fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MACD'],
                name='MACD', line=dict(color='#3b82f6', width=1.5)), row=3, col=1)
            fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MACD_Signal'],
                name='Signal', line=dict(color='#f59e0b', width=1.5)), row=3, col=1)

            fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['RSI'],
                name='RSI', line=dict(color='#a78bfa', width=1.5)), row=4, col=1)
            fig_main.add_hline(y=70, line_dash='dash', line_color='#f87171', line_width=1, row=4, col=1)
            fig_main.add_hline(y=30, line_dash='dash', line_color='#34d399', line_width=1, row=4, col=1)

        fig_main.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,24,40,0.9)',
            font_color='#e2e8f0', height=700,
            margin=dict(t=20, b=20, l=50, r=30),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
            title=dict(text=f"{chart_stock_name} ({chart_symbol})", font=dict(color='#c7d2fe', size=16)),
        )
        for i in range(1, rows_count+1):
            fig_main.update_xaxes(gridcolor='#1e2130', showgrid=True, row=i, col=1)
            fig_main.update_yaxes(gridcolor='#1e2130', showgrid=True, row=i, col=1)

        st.plotly_chart(fig_main, use_container_width=True)

        if indicators_ok:
            st.markdown('<div class="section-header">Latest Indicator Values</div>', unsafe_allow_html=True)
            last = df_ind.iloc[-1]
            ci1,ci2,ci3,ci4,ci5 = st.columns(5)
            rsi_val = last.get('RSI', float('nan'))
            rsi_color = '#f87171' if rsi_val > 70 else '#34d399' if rsi_val < 30 else '#e2e8f0'
            for col, lbl, key, clr in [
                (ci1, "RSI (14)",    'RSI',       rsi_color),
                (ci2, "SMA 20",      'SMA_20',    '#f59e0b'),
                (ci3, "SMA 50",      'SMA_50',    '#a78bfa'),
                (ci4, "BB Upper",    'BB_Upper',  '#94a3b8'),
                (ci5, "MACD",        'MACD',      '#3b82f6'),
            ]:
                val = last.get(key, float('nan'))
                disp = f"{val:.2f}" if not pd.isna(val) else "N/A"
                col.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value" style="color:{clr};font-size:1.3rem">{disp}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.error(f"Could not fetch data for {chart_symbol}. Please check your internet connection.")