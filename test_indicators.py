# test_indicators.py

from screener import download_data
from indicators import add_indicators

# Step 1: Kisi ek stock ka data lao
data = download_data(["TCS.NS"])
df = data["TCS.NS"]

# Step 2: Indicators apply karo
df = add_indicators(df)

# Step 3: Sirf kuch columns print karke check karo
print(df[['Close', 'RSI', 'SMA_20', 'EMA_20', 'MACD', 'MACD_Signal']].tail())
