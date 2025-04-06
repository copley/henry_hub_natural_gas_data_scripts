from ib_insync import *
import pandas as pd
import numpy as np
import datetime as dt
import pandas_ta as ta
"""
This Python script connects to Interactive Brokers' TWS using ib_insync and retrieves 
one year of hourly historical data for the May 2025 Natural Gas futures contract (NGK5). 
It calculates technical indicators including RSI, CCI, ADX, volume, and price ranges 
using pandas_ta, and applies a mean-reversion trading strategy to identify long or 
short entry signals. For each signal, it computes the 24-hour future return and evaluates 
historical average performance. Based on the current market conditions and past signal 
success, it prints a 24-hour trading bias forecast: Long, Short, or Neutral. 
All results are displayed directly in the terminal.
"""

# Setup
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=2)

# Define NG contract
contract = Future(
    symbol='NG',
    lastTradeDateOrContractMonth='202505',
    exchange='NYMEX',
    currency='USD'
)

# Get 1 year of hourly data (max: 1 month at a time, so loop it)
def get_1_year_hourly(contract):
    all_data = []
    end_time = dt.datetime.now()
    for _ in range(12):
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_time,
            durationStr='30 D',
            barSizeSetting='1 hour',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        df = util.df(bars)
        all_data.append(df)
        if not df.empty:
            end_time = df['date'].min() - dt.timedelta(hours=1)
    return pd.concat(all_data).drop_duplicates().sort_values('date').reset_index(drop=True)

df = get_1_year_hourly(contract)

# Add indicators
df['RSI'] = ta.rsi(df['close'], length=14)
df['CCI'] = ta.cci(high=df['high'], low=df['low'], close=df['close'], length=14)
df['ADX'] = ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14)['ADX_14']
df['VolMA10'] = df['volume'].rolling(10).mean()
df['VolHigh20'] = df['volume'].rolling(20).max()
df['CloseHigh20'] = df['close'].rolling(20).max()
df['CloseLow13'] = df['close'].rolling(13).min()
df['CloseHigh13'] = df['close'].rolling(13).max()

# Range calculations
df['R1X'] = df['high'].shift(1) - df['low'].shift(1)
df['R2X'] = df['high'].shift(2) - df['low'].shift(2)
df['R3X'] = df['high'].shift(3) - df['low'].shift(3)
df['RangeCond'] = df['R1X'] < (df['R2X'] + df['R3X']) / 3

# Strategy Logic
df['LongSignal'] = (
    (df['volume'].shift(1) > df['VolHigh20'].shift(1)) &
    (df['close'].shift(1) <= df['CloseHigh20'].shift(1)) &
    (df['RSI'] <= 70) &
    (df['volume'] > df['VolMA10']) &
    df['RangeCond'] &
    (df['close'] == df['CloseHigh13'])
)

df['ShortSignal'] = (
    (df['RSI'] < 40) &
    (df['CCI'] < 0) &
    (df['ADX'] <= 25) &
    (df['close'] == df['CloseLow13'])
)

# Calculate 24-hour future return for each signal
df['future_return_24h'] = df['close'].shift(-24) / df['close'] - 1

long_returns = df[df['LongSignal']]['future_return_24h'].dropna()
short_returns = df[df['ShortSignal']]['future_return_24h'].dropna()

# Get current signal
current_row = df.iloc[-1]
long_now = df['LongSignal'].iloc[-1]
short_now = df['ShortSignal'].iloc[-1]

print("\n--- Natural Gas 24-Hour Position Bias Forecast ---")
print(f"Date: {current_row['date']}")
print(f"Current Price: {current_row['close']:.4f}")

# Evaluate bias
if long_now:
    avg_gain = long_returns.mean()
    print(f"Signal: LONG")
    print(f"Historical Avg Return (next 24h after long): {avg_gain:.4%}")
    if avg_gain > 0:
        print("➡️  Bias: LONG BIAS for next 24 hours")
    else:
        print("⚠️  Bias: WEAK LONG (low or negative avg return historically)")
elif short_now:
    avg_gain = short_returns.mean()
    print(f"Signal: SHORT")
    print(f"Historical Avg Return (next 24h after short): {avg_gain:.4%}")
    if avg_gain < 0:
        print("➡️  Bias: SHORT BIAS for next 24 hours")
    else:
        print("⚠️  Bias: WEAK SHORT (low or positive avg return historically)")
else:
    print("Signal: NO TRADE")
    print("➡️  Bias: NEUTRAL for next 24 hours")

ib.disconnect()

