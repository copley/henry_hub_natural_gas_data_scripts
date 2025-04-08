#!/usr/bin/env python3
"""
SUPER SCRIPT:
Combines:
 - IB API retrieval of 1H / 12H historical data for Natural Gas,
 - NOAA/EIA fundamental aspects,
 - Gann, RSI, MACD, pivot analysis,
 - Additional 'Professional Trader' context (fundamentals, volume/OI, volatility, etc.),
 - EIA weekly storage logic,
 - Single comprehensive prompt for ChatGPT that merges both the
   short-/long-term analysis AND the fundamental drivers from
   the user’s bullet points.

Usage:
  1) Ensure you have installed: ibapi, requests, pandas, numpy, python-dotenv, openai
  2) Create a .env file with:
       OPENAI_API_KEY=YOUR_OPENAI_KEY
       NOAA_API_TOKEN=YOUR_NOAA_TOKEN
       EIA_API_KEY=YOUR_EIA_API_KEY   (if you have one)
  3) Run: python3 this_script.py
"""

import os
import time
import threading
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# -----------------------------
# 1) IB API & OpenAI Setup
# -----------------------------
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

import openai  # Make sure OPENAI_API_KEY is set

# ============ LOGGER SETUP ============
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ============ LOAD ENV VARIABLES ============
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
NOAA_API_TOKEN = os.getenv('NOAA_API_TOKEN', 'MISSING_NOAA_TOKEN')
EIA_API_KEY    = os.getenv('EIA_API_KEY', 'MISSING_EIA_API_KEY')

if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment. Please set it in .env")

# -----------------------------
# 2) Placeholder City / Region Info
# -----------------------------
CITY_INFO = {
    "New York City": {"station_id": "GHCND:USW00094728", "population": 8804190},
    "Boston":        {"station_id": "GHCND:USW00014739", "population": 684379},
    "Chicago":       {"station_id": "GHCND:USW00094846", "population": 2746388},
    "Detroit":       {"station_id": "GHCND:USW00094847", "population": 632464},
    "Dallas":        {"station_id": "GHCND:USW00003927", "population": 1288457},
    "Houston":       {"station_id": "GHCND:USW00012960", "population": 2304580},
    "Denver":        {"station_id": "GHCND:USW00003017", "population": 715522},
    "Salt Lake City":{"station_id": "GHCND:USW00024127", "population": 200133},
    "Los Angeles":   {"station_id": "GHCND:USW00023174", "population": 3898747},
}

REGIONS = {
    "Northeast": ["New York City", "Boston"],
    "Midwest": ["Chicago", "Detroit"],
    "South Central": ["Dallas", "Houston"],
    "Mountain": ["Denver", "Salt Lake City"],
    "Pacific": ["Los Angeles"]
}

HDD_BASE_TEMP = 65
FORECAST_DAYS = 7
HISTORICAL_YEARS = 1

# -----------------------------
# 3) IBApp for 1H Historical Data
# -----------------------------
class IBApp(EWrapper, EClient):
    """
    For retrieving 1-hour historical data (e.g. last N days) from TWS for Natural Gas futures.
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []

    def historicalData(self, reqId, bar):
        self.data.append({
            'Date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        print("Historical data download complete.")

    def error(self, reqId, errorCode, errorString):
        print(f"Error (reqId {reqId}): {errorCode} - {errorString}")


def create_ng_contract():
    """
    Create an IB Contract object for Natural Gas futures on NYMEX.
    Adjust as needed for your contract specs.
    """
    contract = Contract()
    contract.symbol = "NG"
    contract.secType = "FUT"
    contract.exchange = "NYMEX"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = "20250428"  # Example future date
    contract.localSymbol = "NGK5"                      # Example: NGK5
    contract.multiplier = "10000"
    return contract

# -----------------------------
# 4) Basic Technical Indicator Functions
# -----------------------------
def rsi_calculation(series, period=14):
    """
    Calculate RSI.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd_calculation(series, fast=12, slow=26, signal=9):
    """
    MACD
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def detect_swing_points(df, window=3):
    """
    Mark swing highs/lows in a rolling window.
    """
    df.sort_values(by='Date', inplace=True)
    df['isSwingHigh'] = False
    df['isSwingLow'] = False
    for i in range(window, len(df) - window):
        local_high = df.loc[i, 'High']
        if local_high == max(df.loc[i-window : i+window, 'High']):
            df.at[i, 'isSwingHigh'] = True
        local_low = df.loc[i, 'Low']
        if local_low == min(df.loc[i-window : i+window, 'Low']):
            df.at[i, 'isSwingLow'] = True
    return df

def project_gann_cycles(df, cycles=(90, 144, 180, 360), anniv_years=(1,2,3,4,5)):
    """
    Gann cycle projection.
    """
    # --- FIX: avoid SettingWithCopy by using df.copy() if needed; or just do an in-place assignment carefully
    df['Date'] = pd.to_datetime(df['Date'])
    pivots = df[(df['isSwingHigh']) | (df['isSwingLow'])].copy()
    future_turns = []
    for _, pivot in pivots.iterrows():
        pivot_date = pivot['Date']
        pivot_type = 'High' if pivot['isSwingHigh'] else 'Low'
        # Cycle projections
        for c in cycles:
            forecast_date = pivot_date + pd.Timedelta(days=c)
            future_turns.append({
                'pivotDate': pivot_date,
                'pivotType': pivot_type,
                'cycleDays': c,
                'forecastDate': forecast_date
            })
        # Anniversary
        for y in anniv_years:
            anniv_date = pivot_date + pd.DateOffset(years=y)
            future_turns.append({
                'pivotDate': pivot_date,
                'pivotType': pivot_type,
                'cycleDays': f'{y}Y_Anniv',
                'forecastDate': anniv_date
            })
    return pd.DataFrame(future_turns)

def find_upcoming_turns(future_turns_df, hours_ahead=1):
    """
    Filter Gann turns within next X hours.
    """
    now = pd.Timestamp.now()
    upper_bound = now + pd.Timedelta(hours=hours_ahead)
    mask = (future_turns_df['forecastDate'] >= now) & (future_turns_df['forecastDate'] <= upper_bound)
    upcoming = future_turns_df[mask].copy()
    upcoming.sort_values('forecastDate', inplace=True)
    return upcoming

def process_dataframe(df):
    # -----------------
    # FIX #1 to avoid SettingWithCopyWarning:
    # Make a copy of df so we can safely assign:
    # -----------------
    df = df.copy()
    # -----------------
    # FIX #2: Use loc[:, 'Date'] to avoid chained assignment warnings
    # -----------------
    df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'])
    df.sort_values(by='Date', inplace=True)
    df['RSI'] = rsi_calculation(df['Close'], period=14)
    macd_line, macd_signal, macd_hist = macd_calculation(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    df = detect_swing_points(df, window=3)
    return df

def resample_df(df, rule):
    """
    Resample e.g. to '12h'.
    """
    df.set_index('Date', inplace=True)
    # -----------------
    # FIX #3 to avoid FutureWarning:
    # Change '12H' -> '12h' (lowercase h)
    # -----------------
    df_resampled = df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    return df_resampled

# -----------------------------
# 5) Multi-timeframe Analysis
# -----------------------------
def multi_timeframe_analysis(hour_df, resampled_df):
    """
    Print short-term (1H) & long-term (12H) analysis to console.
    Return key data points so we can feed them into placeholders.
    """
    # 1H
    current_price_1h = hour_df['Close'].iloc[-1]
    current_rsi_1h = hour_df['RSI'].iloc[-1]
    last_pivot_low_1h = hour_df[hour_df['isSwingLow']]
    last_pivot_high_1h = hour_df[hour_df['isSwingHigh']]
    pivot_low_1h = last_pivot_low_1h.iloc[-1] if not last_pivot_low_1h.empty else None
    pivot_high_1h = last_pivot_high_1h.iloc[-1] if not last_pivot_high_1h.empty else None
    if pivot_low_1h is not None:
        proj_1h = project_gann_cycles(hour_df.tail(30), cycles=(0.0417,), anniv_years=())
    else:
        proj_1h = pd.DataFrame()

    # 12H
    current_price_12h = resampled_df['Close'].iloc[-1]
    current_rsi_12h = resampled_df['RSI'].iloc[-1]
    last_pivot_low_12h = resampled_df[resampled_df['isSwingLow']]
    last_pivot_high_12h = resampled_df[resampled_df['isSwingHigh']]
    pivot_low_12h = last_pivot_low_12h.iloc[-1] if not last_pivot_low_12h.empty else None
    pivot_high_12h = last_pivot_high_12h.iloc[-1] if not last_pivot_high_12h.empty else None
    if pivot_low_12h is not None:
        proj_12h = project_gann_cycles(resampled_df.tail(30), cycles=(1,), anniv_years=())
    else:
        proj_12h = pd.DataFrame()

    # Print out
    print("=== SHORT-TERM (1-Hour) FORECAST ===")
    print(f"Current Price (1H): {current_price_1h:.3f}")
    print(f"RSI (1H): {current_rsi_1h:.2f}")
    if pivot_low_1h is not None:
        print(f"Recent Pivot Low (1H): {pivot_low_1h['Close']:.3f} at {pivot_low_1h['Date']}")
    if pivot_high_1h is not None:
        print(f"Recent Pivot High (1H): {pivot_high_1h['Close']:.3f} at {pivot_high_1h['Date']}")
    if not proj_1h.empty:
        upcoming_1h = find_upcoming_turns(proj_1h, hours_ahead=1)
        print("\nGann Projections (1H) for next hour:")
        print(upcoming_1h.to_string(index=False))
    else:
        print("No recent pivot detected for 1H forecast.")

    print("\n=== LONG-TERM (12H/24H) FORECAST ===")
    print(f"Current Price (12H): {current_price_12h:.3f}")
    print(f"RSI (12H): {current_rsi_12h:.2f}")
    if pivot_low_12h is not None:
        print(f"Recent Pivot Low (12H): {pivot_low_12h['Close']:.3f} at {pivot_low_12h['Date']}")
    if pivot_high_12h is not None:
        print(f"Recent Pivot High (12H): {pivot_high_12h['Close']:.3f} at {pivot_high_12h['Date']}")
    if not proj_12h.empty:
        upcoming_12h = proj_12h[proj_12h['forecastDate'] <= (pd.Timestamp.now() + pd.Timedelta(days=1))]
        print("\nGann Projections (12H) for next 24 hours:")
        print(upcoming_12h.to_string(index=False))
    else:
        print("No recent pivot detected for 12H forecast.")

    # Combine some signals
    return {
        'current_price_1h': current_price_1h,
        'RSI_1h': current_rsi_1h,
        'pivot_low_1h_val': f"{pivot_low_1h['Close']:.3f}" if pivot_low_1h is not None else "N/A",
        'time_pivot_low_1h': str(pivot_low_1h['Date']) if pivot_low_1h is not None else "N/A",
        'pivot_high_1h_val': f"{pivot_high_1h['Close']:.3f}" if pivot_high_1h is not None else "N/A",
        'time_pivot_high_1h': str(pivot_high_1h['Date']) if pivot_high_1h is not None else "N/A",
        'gann_proj_1h': "No upcoming Gann turning points in the next hour." if proj_1h.empty else "Detected. See console output.",
        'macd_signal_1h': "bullish" if hour_df['MACD'].iloc[-1] > hour_df['MACD_Signal'].iloc[-1] else "bearish",

        'current_price_12h': current_price_12h,
        'RSI_12h': current_rsi_12h,
        'pivot_points_12h': (
            "No significant pivot lows/highs detected on 12H."
            if (pivot_low_12h is None and pivot_high_12h is None)
            else f"PivotLow: {pivot_low_12h['Close']:.3f} at {pivot_low_12h['Date']} | "
                 f"PivotHigh: {pivot_high_12h['Close']:.3f} at {pivot_high_12h['Date']}"
            if (pivot_low_12h is not None and pivot_high_12h is not None)
            else "Partial pivot data."
        ),
        'macd_signal_12h': "bullish" if resampled_df['MACD'].iloc[-1] > resampled_df['MACD_Signal'].iloc[-1] else "bearish",
    }

# -----------------------------
# 6) The “Professional Trader” Extended Prompt
# -----------------------------
def build_professional_context_prompt():
    """
    This is the big block of bullet points describing fundamentals,
    volume/OI analysis, volatility/option markets, risk management, correlations, etc.
    We'll incorporate it into the final ChatGPT prompt.
    """
    return (
        "IMPORTANT ADDITIONAL PROFESSIONAL CONSIDERATIONS:\n"
        "\n"
        "Fundamental Drivers & Market Context:\n"
        " - Supply/Demand Factors: EIA storage reports, production trends, LNG exports, weather forecasts (HDD/CDD), etc.\n"
        " - Macro or Seasonal Factors: Time of year, economic conditions, potential changes in gas demand for heating/cooling.\n"
        " - Relevant News or Events: Scheduled economic data releases, geopolitical tensions, pipeline disruptions, etc.\n"
        "\n"
        "Volume & Open Interest Analysis:\n"
        " - A professional trader often checks futures volume and open interest to confirm technical signals or divergences.\n"
        " - High volume on a breakout or pivot level can strengthen conviction.\n"
        "\n"
        "Volatility & Option Markets:\n"
        " - Implied volatility from options can refine entry/exit. High IV suggests explosive price moves.\n"
        " - Option flow or skew (calls vs. puts) can reveal market positioning.\n"
        "\n"
        "Position Sizing & Advanced Risk Management:\n"
        " - Pros rarely just 'place a stop below support.' They size based on volatility or VaR.\n"
        " - They might scale in/out, hedge with options, or adjust stops dynamically.\n"
        "\n"
        "Event‐Driven Time Horizons:\n"
        " - NG is highly sensitive to weekly EIA storage data, monthly OPEC forecasts.\n"
        " - A 'Plan A / Plan B' approach for bullish/bearish scenarios after these events.\n"
        "\n"
        "Correlation With Broader Energy Complex:\n"
        " - NG rarely trades in isolation. Watch crude oil, refined products, electricity.\n"
        " - Look for divergences or confirmation in correlated markets.\n"
        "\n"
        "Trade Scenarios & Probability:\n"
        " - Instead of just buy/sell, professionals map out scenario analysis (Bullish if X, Bearish if Y).\n"
        " - Probability weights or confidence levels help manage risk–reward.\n"
        "\n"
        "Exit Strategy & Profit‐Taking:\n"
        " - Not just a stop loss; consider partial profits, trailing stops, times of day with low liquidity.\n"
        "\n"
        "Position‐Roll & Contract Nuances:\n"
        " - Futures roll issues, front‐month vs. next‐month spreads, seasonal differences.\n"
        "\n"
        "Detailed Risk–Reward:\n"
        " - Many pros state explicit R:R ratio (risk 1 to make 2 or 3), watch break‐evens, daily P&L checks.\n"
    )

# -----------------------------
# 7) The Original Multi-Timeframe Prompt
# -----------------------------
def build_ng_market_analysis_prompt():
    """
    Original short/long timeframe prompt with placeholders.
    We'll later splice in the 'professional context' text.
    """
    prompt = (
        "You are a highly experienced natural gas trader and technical analyst. I will provide you with detailed "
        "multi‐timeframe analysis data including short‐term (1‑hour) and long‐term (12‑/24‑hour) forecasts, derived "
        "from Gann levels, pivot points, RSI, and MACD. Based on this information, please extract the maximum tradable "
        "insights and give a clear, actionable trading strategy. Use the following data points:\n\n"

        "1. Short-Term (1-Hour) Forecast:\n"
        "   - **Current Price (1H):** {current_price_1h}\n"
        "   - **RSI (1H):** {RSI_1h}\n"
        "   - **Recent Pivot Points (1H):**\n"
        "       - **Pivot Low:** {pivot_low_1h} at {time_pivot_low}\n"
        "       - **Pivot High:** {pivot_high_1h} at {time_pivot_high}\n"
        "   - **Gann Projections (1H) for Next Hour:** {gann_projections_1h}\n"
        "   - **Technical Indicator Signals (1H):**\n"
        "       - **RSI Signal:** {rsi_signal_1h}\n"
        "       - **MACD Signal:** {macd_signal_1h}\n\n"

        "2. Long-Term (12H/24H) Forecast:\n"
        "   - **Current Price (12H):** {current_price_12h}\n"
        "   - **RSI (12H):** {RSI_12h}\n"
        "   - **Recent Pivot Points (12H):** {pivot_points_12h}\n"
        "   - **Technical Indicator Signals (12H):**\n"
        "       - **RSI Signal:** {rsi_signal_12h}\n"
        "       - **MACD Signal:** {macd_signal_12h}\n\n"

        "3. Trading Considerations:\n"
        "   - **Short-Term Considerations:**\n"
        "       - The 1‑hour RSI suggests possible bounce/pullback if oversold/overbought.\n"
        "       - MACD bias for short-term. Look for crossovers or candlestick patterns near pivot.\n"
        "   - **Long-Term Considerations:**\n"
        "       - The 12‑hour data: watch RSI, MACD momentum for bigger trend.\n"
        "       - For a 12‑24 hour horizon, wait for confluence of long-term trend & indicators.\n"
        "   - **General Strategy & Risk Management:**\n"
        "       - Identify areas where both timeframes agree.\n"
        "       - Use stop-loss orders around recent pivots.\n"
        "       - Adjust cycle parameters if Gann projections are weak.\n"
        "       - Always incorporate proper risk management.\n\n"

        "**Task:**\n"
        "Based on this information, provide a comprehensive trading analysis. Summarize actionable signals, suggest trade entries/exits, "
        "risk management, and state whether overall bias is bullish, bearish, or neutral. Highlight caution conditions.\n\n"

        "Please provide a detailed trade recommendation based on the above data.\n"
    )
    return prompt


# -----------------------------
# 8) The Combined Prompt Builder
# -----------------------------
def build_combined_prompt(short_long_data, fundamental_text):
    """
    Takes the original multi-timeframe prompt + the new fundamental section
    and merges them into one big prompt, updating placeholders from 'short_long_data'.
    """
    base_prompt = build_ng_market_analysis_prompt()

    # Replace placeholders:
    prompt_filled = base_prompt.replace("{current_price_1h}", f"{short_long_data['current_price_1h']:.3f}")
    prompt_filled = prompt_filled.replace("{RSI_1h}", f"{short_long_data['RSI_1h']:.2f}")
    prompt_filled = prompt_filled.replace("{pivot_low_1h}", short_long_data['pivot_low_1h_val'])
    prompt_filled = prompt_filled.replace("{time_pivot_low}", short_long_data['time_pivot_low_1h'])
    prompt_filled = prompt_filled.replace("{pivot_high_1h}", short_long_data['pivot_high_1h_val'])
    prompt_filled = prompt_filled.replace("{time_pivot_high}", short_long_data['time_pivot_high_1h'])
    prompt_filled = prompt_filled.replace("{gann_projections_1h}", short_long_data['gann_proj_1h'])
    # RSI & MACD signals for 1H
    if short_long_data['RSI_1h'] < 30:
        rsi_signal_1h = "oversold"
    elif short_long_data['RSI_1h'] > 70:
        rsi_signal_1h = "overbought"
    else:
        rsi_signal_1h = "neutral"
    prompt_filled = prompt_filled.replace("{rsi_signal_1h}", rsi_signal_1h)
    prompt_filled = prompt_filled.replace("{macd_signal_1h}", short_long_data['macd_signal_1h'])

    # 12H
    prompt_filled = prompt_filled.replace("{current_price_12h}", f"{short_long_data['current_price_12h']:.3f}")
    prompt_filled = prompt_filled.replace("{RSI_12h}", f"{short_long_data['RSI_12h']:.2f}")
    prompt_filled = prompt_filled.replace("{pivot_points_12h}", short_long_data['pivot_points_12h'])
    if short_long_data['RSI_12h'] < 30:
        rsi_signal_12h = "oversold"
    elif short_long_data['RSI_12h'] > 70:
        rsi_signal_12h = "overbought"
    elif str(short_long_data['RSI_12h']) == "nan":
        rsi_signal_12h = "NaN"
    else:
        rsi_signal_12h = "neutral"
    prompt_filled = prompt_filled.replace("{rsi_signal_12h}", rsi_signal_12h)
    prompt_filled = prompt_filled.replace("{macd_signal_12h}", short_long_data['macd_signal_12h'])

    # Merge fundamental text
    final_prompt = (
        prompt_filled
        + "\n\n"
        + fundamental_text
        + "\n\n"
        "Please consider all these professional/trader-level fundamentals and the multi-timeframe technical data "
        "when giving your final analysis and trade recommendation.\n"
    )

    return final_prompt


# -----------------------------
# 9) get_ai_analysis with a larger max_tokens
# -----------------------------
def get_ai_analysis(prompt):
    response = openai.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# 10) NOAA/EIA Storage placeholders (OPTIONAL)
# -----------------------------
def fetch_mock_fundamental_data():
    """
    In a real script, you'd gather fundamental data (EIA storage yoy,
    NOAA HDD/CDD, etc.). We’ll just return a placeholder for demonstration.
    """
    return {
        "supply_demand": "Storage is slightly below 5-year avg; LNG exports remain high.",
        "macro_factors": "Mild weather so far, but a cold front is predicted next week.",
        "news_events": "Weekly EIA draw was bigger than expected. Geopolitical tensions in X region."
    }

# -----------------------------
# 11) The Main Super Script
# -----------------------------
def main():
    print("=== STARTING SUPER SCRIPT ===")

    # 1) Connect to TWS, retrieve 1-Hour data
    app = IBApp()
    app.connect("127.0.0.1", 7496, clientId=87)
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    time.sleep(1)

    # Request 1-hour data for 365 days
    contract = create_ng_contract()
    end_date_time = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S")
    duration = "365 D"
    bar_size = "1 hour"

    print(f"Requesting historical data: end={end_date_time}, duration={duration}, barSize={bar_size}")
    app.reqHistoricalData(
        reqId=1,
        contract=contract,
        endDateTime=end_date_time,
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=0,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    # 2) Wait for data or time out.
    time.sleep(120)  # 2 minutes
    if not app.data:
        print("No historical data received from TWS. Exiting.")
        return

    # 3) Build a DataFrame & process it for 1h
    hour_df = pd.DataFrame(app.data)

    # -- We won't remove the existing logic, but to avoid SettingWithCopy, let's do a loc assignment here as well:
    hour_df.loc[:, 'Date'] = pd.to_datetime(hour_df.loc[:, 'Date'])

    hour_df = process_dataframe(hour_df)

    # 4) Resample to 12-hour
    # -- FIX: '12H' -> '12h'
    resampled_df = resample_df(hour_df.copy(), '12h')
    resampled_df = process_dataframe(resampled_df)

    # 5) Multi-timeframe analysis
    short_long_data = multi_timeframe_analysis(hour_df, resampled_df)

    # 6) Professional context text
    fundamental_text = build_professional_context_prompt()

    # 7) Combine into final prompt
    combined_prompt = build_combined_prompt(short_long_data, fundamental_text)

    # 8) Send prompt to GPT
    print("\n\n=== FINAL PROMPT ===\n")
    print(combined_prompt)
    print("\n=== ChatGPT Analysis & Recommendation ===\n")
    analysis_response = get_ai_analysis(combined_prompt)
    print(analysis_response)

    # 9) Disconnect from TWS
    app.disconnect()
    print("=== END OF SUPER SCRIPT ===")


if __name__ == "__main__":
    main()
