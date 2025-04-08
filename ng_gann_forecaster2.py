import os
import time
import threading
import pandas as pd
import numpy as np

from datetime import datetime
from dotenv import load_dotenv

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# ============ LOAD ENVIRONMENT VARIABLES (.env) ============
load_dotenv()  # Reads .env file and loads variables
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key

# ============ PROMPT-BUILDING FUNCTIONS FOR ChatGPT ============

def build_ng_market_analysis_prompt():
    """
    Builds a dynamic prompt that instructs ChatGPT to provide a detailed,
    multi-timeframe natural gas trading analysis (1-hour, 12H/24H)
    incorporating Gann levels, pivot points, RSI, and MACD.

    HOW TO USE:
    1. Replace the placeholder variables inside { } with your real-time data.
    2. Pass the resulting prompt to the get_ai_analysis() function.
    """

    prompt = (
        "You are a highly experienced natural gas trader and technical analyst. I will provide you with detailed "
        "multi‐timeframe analysis data including short‐term (1‑hour) and long‐term (12‑/24‑hour) forecasts, derived "
        "from Gann levels, pivot points, RSI, and MACD. Based on this information, please extract the maximum tradable "
        "insights and give a clear, actionable trading strategy. Use the following data points:\n\n"

        "1. Short-Term (1-Hour) Forecast:\n"
        "   - **Current Price (1H):** {current_price_1h}\n"
        "     (This is the last closing price from the 1‑hour bars.)\n"
        "   - **RSI (1H):** {RSI_1h}\n"
        "     (An RSI of {RSI_1h} indicates that the market is [if below 30 then \"oversold and may be due for a rebound\"; "
        "if above 70 then \"overbought and may reverse downward\"; otherwise \"neutral\"].)\n"
        "   - **Recent Pivot Points (1H):**\n"
        "       - **Pivot Low:** {pivot_low_1h} at {time_pivot_low}\n"
        "         (This level is the most recent significant low in the last 30 bars and can act as support.)\n"
        "       - **Pivot High:** {pivot_high_1h} at {time_pivot_high}\n"
        "         (This level is the recent significant high and can act as resistance.)\n"
        "   - **Gann Projections (1H) for Next Hour:** {gann_projections_1h}\n"
        "       (Note: If this is an empty result, it means no projected turning point fell within the next hour. Consider "
        "adjusting the cycle projection parameters.)\n"
        "   - **Technical Indicator Signals (1H):**\n"
        "       - **RSI Signal:** {rsi_signal_1h} (e.g. \"oversold\", suggesting potential bullish reversal)\n"
        "       - **MACD Signal:** {macd_signal_1h} (e.g. \"bearish\" if the MACD line is below its signal line)\n\n"

        "2. Long-Term (12H/24H) Forecast:\n"
        "   - **Current Price (12H):** {current_price_12h}\n"
        "     (The current price as computed on the resampled 12‑hour data.)\n"
        "   - **RSI (12H):** {RSI_12h}\n"
        "     (Note: If this is reported as “NaN” due to insufficient data, please mention that the RSI could not be computed reliably.)\n"
        "   - **Recent Pivot Points (12H):** {pivot_points_12h}\n"
        "     (If none are detected, note that the 12‑hour timeframe appears “smooth” with minimal significant swings.)\n"
        "   - **Technical Indicator Signals (12H):**\n"
        "       - **RSI Signal:** {rsi_signal_12h} (e.g., \"neutral\" or \"oversold/overbought\")\n"
        "       - **MACD Signal:** {macd_signal_12h} (e.g., indicating \"bullish\" or \"bearish\" momentum)\n\n"

        "3. Trading Considerations:\n"
        "   - **Short-Term Considerations:**\n"
        "       - The 1‑hour RSI at {RSI_1h} suggests that the market is [if oversold then \"potentially due for a bounce "
        "near the recent pivot low of {pivot_low_1h}\"; if overbought then \"likely to experience a pullback near the recent pivot "
        "high of {pivot_high_1h}\"]\n"
        "       - However, the MACD on the 1‑hour chart indicates a {macd_signal_1h} bias. Look for additional confirmation such as "
        "a bullish crossover or reversal candlestick pattern near {pivot_low_1h} before entering a long position.\n"
        "   - **Long-Term Considerations:**\n"
        "       - The 12‑hour data shows a current price of {current_price_12h} and an RSI of {RSI_12h} ([if NaN, then "
        "\"RSI not available; additional data may be needed\"]).\n"
        "       - With MACD indicating {macd_signal_12h} momentum on the long-term chart, consider monitoring for a break above "
        "resistance or a reversal at key levels.\n"
        "       - If you plan to trade on a 12‑to‑24‑hour horizon, wait for clearer confluence between long‑term trend and "
        "supporting indicators.\n"
        "   - **General Strategy & Risk Management:**\n"
        "       - Identify areas where both timeframes agree (for instance, short-term oversold conditions confirmed by bullish MACD "
        "crossovers may support a long trade).\n"
        "       - Utilize stop-loss orders just below the recent support (e.g. below {pivot_low_1h}) for long positions or above "
        "recent resistance for short positions.\n"
        "       - Consider fine-tuning cycle parameters if Gann projections are absent or weak.\n"
        "       - Always incorporate proper risk management and adjust position sizes based on market volatility.\n\n"

        "**Task:**\n"
        "Based on this information, provide a comprehensive trading analysis. Please summarize the actionable signals and suggest "
        "specific trade entries, exits, and risk management techniques. Also, state whether the overall market bias appears bullish, "
        "bearish, or neutral, and indicate any conditions that warrant caution.\n\n"

        "Notes:\n"
        "- Replace all the placeholder variables (everything in {curly brackets}) with your updated real-time data before sending "
        "the prompt.\n"
        "- Use conditional language where needed (e.g. \"if RSI is below 30, then …\", \"if the MACD is bearish, then …\").\n"
        "- Be as detailed as necessary to extract the maximum tradable information from this multi‑timeframe analysis.\n\n"

        "Please provide a detailed trade recommendation based on the above data."
    )

    return prompt

def get_ai_analysis(prompt):
    """
    Takes the built prompt and sends it to the OpenAI API to get the analysis.
    Adjust parameters as needed (model, max_tokens, temperature, etc.).
    """
    response = openai.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# ============ IB API APPLICATION: Retrieve Historical Data ============

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # To store historical data bars

    def historicalData(self, reqId, bar):
        """Called for each historical data bar."""
        self.data.append({
            'Date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        """Called when historical data download is complete."""
        print("Historical data download complete.")

    def error(self, reqId, errorCode, errorString):
        print(f"Error (reqId {reqId}): {errorCode} - {errorString}")


def create_ng_contract():
    """
    Create an IB Contract object for Natural Gas futures on NYMEX.
    (Adjust parameters as needed.)
    """
    contract = Contract()
    contract.symbol = "NG"                # Symbol
    contract.secType = "FUT"             # Security Type
    contract.exchange = "NYMEX"          # Exchange
    contract.currency = "USD"            # Currency
    contract.lastTradeDateOrContractMonth = "20250428"  # Expiry
    contract.localSymbol = "NGK5"        # Local Symbol
    contract.multiplier = "10000"        # Multiplier
    return contract

# ============ TECHNICAL INDICATOR FUNCTIONS ============

def rsi_calculation(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
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
    Calculate the MACD line, signal line, and histogram.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

# ============ GANN ANALYSIS FUNCTIONS ============

def detect_swing_points(df, window=3):
    """
    Detect swing highs and lows using a simple rolling-window.
    A bar is flagged as swing high/low if its high/low is the maximum/minimum in a window.
    """
    df.sort_values(by='Date', inplace=True)
    df['isSwingHigh'] = False
    df['isSwingLow'] = False

    for i in range(window, len(df) - window):
        # For swing highs
        local_high = df.loc[i, 'High']
        if local_high == max(df.loc[i-window : i+window, 'High']):
            df.at[i, 'isSwingHigh'] = True
        # For swing lows
        local_low = df.loc[i, 'Low']
        if local_low == min(df.loc[i-window : i+window, 'Low']):
            df.at[i, 'isSwingLow'] = True
    return df

def project_gann_cycles(df, cycles=(90, 144, 180, 360), anniv_years=(1,2,3,4,5)):
    """
    Project forward Gann cycles (in days) and anniversaries from detected pivot points.
    (For this example, cycles can be tweaked for a short horizon by using smaller numbers.)
    """
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
        # Anniversary projections
        for y in anniv_years:
            anniv_date = pivot_date + pd.DateOffset(years=y)
            future_turns.append({
                'pivotDate': pivot_date,
                'pivotType': pivot_type,
                'cycleDays': f'{y}Y_Anniv',
                'forecastDate': anniv_date
            })

    future_turns_df = pd.DataFrame(future_turns)
    return future_turns_df

def find_upcoming_turns(future_turns_df, hours_ahead=1):
    """
    Return forecast dates that fall within the next 'hours_ahead' hours.
    The original function works in days; here, we convert hours_ahead to days.
    """
    now = pd.Timestamp.now()
    upper_bound = now + pd.Timedelta(hours=hours_ahead)
    mask = (future_turns_df['forecastDate'] >= now) & (future_turns_df['forecastDate'] <= upper_bound)
    upcoming = future_turns_df[mask].copy()
    upcoming.sort_values('forecastDate', inplace=True)
    return upcoming

# ============ DATA PROCESSING FUNCTIONS ============

def process_dataframe(df):
    """
    Process the DataFrame: ensure datetime conversion, calculate RSI and MACD,
    and detect swing points.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    # Calculate RSI on 'Close'
    df['RSI'] = rsi_calculation(df['Close'], period=14)
    # Calculate MACD on 'Close'
    macd_line, macd_signal, macd_hist = macd_calculation(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    # Detect swing highs/lows
    df = detect_swing_points(df, window=3)
    return df

def resample_df(df, rule):
    """
    Resample the DataFrame to a higher time frame (e.g., '12H').
    """
    df.set_index('Date', inplace=True)
    df_resampled = df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    return df_resampled

# ============ MULTI-TIMEFRAME ANALYSIS & FORECASTING ============

def multi_timeframe_analysis(hour_df, resampled_df):
    """
    Combine technical indicators with Gann levels across two time frames.
    Produces a short-term (1-hour) and a longer-term (12/24-hour) forecast.
    """
    # --- Short-term (1-Hour) Data ---
    current_price_1h = hour_df['Close'].iloc[-1]
    current_rsi_1h = hour_df['RSI'].iloc[-1]
    last_pivot_low_1h = hour_df[hour_df['isSwingLow']]
    last_pivot_high_1h = hour_df[hour_df['isSwingHigh']]
    pivot_low_1h = last_pivot_low_1h.iloc[-1] if not last_pivot_low_1h.empty else None
    pivot_high_1h = last_pivot_high_1h.iloc[-1] if not last_pivot_high_1h.empty else None

    # For short-term Gann cycle, use ~1-hour projection
    if pivot_low_1h is not None:
        proj_1h = project_gann_cycles(hour_df.tail(30), cycles=(0.0417,), anniv_years=())
    else:
        proj_1h = pd.DataFrame()

    # --- Long-term (12H/24H) Data ---
    current_price_12h = resampled_df['Close'].iloc[-1]
    current_rsi_12h = resampled_df['RSI'].iloc[-1]
    last_pivot_low_12h = resampled_df[resampled_df['isSwingLow']]
    last_pivot_high_12h = resampled_df[resampled_df['isSwingHigh']]
    pivot_low_12h = last_pivot_low_12h.iloc[-1] if not last_pivot_low_12h.empty else None
    pivot_high_12h = last_pivot_high_12h.iloc[-1] if not last_pivot_high_12h.empty else None

    if pivot_low_12h is not None:
        # 1 day cycle on 12H data => ~24 hours
        proj_12h = project_gann_cycles(resampled_df.tail(30), cycles=(1,), anniv_years=())
    else:
        proj_12h = pd.DataFrame()

    # --- Display Short-term Analysis ---
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

    # --- Display Long-term Analysis ---
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

    # --- Combined Technical Indicator Interpretation ---
    print("\n=== TECHNICAL INDICATOR SIGNALS ===")
    # RSI analysis
    if current_rsi_1h < 30:
        print("Short-term (1H) RSI is oversold: potential bullish reversal.")
    elif current_rsi_1h > 70:
        print("Short-term (1H) RSI is overbought: potential bearish reversal.")
    else:
        print("Short-term (1H) RSI is neutral.")

    if current_rsi_12h < 30:
        print("Long-term (12H) RSI is oversold: potential bullish reversal.")
    elif current_rsi_12h > 70:
        print("Long-term (12H) RSI is overbought: potential bearish reversal.")
    else:
        print("Long-term (12H) RSI is neutral.")

    # MACD analysis
    last_macd_1h = hour_df['MACD'].iloc[-1]
    last_macd_signal_1h = hour_df['MACD_Signal'].iloc[-1]
    if last_macd_1h > last_macd_signal_1h:
        print("Short-term (1H) MACD indicates bullish momentum.")
    else:
        print("Short-term (1H) MACD indicates bearish momentum.")

    last_macd_12h = resampled_df['MACD'].iloc[-1]
    last_macd_signal_12h = resampled_df['MACD_Signal'].iloc[-1]
    if last_macd_12h > last_macd_signal_12h:
        print("Long-term (12H) MACD indicates bullish momentum.")
    else:
        print("Long-term (12H) MACD indicates bearish momentum.")

# ============ MAIN ROUTINE ============

def main():
    # 1. Connect to TWS using IB API
    app = IBApp()
    app.connect("127.0.0.1", 7496, clientId=87)

    # Start the IB API thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    time.sleep(1)  # Allow connection time

    # 2. Request 1-Hour historical data (e.g., last 2 days)
    contract = create_ng_contract()
    end_date_time = datetime.now().strftime("%Y%m%d %H:%M:%S")
    duration = "2 D"
    bar_size = "1 hour"
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

    # Wait for data
    time.sleep(10)
    if not app.data:
        print("No historical data received from TWS.")
        return

    # 3. Create DataFrame from IB data and process it for the 1-hour timeframe
    hour_df = pd.DataFrame(app.data)
    try:
        hour_df['Date'] = pd.to_datetime(hour_df['Date'])
    except Exception as e:
        hour_df['Date'] = pd.to_datetime(hour_df['Date'], unit='s')

    hour_df = process_dataframe(hour_df)

    # 4. Resample the 1-hour data to a 12-hour timeframe
    resampled_df = resample_df(hour_df.copy(), '12H')
    resampled_df = process_dataframe(resampled_df)

    # 5. Perform multi-timeframe analysis
    multi_timeframe_analysis(hour_df, resampled_df)

    # ---------------------------
    # 6. Demonstrate calling ChatGPT with real placeholder values
    # ---------------------------

    # Prepare the same variables we saw in multi_timeframe_analysis:
    # Short-term
    current_price_1h = hour_df['Close'].iloc[-1]
    rsi_1h = hour_df['RSI'].iloc[-1]
    # Generate a quick RSI signal label
    if rsi_1h < 30:
        rsi_signal_1h = "oversold"
    elif rsi_1h > 70:
        rsi_signal_1h = "overbought"
    else:
        rsi_signal_1h = "neutral"

    last_macd_1h = hour_df['MACD'].iloc[-1]
    last_macd_signal_1h = hour_df['MACD_Signal'].iloc[-1]
    macd_signal_1h = "bullish" if last_macd_1h > last_macd_signal_1h else "bearish"

    pivots_low_1h = hour_df[hour_df['isSwingLow']]
    pivot_low_1h = pivots_low_1h.iloc[-1] if not pivots_low_1h.empty else None
    if pivot_low_1h is not None:
        pivot_low_1h_val = f"{pivot_low_1h['Close']:.3f}"
        pivot_low_1h_time = str(pivot_low_1h['Date'])
    else:
        pivot_low_1h_val = "N/A"
        pivot_low_1h_time = "N/A"

    pivots_high_1h = hour_df[hour_df['isSwingHigh']]
    pivot_high_1h = pivots_high_1h.iloc[-1] if not pivots_high_1h.empty else None
    if pivot_high_1h is not None:
        pivot_high_1h_val = f"{pivot_high_1h['Close']:.3f}"
        pivot_high_1h_time = str(pivot_high_1h['Date'])
    else:
        pivot_high_1h_val = "N/A"
        pivot_high_1h_time = "N/A"

    # For Gann 1H, let's see if any upcoming turn was found:
    # (We'll just do a quick check like in multi_timeframe_analysis)
    from copy import deepcopy
    hour_tail = deepcopy(hour_df.tail(30))
    if pivot_low_1h is not None:
        proj_1h = project_gann_cycles(hour_tail, cycles=(0.0417,), anniv_years=())
        upcoming_1h = find_upcoming_turns(proj_1h, hours_ahead=1)
        if not upcoming_1h.empty:
            gann_projections_1h = upcoming_1h.to_string(index=False)
        else:
            gann_projections_1h = "No upcoming Gann turning points in the next hour."
    else:
        gann_projections_1h = "No recent pivot was detected, so no 1H Gann projection."

    # Long-term
    current_price_12h = resampled_df['Close'].iloc[-1]
    rsi_12h = resampled_df['RSI'].iloc[-1]
    if pd.isna(rsi_12h):
        rsi_signal_12h = "NaN"
    elif rsi_12h < 30:
        rsi_signal_12h = "oversold"
    elif rsi_12h > 70:
        rsi_signal_12h = "overbought"
    else:
        rsi_signal_12h = "neutral"

    last_macd_12h = resampled_df['MACD'].iloc[-1]
    last_macd_signal_12h = resampled_df['MACD_Signal'].iloc[-1]
    macd_signal_12h = "bullish" if last_macd_12h > last_macd_signal_12h else "bearish"

    # Let's define pivot_points_12h as a small string (just listing the latest pivot low/high)
    pivots_low_12h = resampled_df[resampled_df['isSwingLow']]
    pivot_low_12h_ = pivots_low_12h.iloc[-1] if not pivots_low_12h.empty else None

    pivots_high_12h = resampled_df[resampled_df['isSwingHigh']]
    pivot_high_12h_ = pivots_high_12h.iloc[-1] if not pivots_high_12h.empty else None

    if pivot_low_12h_ is None and pivot_high_12h_ is None:
        pivot_points_12h = "No significant pivot lows/highs detected on 12H."
    else:
        desc_low = f"Low={pivot_low_12h_['Close']:.3f} at {pivot_low_12h_['Date']}" if pivot_low_12h_ is not None else "None"
        desc_high = f"High={pivot_high_12h_['Close']:.3f} at {pivot_high_12h_['Date']}" if pivot_high_12h_ is not None else "None"
        pivot_points_12h = f"PivotLow: {desc_low}; PivotHigh: {desc_high}"

    # ---------------------------
    # Build the prompt and replace placeholders
    # ---------------------------
    prompt_text = build_ng_market_analysis_prompt()

    prompt_text = prompt_text.replace("{current_price_1h}", f"{current_price_1h:.3f}")
    prompt_text = prompt_text.replace("{RSI_1h}", f"{rsi_1h:.2f}")
    prompt_text = prompt_text.replace("{rsi_signal_1h}", rsi_signal_1h)
    prompt_text = prompt_text.replace("{macd_signal_1h}", macd_signal_1h)

    prompt_text = prompt_text.replace("{pivot_low_1h}", pivot_low_1h_val)
    prompt_text = prompt_text.replace("{time_pivot_low}", pivot_low_1h_time)
    prompt_text = prompt_text.replace("{pivot_high_1h}", pivot_high_1h_val)
    prompt_text = prompt_text.replace("{time_pivot_high}", pivot_high_1h_time)
    prompt_text = prompt_text.replace("{gann_projections_1h}", gann_projections_1h)

    prompt_text = prompt_text.replace("{current_price_12h}", f"{current_price_12h:.3f}")
    prompt_text = prompt_text.replace("{RSI_12h}", f"{rsi_12h:.2f}")
    prompt_text = prompt_text.replace("{rsi_signal_12h}", rsi_signal_12h)
    prompt_text = prompt_text.replace("{macd_signal_12h}", macd_signal_12h)
    prompt_text = prompt_text.replace("{pivot_points_12h}", pivot_points_12h)

    # Finally, send the prompt to ChatGPT
    analysis_response = get_ai_analysis(prompt_text)
    print("\n\n=== ChatGPT Analysis & Recommendation ===\n")
    print(analysis_response)

    # 7. Disconnect from TWS
    app.disconnect()

if __name__ == "__main__":
    main()
