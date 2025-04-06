import os
import time
import requests
import openai
import pandas as pd
import numpy as np
from tabulate import tabulate

from datetime import datetime, timedelta
from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper
import threading

###############################################################################
# LOAD ENVIRONMENT VARIABLES
###############################################################################
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY environment variable is not set.")
openai.api_key = OPENAI_API_KEY

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def parse_ib_date(date_str):
    if " " in date_str:
        try:
            return datetime.strptime(date_str, "%Y%m%d %H:%M:%S")
        except ValueError:
            pass
    if len(date_str) == 8:
        dt = datetime.strptime(date_str, "%Y%m%d")
        dt = dt.replace(hour=23, minute=59, second=59)
        return dt
    return date_str

def compute_moving_average(series, window):
    return series.rolling(window=window).mean()

def compute_price_change(series, window):
    return series - series.shift(window)

def compute_percent_change(series, window):
    old_price = series.shift(window)
    return (series - old_price) / (old_price) * 100

def compute_average_volume(volume_series, window):
    return volume_series.rolling(window=window).mean()

def compute_stochastics(df, period=14, k_smooth=3, d_smooth=3):
    low_min = df['low'].rolling(period).min()
    high_max = df['high'].rolling(period).max()
    raw_stoch = (df['close'] - low_min) / (high_max - low_min) * 100
    k = raw_stoch.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return raw_stoch, k, d

def compute_atr(df, period=14):
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['true_range'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(period).mean()
    return df['atr']

def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.reindex(series.index)

def compute_percent_r(df, period=14):
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    return (high_max - df['close']) / (high_max - low_min) * -100

def compute_historic_volatility(series, period=20):
    log_ret = np.log(series / series.shift(1))
    stdev = log_ret.rolling(window=period).std()
    return stdev * np.sqrt(260) * 100

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def compute_pivots_s_r(H, L, C):
    pp = (H + L + C)/3
    r1 = 2*pp - L
    s1 = 2*pp - H
    r2 = pp + (r1 - s1)
    s2 = pp - (r1 - s1)
    r3 = H + 2*(pp - L)
    s3 = L - 2*(H - pp)
    return pp, r1, r2, r3, s1, s2, s3

###############################################################################
# REPLACED THE UNUSUAL STD DEV FUNCTION WITH A 20-BAR ROLLING APPROACH
###############################################################################
def compute_std_devs(series_close, stdev_multiplier=[1,2,3]):
    # Calculate the 20-bar rolling mean and std for the close
    rolling_mean_20 = series_close.rolling(20).mean()
    rolling_std_20  = series_close.rolling(20).std()

    # Get the latest (most recent bar) mean and std
    last_mean_20 = rolling_mean_20.iloc[-1]
    last_std_20  = rolling_std_20.iloc[-1]

    out = {}
    for m in stdev_multiplier:
        out[m] = last_std_20 * m

    return out, last_mean_20

def mock_trend_seeker(df):
    short_ema = df['close'].ewm(span=10).mean().iloc[-1]
    long_ema  = df['close'].ewm(span=50).mean().iloc[-1]
    last_price = df['close'].iloc[-1]
    if short_ema > long_ema and last_price > short_ema:
        return 1
    elif short_ema < long_ema and last_price < short_ema:
        return -1
    return 0

def indicator_signal(value, threshold_up=0, threshold_down=0):
    if value > threshold_up:
        return 1
    elif value < threshold_down:
        return -1
    else:
        return 0

def barchart_opinion_logic(df):
    last = df.iloc[-1]
    short_signals = []
    price20ma = last['close'] - last['MA_20']
    short_signals.append(indicator_signal(price20ma))
    short_slope_7 = df['close'].diff().tail(7).mean()
    short_signals.append(indicator_signal(short_slope_7))
    short_signals.append(0)
    if last['MA_20'] > last['MA_50']:
        short_signals.append(1)
    else:
        short_signals.append(-1)
    short_signals.append(0)

    medium_signals = []
    price50ma = last['close'] - last['MA_50']
    medium_signals.append(indicator_signal(price50ma))
    if last['MA_20'] > last['MA_100']:
        medium_signals.append(1)
    else:
        medium_signals.append(-1)
    medium_signals.append(0)
    medium_signals.append(0)

    long_signals = []
    price100ma = last['close'] - last['MA_100']
    long_signals.append(indicator_signal(price100ma))
    if last['MA_50'] > last['MA_100']:
        long_signals.append(1)
    else:
        long_signals.append(-1)
    price200ma = last['close'] - last['MA_200']
    long_signals.append(indicator_signal(price200ma))

    trend_seeker_signal = mock_trend_seeker(df)

    combined = short_signals + medium_signals + long_signals + [trend_seeker_signal]
    overall = sum(combined) / len(combined)

    raw_percent = overall * 100
    factored_percent = abs(raw_percent) * 1.04
    possible_vals = [8,16,24,32,40,48,56,64,72,80,88,96,100]

    def nearest_opinion_pct(x):
        if x >= 95:
            return 100
        return min(possible_vals, key=lambda v: abs(v - x))

    final_opinion_pct = nearest_opinion_pct(factored_percent)
    sign = np.sign(overall)
    if sign > 0:
        final_opinion_str = f"{final_opinion_pct}% Buy"
    elif sign < 0:
        final_opinion_str = f"{final_opinion_pct}% Sell"
    else:
        final_opinion_str = "Hold"

    return {
        'short_avg': sum(short_signals)/len(short_signals),
        'medium_avg': sum(medium_signals)/len(medium_signals),
        'long_avg': sum(long_signals)/len(long_signals),
        'trend_seeker_signal': trend_seeker_signal,
        'overall_numeric': overall,
        'overall_opinion': final_opinion_str
    }

def approximate_price_for_ma_cross(df, short_ma_days, long_ma_days):
    if len(df) < max(short_ma_days, long_ma_days):
        return None
    closes = df['close'].values
    short_old_vals = closes[-short_ma_days:]
    long_old_vals  = closes[-long_ma_days:]
    short_oldest = short_old_vals[0]
    long_oldest  = long_old_vals[0]
    sum_short = short_old_vals.sum() - short_oldest
    sum_long  = long_old_vals.sum()  - long_oldest
    n_short = short_ma_days
    n_long  = long_ma_days
    numerator = sum_long*n_short - sum_short*n_long
    denominator = (n_long - n_short)
    if denominator == 0:
        return None
    P = numerator / denominator
    if P < 0:
        return None
    return round(P, 2)

# ---------------------------------------
# REMOVED "approximate_price_for_rsi" ENTIRELY
# ---------------------------------------

def approximate_price_for_stoch(df, period, target_stoch=80, k_smooth=3, d_smooth=3):
    if len(df) < period:
        return None
    sub = df.tail(period).copy()
    lo, hi = 0, 2*df['close'].iloc[-1]
    for _ in range(30):
        mid = (lo + hi)/2
        test_bar = pd.DataFrame([{'high': mid, 'low': mid, 'close': mid}])
        test_df = pd.concat([sub, test_bar], ignore_index=True)
        raw_stoch, _, _ = compute_stochastics(test_df, period=period, k_smooth=k_smooth, d_smooth=d_smooth)
        new_val = raw_stoch.iloc[-1]
        if pd.isna(new_val):
            return None
        if abs(new_val - target_stoch) < 0.1:
            return round(mid, 2)
        if new_val > target_stoch:
            hi = mid
        else:
            lo = mid
    return None

def compute_fibonacci_price(low, high, ratio):
    return round(high - ratio*(high - low), 2)

def build_expanded_cheatsheet(df):
    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2] if len(df) > 1 else None
    the_high = df['high'].iloc[-1]
    the_low  = df['low'].iloc[-1]

    len_52week = 252 if len(df) >= 252 else len(df)
    len_13week = 65  if len(df) >= 65  else len(df)
    len_1month = 22  if len(df) >= 22 else len(df)

    df_52 = df.tail(len_52week)
    df_13 = df.tail(len_13week)
    df_1m = df.tail(len_1month)

    high_52 = df_52['high'].max()
    low_52  = df_52['low'].min()
    high_13 = df_13['high'].max()
    low_13  = df_13['low'].min()
    high_1m = df_1m['high'].max()
    low_1m  = df_1m['low'].min()

    pivot_pp, pivot_r1, pivot_r2, pivot_r3, pivot_s1, pivot_s2, pivot_s3 = compute_pivots_s_r(
        the_high, the_low, last_close
    )

    # Use the new 20-bar rolling approach
    stdev_dict, avg_5day = compute_std_devs(df['close'], [1,2,3])
    p1_res = avg_5day + stdev_dict[1]
    p1_sup = avg_5day - stdev_dict[1]
    p2_res = avg_5day + stdev_dict[2]
    p2_sup = avg_5day - stdev_dict[2]
    p3_res = avg_5day + stdev_dict[3]
    p3_sup = avg_5day - stdev_dict[3]

    # ---------------------------------------
    # REMOVED THE "approximate_price_for_rsi" CALLS AND ROWS
    # ---------------------------------------

    cross_9_18 = approximate_price_for_ma_cross(df, 9, 18)
    cross_9_40 = approximate_price_for_ma_cross(df, 9, 40)
    cross_18_40= approximate_price_for_ma_cross(df, 18, 40)

    stoch_80 = approximate_price_for_stoch(df, 14, 80)
    stoch_70 = approximate_price_for_stoch(df, 14, 70)
    stoch_50 = approximate_price_for_stoch(df, 14, 50)
    stoch_30 = approximate_price_for_stoch(df, 14, 30)
    stoch_20 = approximate_price_for_stoch(df, 14, 20)

    len_4week = 20 if len(df) >= 20 else len(df)
    df_4w = df.tail(len_4week)
    high_4w = df_4w['high'].max()
    low_4w  = df_4w['low'].min()

    fib_4w_382 = compute_fibonacci_price(low_4w, high_4w, 0.382)
    fib_4w_50  = compute_fibonacci_price(low_4w, high_4w, 0.5)
    fib_4w_618 = compute_fibonacci_price(low_4w, high_4w, 0.618)

    fib_13w_382 = compute_fibonacci_price(low_13, high_13, 0.382)
    fib_13w_50  = compute_fibonacci_price(low_13, high_13, 0.5)
    fib_13w_618 = compute_fibonacci_price(low_13, high_13, 0.618)

    cheat_sheet = []

    def add_row(price, desc):
        if price is None:
            cheat_sheet.append({"price": "N/A", "description": desc})
        else:
            cheat_sheet.append({"price": round(price,3), "description": desc})

    # MA crosses
    add_row(cross_18_40, "Price Crosses 18-40 Day Moving Average")
    add_row(cross_9_40,  "Price Crosses 9-40 Day Moving Average")
    add_row(cross_9_18,  "Price Crosses 9-18 Day Moving Average")

    # (We keep the real/current RSI row only)
    real_rsi_14 = df["rsi_14"].iloc[-1]  # from DataFrame
    add_row(real_rsi_14, "14 Day RSI (Current)")

    add_row(high_52,  "52-Week High")
    add_row(high_13,  "13-Week High")
    add_row(high_1m,  "1-Month High")

    add_row(stoch_80, "14-3 Day Raw Stochastic at 80%")
    add_row(fib_4w_382, "38.2% Retracement From 4 Week High")
    add_row(fib_13w_382,"38.2% Retracement From 13 Week High")
    add_row(stoch_70, "14-3 Day Raw Stochastic at 70%")

    add_row(fib_4w_50,  "50% Retracement From 4 Week High/Low")
    add_row(fib_13w_50, "50% Retracement From 13 Week High/Low")
    add_row(pivot_r3,   "Pivot Point 3rd Level Resistance")
    add_row(p3_res,     "Price 3 Std Deviations Resistance")
    add_row(pivot_r2,   "Pivot Point 2nd Level Resistance")
    add_row(p2_res,     "Price 2 Std Deviations Resistance")
    add_row(stoch_50,   "14-3 Day Raw Stochastic at 50%")
    add_row(p1_res,     "Price 1 Std Deviation Resistance")
    add_row(pivot_r1,   "Pivot Point 1st Resistance Point")
    add_row(the_high,   "High")
    add_row(prev_close, "Previous Close")
    add_row(df['close'].iloc[-1], "Last")
    add_row(stoch_30,   "14-3 Day Raw Stochastic at 30%")
    add_row(pivot_pp,   "Pivot Point")
    add_row(the_low,    "Low")
    add_row(p1_sup,     "Price 1 Std Deviation Support")
    add_row(pivot_s1,   "Pivot Point 1st Support Point")
    add_row(stoch_20,   "14-3 Day Raw Stochastic at 20%")
    add_row(p2_sup,     "Price 2 Std Deviations Support")
    add_row(p3_sup,     "Price 3 Std Deviations Support")
    add_row(pivot_s2,   "Pivot Point 2nd Support Point")
    add_row(low_1m,     "1-Month Low")
    add_row(low_13,     "13-Week Low")
    add_row(pivot_s3,   "Pivot Point 3rd Support Point")
    add_row(df["rsi_14"].iloc[-1], "14 Day RSI at 30%")  # Example if you want it
    add_row(low_52,     "52-Week Low")
    # or remove the above line if you don't want a second RSI row.

    cheat_sheet_sorted = sorted(
        cheat_sheet,
        key=lambda x: float(x['price']) if x['price'] != "N/A" else -999999999,
        reverse=True
    )
    return cheat_sheet_sorted

###############################################################################
# S/R CLASSIFIER
###############################################################################
def classify_sr(description):
    desc_lower = description.lower()
    if ("resistance" in desc_lower or "high" in desc_lower or
        "r1" in desc_lower or "r2" in desc_lower or "r3" in desc_lower):
        return "Resistance"
    elif ("support" in desc_lower or "low" in desc_lower or
          "s1" in desc_lower or "s2" in desc_lower or "s3" in desc_lower):
        return "Support"
    else:
        return ""

###############################################################################
# IBAPI APPLICATION
###############################################################################
class IBApp(EWrapper, EClient):
    def __init__(self, ipaddress, portid, clientid):
        EClient.__init__(self, self)
        self.ipaddress = ipaddress
        self.portid = portid
        self.clientid = clientid
        self.historical_data = []
        self.request_completed = False
        self.final_technical_data = None

    def connect_and_run(self):
        self.connect(self.ipaddress, self.portid, self.clientid)
        thread = threading.Thread(target=self.run)
        thread.start()

    @iswrapper
    def nextValidId(self, orderId: int):
        print(f"[IBApp] nextValidId called with orderId={orderId}")
        self.request_historical_data()

    def request_historical_data(self):
        print("[IBApp] Requesting historical data for Contract (1 year, daily bars)...")
        contract = self.create_mes_contract()
        self.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime="",
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

    def create_mes_contract(self):
        contract = Contract()
        contract.symbol = "MHNG"
        contract.secType = "FUT"
        contract.exchange = "NYMEX"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = "20250425"
        contract.localSymbol = "MNGK5"
        contract.multiplier = "1000"
        return contract

    @iswrapper
    def historicalData(self, reqId, bar):
        parsed_dt = parse_ib_date(str(bar.date))
        self.historical_data.append({
            'date': bar.date,
            'datetime_obj': parsed_dt,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    @iswrapper
    def historicalDataEnd(self, reqId, start, end):
        print("[IBApp] Historical data download complete.")
        self.request_completed = True
        self.process_data()
        self.disconnect()

    def print_historical_data_info(self):
        if not self.historical_data:
            print("No historical data available.")
            return
        first_bar = self.historical_data[0]
        last_bar  = self.historical_data[-1]
        first_dt = first_bar['datetime_obj']
        last_dt  = last_bar['datetime_obj']

        if isinstance(first_dt, datetime):
            start_str = first_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_str = str(first_dt)
        if isinstance(last_dt, datetime):
            end_str = last_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            end_str = str(last_dt)

        print("\n==============================================")
        print("Historical Data Set Information")
        print("==============================================")
        print(f"Symbol             : {self.create_mes_contract().symbol}")
        print(f"Data Duration      : 1 Year")
        print(f"Bar Size           : 1 Day")
        print(f"Date Range         : {start_str} to {end_str}")
        print(f"Total Bars Received: {len(self.historical_data)}")
        print("==============================================\n")

    def print_last_bar_timestamp(self, df):
        if df.empty:
            print("DataFrame is empty. No last bar to show.")
            return
        last_bar = df.iloc[-1]
        dt_field = last_bar.get('datetime_obj', None)
        print("\n================== LAST BAR TIMESTAMP ==================")
        if dt_field is not None and isinstance(dt_field, datetime):
            print(f"The last bar in the data set is from {dt_field.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            raw_date = last_bar['date']
            print(f"The last bar in the data set has a date: {raw_date}")
        print("========================================================\n")

    def print_analysis_validity(self, validity_days=1):
        analysis_timestamp = datetime.now()
        valid_until = analysis_timestamp + timedelta(days=validity_days)
        print("\n==============================================")
        print("Analysis Validity Information")
        print("==============================================")
        print(f"Analysis Timestamp     : {analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Prediction Validity    : {validity_days} day(s)")
        print(f"Prediction Valid Until : {valid_until.strftime('%Y-%m-%d %H:%M:%S')}")
        print("==============================================\n")

    def process_data(self):
        try:
            df = pd.DataFrame(self.historical_data)
            print(f"DEBUG: Received {len(df)} bars from IB.")
            if df.empty:
                print("No data returned. Can't set final technical data.")
                return

            df.sort_values(by="date", inplace=True)
            df.reset_index(drop=True, inplace=True)

            self.print_historical_data_info()
            self.print_analysis_validity(validity_days=1)
            self.print_last_bar_timestamp(df)

            # Calculate indicators
            df["MA_5"]   = compute_moving_average(df['close'], 5)
            df["MA_20"]  = compute_moving_average(df['close'], 20)
            df["MA_50"]  = compute_moving_average(df['close'], 50)
            df["MA_100"] = compute_moving_average(df['close'], 100)
            df["MA_200"] = compute_moving_average(df['close'], 200)

            df["pc_5"]   = compute_price_change(df['close'], 5)
            df["pc_20"]  = compute_price_change(df['close'], 20)
            df["pc_50"]  = compute_price_change(df['close'], 50)
            df["pc_100"] = compute_price_change(df['close'], 100)
            df["pc_200"] = compute_price_change(df['close'], 200)

            df["pct_5"]   = compute_percent_change(df['close'], 5)
            df["pct_20"]  = compute_percent_change(df['close'], 20)
            df["pct_50"]  = compute_percent_change(df['close'], 50)
            df["pct_100"] = compute_percent_change(df['close'], 100)
            df["pct_200"] = compute_percent_change(df['close'], 200)

            df["vol_5"]   = compute_average_volume(df['volume'], 5)
            df["vol_20"]  = compute_average_volume(df['volume'], 20)
            df["vol_50"]  = compute_average_volume(df['volume'], 50)
            df["vol_100"] = compute_average_volume(df['volume'], 100)
            df["vol_200"] = compute_average_volume(df['volume'], 200)

            df["raw_9"],  df["k_9"],  df["d_9"]  = compute_stochastics(df, 9)
            df["raw_14"], df["k_14"], df["d_14"] = compute_stochastics(df, 14)
            df["raw_20"], df["k_20"], df["d_20"] = compute_stochastics(df, 20)

            df["atr_14"] = compute_atr(df, 14)
            df["rsi_9"]  = compute_rsi(df['close'], 9)
            df["rsi_14"] = compute_rsi(df['close'], 14)
            df["rsi_20"] = compute_rsi(df['close'], 20)
            df["pr_9"]   = compute_percent_r(df, 9)
            df["pr_14"]  = compute_percent_r(df, 14)
            df["pr_20"]  = compute_percent_r(df, 20)
            df["hv_20"]  = compute_historic_volatility(df['close'], 20)
            df["macd"]   = compute_macd(df['close'])

            def safe_round(val, decimals=3):
                return round(val, decimals) if pd.notnull(val) else None

            # Snapshot
            last = df.iloc[-1]

            print("\n==================== TECHNICAL ANALYSIS SNAPSHOT ====================\n")

            # 1) Print a table for Period / Moving Average / Price Change / ...
            ma_headers = ["Period", "Moving Average", "Price Change", "Percent Change", "Avg Volume"]
            ma_data = []
            ma_periods = [
                ("5-Day",   'MA_5',   'pc_5',   'pct_5',   'vol_5'),
                ("20-Day",  'MA_20',  'pc_20',  'pct_20',  'vol_20'),
                ("50-Day",  'MA_50',  'pc_50',  'pct_50',  'vol_50'),
                ("100-Day", 'MA_100', 'pc_100', 'pct_100', 'vol_100'),
                ("200-Day", 'MA_200', 'pc_200', 'pct_200', 'vol_200'),
            ]
            for label, ma_col, pc_col, pct_col, vol_col in ma_periods:
                ma_data.append([
                    label,
                    safe_round(last[ma_col],3),
                    safe_round(last[pc_col],3),
                    f"{safe_round(last[pct_col],2)}%",
                    int(safe_round(last[vol_col],0) or 0)
                ])
            print(tabulate(ma_data, headers=ma_headers, tablefmt="pretty"))

            # 2) Stochastics table
            stoch_headers = ["Period", "Raw Stochastic", "Stoch %K", "Stoch %D", "ATR"]
            stoch_data = []
            stoch_periods = [
                ("9-Day",  'raw_9','k_9','d_9'),
                ("14-Day", 'raw_14','k_14','d_14'),
                ("20-Day", 'raw_20','k_20','d_20'),
            ]
            for label, rcol, kcol, dcol in stoch_periods:
                stoch_data.append([
                    label,
                    f"{safe_round(last[rcol],2)}%",
                    f"{safe_round(last[kcol],2)}%",
                    f"{safe_round(last[dcol],2)}%",
                    safe_round(last["atr_14"],2)
                ])
            print("\n-- Stochastics --")
            print(tabulate(stoch_data, headers=stoch_headers, tablefmt="pretty"))

            # 3) RSI/PercentR/HV/MACD table
            rsi_headers = ["Period","Relative Strength","Percent R","Historic Vol","MACD Osc"]
            rsi_data = []
            rsi_periods = [
                ("9-Day",  'rsi_9', 'pr_9', 'hv_20'),
                ("14-Day", 'rsi_14','pr_14','hv_20'),
                ("20-Day", 'rsi_20','pr_20','hv_20'),
            ]
            for label, rsi_col, pr_col, hv_col in rsi_periods:
                rsi_data.append([
                    label,
                    f"{safe_round(last[rsi_col],2)}%",
                    f"{safe_round(last[pr_col],2)}%",
                    f"{safe_round(last[hv_col],2)}%",
                    safe_round(last['macd'],2)
                ])
            print("\n-- RSI / %R / Vol / MACD --")
            print(tabulate(rsi_data, headers=rsi_headers, tablefmt="pretty"))

            # 4) Pivot Points & SD (using the new rolling-20 approach behind the scenes)
            pivot_pp, pivot_r1, pivot_r2, pivot_r3, pivot_s1, pivot_s2, pivot_s3 = compute_pivots_s_r(
                last['high'], last['low'], last['close']
            )
            stdev_dict, avg_5day = compute_std_devs(df['close'], [1,2,3])
            p1_res = avg_5day + stdev_dict[1]
            p1_sup = avg_5day - stdev_dict[1]
            p2_res = avg_5day + stdev_dict[2]
            p2_sup = avg_5day - stdev_dict[2]
            p3_res = avg_5day + stdev_dict[3]
            p3_sup = avg_5day - stdev_dict[3]

            pivot_headers = ["PP","R1","R2","R3","S1","S2","S3"]
            pivot_data = [[
                safe_round(pivot_pp,2),
                safe_round(pivot_r1,2),
                safe_round(pivot_r2,2),
                safe_round(pivot_r3,2),
                safe_round(pivot_s1,2),
                safe_round(pivot_s2,2),
                safe_round(pivot_s3,2)
            ]]

            print("\n-- Trader's Pivot Cheat Sheet --")
            print(tabulate(pivot_data, headers=pivot_headers, tablefmt="pretty"))

            print(f"\n5-Day Avg Price: {safe_round(avg_5day,2)}")
            print(f"1 SD Res: {safe_round(p1_res,2)} / Sup: {safe_round(p1_sup,2)}")
            print(f"2 SD Res: {safe_round(p2_res,2)} / Sup: {safe_round(p2_sup,2)}")
            print(f"3 SD Res: {safe_round(p3_res,2)} / Sup: {safe_round(p3_sup,2)}\n")

            # 5) Barchart Opinion
            opinion_results = barchart_opinion_logic(df)
            print("-- Barchart Opinion --")
            bc_data = [
                ["Short-Term Avg", opinion_results['short_avg']],
                ["Medium-Term Avg", opinion_results['medium_avg']],
                ["Long-Term Avg", opinion_results['long_avg']],
                ["Trend Seeker (mock)", opinion_results['trend_seeker_signal']],
                ["Overall Numeric Avg", opinion_results['overall_numeric']],
                ["Final Opinion", opinion_results['overall_opinion']],
            ]
            print(tabulate(bc_data, tablefmt="pretty"))

            # Final snapshot
            self.final_technical_data = {
                "last_close": last["close"],
                "ma_20": safe_round(last["MA_20"],2),
                "ma_50": safe_round(last["MA_50"],2),
                "rsi_14": safe_round(last["rsi_14"],2),
                "macd": safe_round(last["macd"],2),
                "atr_14": safe_round(last["atr_14"],2),
                "s1": safe_round(pivot_s1,2),
                "s2": safe_round(pivot_s2,2),
                "r1": safe_round(pivot_r1,2),
                "r2": safe_round(pivot_r2,2)
            }

            # 6) Extended cheat sheet (3-column style)
            cheat_sheet_rows = build_expanded_cheatsheet(df)
            three_col_headers = ["Support/Resistance Levels", "Price", "Key Turning Points"]
            three_col_data = []
            for row in cheat_sheet_rows:
                sr_type = classify_sr(row['description'])
                if sr_type in ["Support","Resistance"]:
                    left_col = row['description']
                    mid_col  = row['price']
                    right_col= ""
                else:
                    left_col = ""
                    mid_col  = row['price']
                    right_col= row['description']
                three_col_data.append([left_col, mid_col, right_col])

            print("\n================== EXTENDED CHEAT SHEET (3-COLUMN STYLE) ==================\n")
            print(tabulate(three_col_data, headers=three_col_headers, tablefmt="pretty"))
            print("\n============ END OF 3-COLUMN EXTENDED CHEAT SHEET (BARCHART STYLE) =========\n")

        except Exception as e:
            print("Error in process_data:", e)
            self.final_technical_data = None

###############################################################################
# OPENAI INTEGRATION
###############################################################################
def build_ai_prompt(technicals):
    prompt = f"""
    You are an advanced trading assistant analyzing the May 2025 Natural Gas Futures contract (NGK25).
    I have technical snapshot data including current price, change %, moving averages, stochastic, RSI, MACD,
    and a trader’s cheat sheet with pivot points, Fibonacci retracements, and standard deviation levels.

    Please provide:
    - A clean summary of the current market condition,
    - Analysis of short-term vs long-term trends,
    - Key support/resistance zones with interpretation,
    - Momentum indicators (Stochastic, RSI, MACD) and what they imply,
    - A trading plan or setup idea (both scalping and swing),
    - Any oversold/bounce potential based on the data.

    Make it actionable, as if advising a trader who must decide within the next 24 hours.

    ### Technical Data (via IBAPI):
    - Current NGK25 Price: {technicals['last_close']}
    - 20-day MA: {technicals['ma_20']}, 50-day MA: {technicals['ma_50']}
    - RSI (14-day): {technicals['rsi_14']}
    - MACD: {technicals['macd']}
    - ATR (14): {technicals['atr_14']}
    - Support Levels: {technicals['s1']}, {technicals['s2']}
    - Resistance Levels: {technicals['r1']}, {technicals['r2']}

    ### TASK:
    1. Provide a trade recommendation (Buy/Sell/Hold). Provide entry, exit and stop loss please. Identify Support & Resistance Levels.
    2. Estimate the probability (0–100%) that NGK25 will move in the recommended direction over the next 24 Hours.
    3. Research storage, weather, supply, demand, OPEC news and give a brief fundamental justification for price direction.
    4. Give a brief technical justification (RSI, MACD, S/R, etc.).
    5. Mention one macro risk that might invalidate this trade.

    Rules:
    - Probability must be a single integer from 40–80%.
    - Keep your answer under 200 words total.
    """
    return prompt

def get_ai_analysis(prompt):
    response = openai.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

###############################################################################
# MAIN
###############################################################################
def main():
    app = IBApp("127.0.0.1", 7496, clientid=1)
    app.connect_and_run()

    timeout = 60
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if app.request_completed:
            break
        time.sleep(1)

    if not app.request_completed:
        print("[Main] Timed out waiting for historical data.")
        return

    technicals = app.final_technical_data
    if not technicals:
        print("[Main] No final technical data found. Exiting.")
        return

    prompt = build_ai_prompt(technicals)
    ai_decision = get_ai_analysis(prompt)
    print("AI Analysis:\n", ai_decision)

if __name__ == "__main__":
    main()
