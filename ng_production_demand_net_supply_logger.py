#!/usr/bin/env python3
"""
ng_fundamentals_logger.py

Purpose:
  - Pull EIA data for natural gas supply (Marketed Production) or consumption (Total).
  - Connect to IB TWS for a real-time Henry Hub futures price.
  - Estimate net supply/demand and classify the market as bullish/bearish.
  - Print final results, log them with timestamps, and show a 24-hour forecast window.
"""

import os
import time
import requests
import math
import threading
from dotenv import load_dotenv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # For New Zealand timezone

# IB API imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBApiWrapper(EWrapper):
    def __init__(self):
        super().__init__()
        self.lastPrice = None
        self.ready = False

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType in (4, 68) and price > 0:
            self.lastPrice = price
            self.ready = True

    def error(self, reqId, errorCode, errorString):
        print(f"[IB ERROR] reqId={reqId} code={errorCode} msg={errorString}")

class IBApiClient(EClient):
    def __init__(self, wrapper):
        super().__init__(wrapper)

def fetch_eia_data_with_retry(url, max_retries=3, backoff=5):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"[EIA] Request error: {resp.status_code} - {resp.text}")
                return None
            return resp.json()
        except Exception as e:
            print(f"[EIA] Attempt {attempt + 1} of {max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = backoff * (attempt + 1)
                print(f"[EIA] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    return None

def get_eia_dry_gas_production(eia_api_key):
    series_id = "NG.N9050US2.M"
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={eia_api_key}"
    raw_data = fetch_eia_data_with_retry(url)
    if not raw_data:
        print("[EIA] Could not fetch production data (all retries failed).")
        return None

    if "response" not in raw_data or "data" not in raw_data["response"]:
        print(f"[EIA] 'response' or 'data' key missing: {raw_data}")
        return None

    data_list = raw_data["response"]["data"]
    if not data_list:
        print("[EIA] 'data' array is empty.")
        return None

    latest_record = data_list[0]
    mmcf_month = latest_record["value"]
    bcf_month = mmcf_month / 1000.0
    daily_bcf = bcf_month / 30.0
    return daily_bcf

def get_eia_demand_estimate(eia_api_key):
    series_id = "NG.N9140US2.M"
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={eia_api_key}"
    raw_data = fetch_eia_data_with_retry(url)
    if not raw_data:
        print("[EIA] Could not fetch consumption data (all retries failed).")
        return None

    if "response" not in raw_data or "data" not in raw_data["response"]:
        print(f"[EIA] 'response' or 'data' key missing: {raw_data}")
        return None

    data_list = raw_data["response"]["data"]
    if not data_list:
        print("[EIA] 'data' array is empty.")
        return None

    latest_record = data_list[0]
    mmcf_month = latest_record["value"]
    bcf_month = mmcf_month / 1000.0
    daily_bcf = bcf_month / 30.0
    return daily_bcf

def price_sensitivity_matrix(net_bcfd):
    nb = round(net_bcfd)
    if nb <= -10:
        return {'sentiment': '🚀 Very Bullish', 'expected_price_move': '+$0.75 to +$1.50+'}
    elif -10 < nb <= -5:
        return {'sentiment': '📈 Bullish', 'expected_price_move': '+$0.30 to +$0.70'}
    elif -5 < nb <= -2:
        return {'sentiment': '🟢 Slightly Bullish', 'expected_price_move': '+$0.10 to +$0.30'}
    elif -2 < nb < 2:
        return {'sentiment': '⚖️ Neutral', 'expected_price_move': '$0.00 ± $0.10'}
    elif 2 <= nb < 5:
        return {'sentiment': '🔻 Slightly Bearish', 'expected_price_move': '-$0.10 to -$0.30'}
    elif 5 <= nb < 10:
        return {'sentiment': '📉 Bearish', 'expected_price_move': '-$0.30 to -$0.75'}
    else:
        return {'sentiment': '🧨 Very Bearish', 'expected_price_move': '-$0.75 to -$1.50+'}

class NGTerminalApp(IBApiWrapper, IBApiClient):
    def __init__(self, host='127.0.0.1', port=7496, clientId=3):
        IBApiWrapper.__init__(self)
        IBApiClient.__init__(self, wrapper=self)
        self.host = host
        self.port = port
        self.clientId = clientId

        self.contract = Contract()
        self.contract.symbol = "NG"
        self.contract.secType = "FUT"
        self.contract.exchange = "NYMEX"
        self.contract.currency = "USD"
        self.contract.lastTradeDateOrContractMonth = "202505"

    def start_app(self):
        print("[INFO] Connecting to IB TWS...")
        self.connect(self.host, self.port, self.clientId)
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def stop_app(self):
        print("[INFO] Disconnecting from IB TWS...")
        self.disconnect()

def main():
    load_dotenv()
    eia_api_key = os.environ.get("EIA_API_KEY", "YOUR_EIA_API_KEY_HERE")

    if not eia_api_key or "YOUR_EIA_API_KEY_HERE" in eia_api_key:
        print("[ERROR] Please set your EIA_API_KEY in a .env file or as an environment variable.")
        return

    log_lines = []

    # 1) Fetch Data
    dry_gas_prod = get_eia_dry_gas_production(eia_api_key)
    total_demand = get_eia_demand_estimate(eia_api_key)

    if (dry_gas_prod is None) or (total_demand is None):
        log_lines.append("[ERROR] Could not retrieve fundamental data. Exiting.")
        print("\n".join(log_lines))
        with open("ng_script.log", "a") as f:
            tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{tstamp} -- {log_lines[-1]}\n")
        return

    # 2) Net supply & matrix
    net_bcfd = dry_gas_prod - total_demand
    sens = price_sensitivity_matrix(net_bcfd)

    # 3) Fundamentals lines
    fundamentals_header = "\n=== NATURAL GAS FUNDAMENTALS ==="
    fundamentals = [
        f"Estimated Production (Bcf/day): {dry_gas_prod:.2f}",
        f"Estimated Demand     (Bcf/day): {total_demand:.2f}",
        f"Net Supply (Bcf/day): {net_bcfd:.2f}",
        f"--> Matrix Sentiment: {sens['sentiment']}",
        f"--> Expected Price Move: {sens['expected_price_move']}",
        "================================"
    ]
    log_lines.append(fundamentals_header)
    log_lines.extend(fundamentals)

    # 4) IB connection
    app = NGTerminalApp(host='127.0.0.1', port=7496, clientId=4)
    app.start_app()

    reqId = 1
    app.reqMarketDataType(1)
    app.reqMktData(reqId, app.contract, "", False, False, [])

    # 5) Wait for IB data
    log_lines.append("[INFO] Waiting for IB price updates...")
    start_time = time.time()
    while True:
        time.sleep(1)
        if app.ready or (time.time() - start_time) > 10:
            break

    from datetime import datetime

    # 6) Check price
    if app.lastPrice:
        now_utc = datetime.now(timezone.utc)
        now_nzt = now_utc.astimezone(ZoneInfo("Pacific/Auckland"))

        price_time_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        price_time_nzt = now_nzt.strftime("%Y-%m-%d %H:%M:%S NZT")

        ib_price_line = (
            f"[IB] Current NG Price (approx): ${app.lastPrice:.3f}\n"
            f"as at (UTC): {price_time_utc}\n"
            f"as at (NZT): {price_time_nzt}"
        )
    else:
        ib_price_line = "[IB] No price received (check TWS settings and contract details)."

    log_lines.append(ib_price_line)

    # 7) Final forecast snippet
    if "Bullish" in sens['sentiment']:
        forecast_text = (
            "Forecast: Potential upward price pressure in the next 24 hours.\n"
            "Reason: Net supply shortfall (under demand). Watch out for weather, LNG flows."
        )
    elif "Bearish" in sens['sentiment']:
        forecast_text = (
            "Forecast: Potential downward price pressure in the next 24 hours.\n"
            "Reason: Net oversupply. Keep an eye on any sudden demand changes or production hiccups."
        )
    else:
        forecast_text = (
            "Forecast: Relatively balanced market. Prices may remain range-bound.\n"
            "Reason: Supply ~ Demand. Major moves would require external shocks (weather, LNG, news)."
        )
    forecast_header = "\n=== 24-HOUR FORECAST ==="
    log_lines.append(forecast_header)
    log_lines.append(forecast_text)

    # 8) Insert date/time stamp for results & mention the 24-hour validity
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"(Timestamp: {now_str}) Forecast valid for the next 24 hours.")
    log_lines.append("================================")

    # 9) Stop IB
    app.stop_app()
    time.sleep(1)

    # 10) Print all final results
    final_output = "\n".join(log_lines)
    print(final_output)

    # Log to file with timestamp
    with open("ng_script.log", "a") as f:
        run_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{run_stamp}] Script Run\n")
        f.write(final_output + "\n")


if __name__ == "__main__":
    main()

