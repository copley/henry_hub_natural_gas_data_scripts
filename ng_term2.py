#!/usr/bin/env python3
"""
ng_daily_terminal.py

Purpose:
  - Pull EIA data for natural gas supply (Marketed Production) or consumption (Total).
  - Connect to IBAPI for live NG price.
  - Estimate daily net supply/demand.
  - Apply a simple price-sensitivity matrix.
  - Print out a 24-hour bullish/bearish "bias."

Uses EIA's v2 API structure, with retry logic + longer timeouts.

Author: YourName
"""

import os
import time
import requests
import math
from dotenv import load_dotenv

# --- IBAPI imports ---
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract


class IBApiWrapper(EWrapper):
    def __init__(self):
        super().__init__()
        self.lastPrice = None
        self.ready = False

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType in (4, 68) and price > 0:  # 4=LAST, 68=DELAYED_LAST
            self.lastPrice = price
            self.ready = True

    def error(self, reqId, errorCode, errorString):
        print(f"[IB ERROR] reqId={reqId} code={errorCode} msg={errorString}")


class IBApiClient(EClient):
    def __init__(self, wrapper):
        super().__init__(wrapper)


def fetch_eia_data_with_retry(url, max_retries=3, backoff=5):
    """
    Generic helper to fetch EIA API data with retries and longer timeout.
    :param url: EIA v2 endpoint
    :param max_retries: number of attempts before giving up
    :param backoff: seconds to wait between retries (increases each attempt)
    :return: parsed JSON dict or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            # Increase timeout to 60s for slow responses
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
    # If we exhaust all retries
    return None


def get_eia_dry_gas_production(eia_api_key):
    """
    Fetch monthly natural gas marketed production data from EIA's v2 API.
    Convert from MMCF to Bcf/day.
    """
    series_id = "NG.N9050US2.M"
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={eia_api_key}"

    raw_data = fetch_eia_data_with_retry(url)
    if not raw_data:
        print("[EIA] Could not fetch production data (all retries failed).")
        return None

    # Parse new v2 structure
    if "response" not in raw_data or "data" not in raw_data["response"]:
        print(f"[EIA] 'response' or 'data' key missing: {raw_data}")
        return None

    data_list = raw_data["response"]["data"]
    if not data_list:
        print("[EIA] 'data' array is empty.")
        return None

    latest_record = data_list[0]
    mmcf_month = latest_record["value"]  # e.g. 3525720 (MMCF)
    # Convert MMCF -> Bcf: mmcf_month / 1000
    bcf_month = mmcf_month / 1000.0
    # Then Bcf/day (approx)
    daily_bcf = bcf_month / 30.0
    return daily_bcf


def get_eia_demand_estimate(eia_api_key):
    """
    Fetch monthly total natural gas consumption from EIA's v2 API.
    Convert from MMCF to Bcf/day.
    """
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

        # Henry Hub NG (NYMEX) front-month contract example:
        self.contract = Contract()
        self.contract.symbol = "NG"
        self.contract.secType = "FUT"
        self.contract.exchange = "NYMEX"
        self.contract.currency = "USD"
        self.contract.lastTradeDateOrContractMonth = "202505"

    def start_app(self):
        print("[INFO] Connecting to IB TWS...")
        self.connect(self.host, self.port, self.clientId)
        self.run()

    def stop_app(self):
        print("[INFO] Disconnecting from IB TWS...")
        self.disconnect()


def main():
    load_dotenv()
    eia_api_key = os.environ.get("EIA_API_KEY", "YOUR_EIA_API_KEY_HERE")

    if not eia_api_key or "YOUR_EIA_API_KEY_HERE" in eia_api_key:
        print("[ERROR] Please set your EIA_API_KEY in a .env file or as an environment variable.")
        return

    dry_gas_prod = get_eia_dry_gas_production(eia_api_key)
    total_demand = get_eia_demand_estimate(eia_api_key)

    if (dry_gas_prod is None) or (total_demand is None):
        print("[ERROR] Could not retrieve fundamental data. Exiting.")
        return

    net_bcfd = dry_gas_prod - total_demand
    sens = price_sensitivity_matrix(net_bcfd)

    print("\n=== NATURAL GAS FUNDAMENTALS ===")
    print(f"Estimated Production (Bcf/day): {dry_gas_prod:.2f}")
    print(f"Estimated Demand     (Bcf/day): {total_demand:.2f}")
    print(f"Net Supply (Bcf/day): {net_bcfd:.2f}")
    print(f"--> Matrix Sentiment: {sens['sentiment']}")
    print(f"--> Expected Price Move: {sens['expected_price_move']}")
    print("================================\n")

    # Connect to IB to fetch real-time NG price
    app = NGTerminalApp(host='127.0.0.1', port=7496, clientId=72)
    app.start_app()

    reqId = 1
    app.reqMarketDataType(1)  # Live market data
    app.reqMktData(reqId, app.contract, "", False, False, [])

    print("[INFO] Waiting for IB price updates...")
    start_time = time.time()
    while True:
        time.sleep(1)
        if app.ready or (time.time() - start_time) > 10:
            break

    if app.lastPrice:
        print(f"[IB] Current NG Price (approx): ${app.lastPrice:.2f}")
    else:
        print("[IB] No price received (check TWS settings and contract details).")

    # Simple final forecast snippet
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

    print("\n=== 24-HOUR FORECAST ===")
    print(forecast_text)
    print("================================\n")

    app.stop_app()


if __name__ == "__main__":
    main()

