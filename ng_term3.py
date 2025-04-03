#!/usr/bin/env python3
"""
ng_daily_terminal.py

Purpose:
  - Pull free EIA data for natural gas supply or storage.
  - Connect to IBAPI for live NG price.
  - Estimate daily net supply/demand.
  - Apply a simple price-sensitivity matrix.
  - Print out a 24-hour bullish/bearish "bias."

Author: YourName
"""

import os
import time
import requests
import math

# --- Add dotenv to manage .env variables ---
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
        """
        Called continuously once we request market data.
        We filter for 'Last Price' or 'Delayed Last Price' from IB.
        """
        # According to IB docs: tickType 4 = LAST, 68 = DELAYED_LAST
        if tickType in (4, 68) and price > 0:
            self.lastPrice = price
            # Mark that we received at least one update
            self.ready = True

    def error(self, reqId, errorCode, errorString):
        """
        Handle errors from the TWS.
        """
        print(f"[IB ERROR] reqId={reqId} code={errorCode} msg={errorString}")


class IBApiClient(EClient):
    def __init__(self, wrapper):
        super().__init__(wrapper)


# -------------------------------------------------------------------
# 1) EIA Data Fetch
# -------------------------------------------------------------------
def get_eia_dry_gas_production(eia_api_key):
    """
    Example: EIA's 'Dry Shale Gas Production' series or 'Dry Natural Gas Production' 
    from monthly or weekly stats.
    
    Series used (monthly for entire US):
      NG.N9050US2.M = Marketed production
    
    Adjust if you find a more up-to-date or daily series that suits your needs.
    """
    series_id = "NG.N9050US2.M"
    url = f"https://api.eia.gov/series/?api_key={eia_api_key}&series_id={series_id}"

    try:
        resp = requests.get(url, timeout=10)

        # Check HTTP status:
        if resp.status_code != 200:
            print(f"[EIA] Request error: {resp.status_code} - {resp.text}")
            return None

        raw_data = resp.json()
        # Check if 'series' is present:
        if "series" not in raw_data:
            print(f"[EIA] 'series' key not found in response: {raw_data}")
            return None

        # Extract the actual data array
        data_list = raw_data["series"]
        if not data_list:
            print("[EIA] 'series' is empty.")
            return None

        # The first element should have a 'data' array
        series_data = data_list[0].get("data", [])
        if not series_data:
            print("[EIA] No 'data' found in the series.")
            return None

        # The most recent monthly value is the first item
        # e.g. ['202305', 28110.5]
        latest_value = series_data[0][1]  # Bcf for the month
        # Convert monthly Bcf to daily (roughly):
        daily_est = latest_value / 30.0
        return daily_est

    except Exception as e:
        print(f"[EIA] Could not fetch production data: {e}")
        return None


def get_eia_demand_estimate(eia_api_key):
    """
    Placeholder for a daily demand estimate from EIA or other free data.
    We'll do a simplified read from total consumption (res/comm + industrial + power).
    
    Series:
      NG.N9140US2.M = Natural gas consumption, total (monthly)
    """
    series_id = "NG.N9140US2.M"
    url = f"https://api.eia.gov/series/?api_key={eia_api_key}&series_id={series_id}"

    try:
        resp = requests.get(url, timeout=10)

        # Check HTTP status:
        if resp.status_code != 200:
            print(f"[EIA] Request error: {resp.status_code} - {resp.text}")
            return None

        raw_data = resp.json()
        # Check if 'series' is present:
        if "series" not in raw_data:
            print(f"[EIA] 'series' key not found in response: {raw_data}")
            return None

        data_list = raw_data["series"]
        if not data_list:
            print("[EIA] 'series' is empty.")
            return None

        series_data = data_list[0].get("data", [])
        if not series_data:
            print("[EIA] No 'data' found in the series.")
            return None

        latest_value = series_data[0][1]  # Bcf for the month
        daily_est = latest_value / 30.0
        return daily_est

    except Exception as e:
        print(f"[EIA] Could not fetch consumption data: {e}")
        return None


# -------------------------------------------------------------------
# 2) Simple Price Sensitivity Logic (Matrix)
# -------------------------------------------------------------------
def price_sensitivity_matrix(net_bcfd):
    """
    net_bcfd: Production - Consumption (Bcf/day)
    
    Returns a dict with:
      - sentiment
      - expected_price_move (string/range)
    """
    nb = round(net_bcfd)

    if nb <= -10:
        return {
            'sentiment': '🚀 Very Bullish',
            'expected_price_move': '+$0.75 to +$1.50+'
        }
    elif -10 < nb <= -5:
        return {
            'sentiment': '📈 Bullish',
            'expected_price_move': '+$0.30 to +$0.70'
        }
    elif -5 < nb <= -2:
        return {
            'sentiment': '🟢 Slightly Bullish',
            'expected_price_move': '+$0.10 to +$0.30'
        }
    elif -2 < nb < 2:
        return {
            'sentiment': '⚖️ Neutral',
            'expected_price_move': '$0.00 ± $0.10'
        }
    elif 2 <= nb < 5:
        return {
            'sentiment': '🔻 Slightly Bearish',
            'expected_price_move': '-$0.10 to -$0.30'
        }
    elif 5 <= nb < 10:
        return {
            'sentiment': '📉 Bearish',
            'expected_price_move': '-$0.30 to -$0.75'
        }
    else:
        return {
            'sentiment': '🧨 Very Bearish',
            'expected_price_move': '-$0.75 to -$1.50+'
        }


# -------------------------------------------------------------------
# 3) Main Terminal App
# -------------------------------------------------------------------
class NGTerminalApp(IBApiWrapper, IBApiClient):
    """
    Combines IB wrapper + client.
    We'll do a simple scenario:
      - Connect to IB
      - Request market data for NG (Henry Hub) futures
      - Pull EIA data for supply/demand
      - Print a 24-hr forecast bias
    """
    def __init__(self, host='127.0.0.1', port=7497, clientId=123):
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
        # Typically we'd set 'localSymbol' or 'lastTradeDateOrContractMonth'
        # for the front-month. For simplicity, let's do:
        self.contract.lastTradeDateOrContractMonth = "202505"
        #   ^ e.g. May 2025 NG contract for demonstration
        #   Adjust to the correct front month in your real usage.

    def start_app(self):
        print("[INFO] Connecting to IB TWS...")
        self.connect(self.host, self.port, self.clientId)
        # Launch network loop in a dedicated thread
        self.run()

    def stop_app(self):
        print("[INFO] Disconnecting from IB TWS...")
        self.disconnect()


# -------------------------------------------------------------------
# 4) Main Execution
# -------------------------------------------------------------------
def main():
    # --- Load environment variables ---
    load_dotenv()

    # 1) Load EIA API Key
    eia_api_key = os.environ.get("EIA_API_KEY", "YOUR_EIA_API_KEY_HERE")
    if not eia_api_key or "YOUR_EIA_API_KEY_HERE" in eia_api_key:
        print("[ERROR] Please set your EIA_API_KEY in a .env file or as an environment variable.")
        return

    # 2) Fetch fundamental data from EIA
    dry_gas_prod = get_eia_dry_gas_production(eia_api_key)
    total_demand = get_eia_demand_estimate(eia_api_key)

    if (dry_gas_prod is None) or (total_demand is None):
        print("[ERROR] Could not retrieve fundamental data. Exiting.")
        return

    net_bcfd = dry_gas_prod - total_demand

    # 3) Evaluate Price Sensitivity
    sens = price_sensitivity_matrix(net_bcfd)

    print("\n=== NATURAL GAS FUNDAMENTALS ===")
    print(f"Estimated Production (Bcf/day): {dry_gas_prod:.2f}")
    print(f"Estimated Demand     (Bcf/day): {total_demand:.2f}")
    print(f"Net Supply (Bcf/day): {net_bcfd:.2f}")
    print(f"--> Matrix Sentiment: {sens['sentiment']}")
    print(f"--> Expected Price Move: {sens['expected_price_move']}")
    print("================================\n")

    # 4) Connect to IB to fetch real-time NG price
    app = NGTerminalApp(host='127.0.0.1', port=7497, clientId=123)
    app.start_app()

    # Request real-time market data
    reqId = 1
    app.reqMarketDataType(1)  # 1 = Live, 2 = Frozen, 3 = Delayed
    app.reqMktData(reqId, app.contract, "", False, False, [])

    # Wait a few seconds to get some ticks
    print("[INFO] Waiting for IB price updates...")
    start_time = time.time()
    while True:
        time.sleep(1)
        if app.ready or (time.time() - start_time) > 10:
            # Either we got a price or 10 seconds passed
            break

    if app.lastPrice:
        print(f"[IB] Current NG Price (approx): ${app.lastPrice:.2f}")
    else:
        print("[IB] No price received (check TWS settings and contract details).")

    # 5) Provide final 24-hour "forecast" snippet
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

    # 6) Disconnect
    app.stop_app()


if __name__ == "__main__":
    main()

