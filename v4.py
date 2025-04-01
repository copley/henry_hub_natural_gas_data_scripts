import os
import logging
import requests
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout

# IB API imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ----------------------------
# 1. Configuration & Constants
# ----------------------------
# Summary of Script Issues and Suggestions
#
# The script runs successfully with no merging errors.
# However, the correlation matrix contains NaN values in the scenario matrix,
# and you're encountering timeout warnings when fetching NOAA data.
#
# 1) NOAA API Timeouts & Retries:
# - You see warnings like:
#   [WARNING] Request timeout on attempt 1/3 for URL: https://www.ncdc.noaa.gov/cdo-web/api/v2/data
#   [INFO] Retrying in 1 seconds...
# - These occur when NOAA API calls exceed your timeout limit (currently timeout=10 or timeout=20).
#
# Handling NOAA API Timeouts:
# - Increase the timeout in safe_request (e.g., timeout=30 or higher).
#   Example:
#     response = safe_request(endpoint, params=params, timeout=30, max_retries=3)
# - Reduce chunk size (fetch data in smaller increments, e.g., 90 or 180 days).
# - Handle partial data gracefully to avoid skipping large chunks of data.
#
# 2) NaN Values in Scenario Matrix:
# - NaNs occur because estimate_price_move returns None when:
#   a) There is no standard deviation in HDD data (e.g., consistently warm regions with zero HDD).
#   b) There is insufficient overlap after merging NOAA and IB data.
# - To debug, print valid row counts:
#     print("df_merged shape:", df_merged.shape)
#     for region in REGIONS:
#         col = f"{region}_Change"
#         if col in df_merged.columns:
#             print(region, "non-null count:", df_merged[col].count())
#
# 3) Suggested Fixes and Improvements:
#   a) Expand Historical Range:
#      Ensure full 2-year coverage from NOAA to match IB data.
#
#   b) Filter Out Warm Months or Regions:
#      Consider analyzing only colder months (fall to spring) when HDD varies.
#
#   c) Exclude Regions with Zero Variation:
#      Skip regions where standard deviation of HDD changes is negligible:
#      def scenario_matrix(df, regions, hdd_changes):
#          scenario_results = []
#          for region_col in regions:
#              if df[region_col].std() < 0.001:
#                  continue
#              slope = estimate_price_move(df, region_col)
#              if slope is None:
#                  continue
#              ...
#
# 4) Final Thoughts:
# - Timeouts indicate NOAA API latency; adjust timeouts/chunk size accordingly.
# - NaN values are common if HDD variation or data overlap is minimal.
# - Implementing these fixes will reduce NaNs and improve correlation quality.

load_dotenv()
NOAA_API_TOKEN = os.getenv('NOAA_API_TOKEN')

if not NOAA_API_TOKEN:
    raise ValueError("NOAA_API_TOKEN is not set in your environment variables.")

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
HISTORICAL_YEARS = 2  # how many years of data to fetch from NOAA

# -----------------------------
# 1b. Safe Request Helper
# -----------------------------

def safe_request(
    url: str,
    headers=None,
    params=None,
    max_retries=3,
    timeout=20,
    backoff_factor=2
):
    """
    A helper function that wraps `requests.get` in a try/except with:
      - timeouts,
      - a simple retry mechanism with exponential backoff,
      - returning None if all retries fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()  # raise for HTTPError (4xx or 5xx)
            return resp
        except Timeout:
            logging.warning(f"Request timeout on attempt {attempt}/{max_retries} for URL: {url}")
        except RequestException as e:
            logging.warning(f"Request exception on attempt {attempt}/{max_retries}: {e}")

        # If not successful, wait before the next attempt (exponential backoff)
        sleep_time = (backoff_factor ** (attempt - 1))
        logging.info(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)

    logging.error(f"All {max_retries} retries failed for {url}.")
    return None

# -----------------------------
# 2. Data Retrieval Functions
# -----------------------------
#
# Note: We now have two NOAA-related functions:
#  - fetch_noaa_historical_data() [existing]  => returns region avg HDD
#  - fetch_noaa_historical_daily_data() [new] => returns daily HDD time series

def fetch_forecast_temperatures(lat: float, lon: float, days: int = FORECAST_DAYS):
    """
    Fetch 7-day forecast data (max & min temps) from the Open-Meteo API,
    return as a DataFrame with columns: [date, temp_min, temp_max].
    """
    endpoint = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": days,
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
    }
    response = safe_request(endpoint, params=params, timeout=10, max_retries=3)
    if response is None:
        logging.error(f"Failed to fetch forecast data after retries for lat={lat}, lon={lon}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["date", "temp_min", "temp_max"])

    data = response.json()
    if "daily" not in data or "time" not in data["daily"]:
        logging.error("Forecast API returned invalid data structure.")
        return pd.DataFrame(columns=["date", "temp_min", "temp_max"])

    dates = data["daily"]["time"]
    temp_mins = data["daily"].get("temperature_2m_min", [])
    temp_maxs = data["daily"].get("temperature_2m_max", [])
    if not dates or not temp_mins or not temp_maxs:
        logging.error("Forecast API returned empty lists for time or temperatures.")
        return pd.DataFrame(columns=["date", "temp_min", "temp_max"])

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "temp_min": temp_mins,
        "temp_max": temp_maxs
    })
    return df

def fetch_noaa_historical_data(regions_dict, years=1, noaa_token=NOAA_API_TOKEN):
    """
    [EXISTING FUNCTION - Returns a single average HDD per region]

    Fetch ~1 year (or more) of NOAA historical daily TMIN/TMAX data for each city,
    compute average HDD (base 65F), then aggregate to a region-weighted average.
    Returns: dict of { region_name: NOAA_1yr_weighted_HDD }
    """
    end_date = (datetime.now().date() - timedelta(days=1))
    start_date = (end_date - timedelta(days=365 * years))

    city_avg_hdds = {}
    max_failures = 3
    fail_count = 0

    for region_name, cities in regions_dict.items():
        for city in cities:
            station_id = CITY_INFO[city]["station_id"]
            population = CITY_INFO[city]["population"]
            logging.info(f"Fetching NOAA historical data for {city} ({station_id}) from {start_date} to {end_date} ...")

            current_start = start_date
            frames = []
            while current_start < end_date:
                chunk_end = current_start + timedelta(days=365)
                if chunk_end > end_date:
                    chunk_end = end_date

                base_params = {
                    "datasetid": "GHCND",
                    "stationid": station_id,
                    "startdate": str(current_start),
                    "enddate": str(chunk_end),
                    "units": "standard",
                    "datatypeid": ["TMIN", "TMAX"],
                    "sortfield": "date",
                    "sortorder": "asc",
                    "limit": 1000,
                }

                offset = 1
                chunk_frames = []

                while True:
                    params_list = list(base_params.items()) + [("offset", offset)]
                    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
                    headers = {"token": noaa_token}

                    r = safe_request(url, headers=headers, params=params_list, max_retries=3, timeout=10)
                    if r is None:
                        logging.error(f"Failed NOAA request for {city}; skipping the rest of this chunk.")
                        break

                    if r.status_code != 200:
                        logging.error(
                            f"NOAA API request failed with status {r.status_code}.\nResponse text:\n{r.text}\n"
                        )
                        break

                    resp_json = r.json()
                    if "results" not in resp_json or not resp_json["results"]:
                        break

                    results = resp_json["results"]
                    chunk_df = pd.DataFrame(results)
                    if chunk_df.empty:
                        break

                    chunk_df["date"] = pd.to_datetime(chunk_df["date"]).dt.date
                    pivoted = chunk_df.pivot_table(
                        index="date",
                        columns="datatype",
                        values="value",
                        aggfunc="mean"
                    ).reset_index()

                    chunk_frames.append(pivoted)

                    meta = resp_json.get("metadata", {})
                    resultset = meta.get("resultset", {})
                    total_count = resultset.get("count", 0)
                    limit = resultset.get("limit", 1000)
                    current_offset = resultset.get("offset", offset)

                    next_offset = current_offset + limit
                    if next_offset > total_count:
                        break
                    else:
                        offset = next_offset

                if chunk_frames:
                    big_chunk_df = pd.concat(chunk_frames, ignore_index=True)
                    frames.append(big_chunk_df)
                else:
                    logging.info(f"No data returned for {city} from {current_start} to {chunk_end}")

                current_start = chunk_end

            if frames:
                city_data = pd.concat(frames, ignore_index=True)
            else:
                logging.warning(f"No historical data returned for {city}. Setting avg_hdd=0.")
                city_avg_hdds[city] = {"avg_hdd": 0, "population": population}
                fail_count += 1
                if fail_count >= max_failures:
                    logging.critical(f"Too many failures. Aborting after {fail_count} failed city retrievals.")
                    return {}
                continue

            if "TMIN" not in city_data.columns or "TMAX" not in city_data.columns:
                logging.warning(f"Missing TMIN/TMAX data for {city}. Setting HDD=0.")
                city_avg_hdds[city] = {"avg_hdd": 0, "population": population}
                continue

            city_data["TAVG"] = (city_data["TMAX"] + city_data["TMIN"]) / 2.0
            city_data["HDD"] = np.maximum(0, HDD_BASE_TEMP - city_data["TAVG"])
            avg_hdd = city_data["HDD"].mean()

            city_avg_hdds[city] = {"avg_hdd": avg_hdd, "population": population}

    # Region-weighted average
    region_averages = {}
    for region_name, cities in regions_dict.items():
        total_pop = sum(city_avg_hdds[c]["population"] for c in cities)
        if total_pop == 0:
            region_averages[region_name] = 0
            continue

        weighted_sum = 0
        for c in cities:
            pop = city_avg_hdds[c]["population"]
            avg_hdd = city_avg_hdds[c]["avg_hdd"]
            weight = pop / total_pop
            weighted_sum += (avg_hdd * weight)

        region_averages[region_name] = weighted_sum

    return region_averages


def fetch_noaa_historical_daily_data(regions_dict, years=1, noaa_token=NOAA_API_TOKEN):
    """
    [NEW FUNCTION]
    Fetch daily NOAA data (TMIN/TMAX) for each city in each region for ~1 year (or more),
    compute daily HDD, and return a single DataFrame with columns:
        [date, region, HDD].
    """
    end_date = (datetime.now().date() - timedelta(days=1))
    # Adjust the range to ~2 years * 365 if you like:
    start_date = (end_date - timedelta(days=728 * years))

    rows = []  # store (date, region, city, HDD)

    for region_name, cities in regions_dict.items():
        for city in cities:
            station_id = CITY_INFO[city]["station_id"]
            population = CITY_INFO[city]["population"]
            logging.info(f"[DAILY] NOAA data for {city} from {start_date} to {end_date} ...")

            current_start = start_date
            frames = []
            while current_start < end_date:
                chunk_end = current_start + timedelta(days=365)
                if chunk_end > end_date:
                    chunk_end = end_date

                base_params = {
                    "datasetid": "GHCND",
                    "stationid": station_id,
                    "startdate": str(current_start),
                    "enddate": str(chunk_end),
                    "units": "standard",
                    "datatypeid": ["TMIN", "TMAX"],
                    "sortfield": "date",
                    "sortorder": "asc",
                    "limit": 1000,
                }

                offset = 1
                chunk_frames = []

                while True:
                    params_list = list(base_params.items()) + [("offset", offset)]
                    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
                    headers = {"token": noaa_token}

                    r = safe_request(url, headers=headers, params=params_list, max_retries=3, timeout=10)
                    if r is None:
                        logging.error(f"Failed NOAA request for {city}; skipping the rest of this chunk.")
                        break

                    if r.status_code != 200:
                        logging.error(
                            f"NOAA API request failed with status {r.status_code}.\nResponse text:\n{r.text}\n"
                        )
                        break

                    resp_json = r.json()
                    if "results" not in resp_json or not resp_json["results"]:
                        break

                    results = resp_json["results"]
                    chunk_df = pd.DataFrame(results)
                    if chunk_df.empty:
                        break

                    chunk_df["date"] = pd.to_datetime(chunk_df["date"]).dt.date
                    pivoted = chunk_df.pivot_table(
                        index="date",
                        columns="datatype",
                        values="value",
                        aggfunc="mean"
                    ).reset_index()

                    chunk_frames.append(pivoted)

                    meta = resp_json.get("metadata", {})
                    resultset = meta.get("resultset", {})
                    total_count = resultset.get("count", 0)
                    limit = resultset.get("limit", 1000)
                    current_offset = resultset.get("offset", offset)

                    next_offset = current_offset + limit
                    if next_offset > total_count:
                        break
                    else:
                        offset = next_offset

                if chunk_frames:
                    big_chunk_df = pd.concat(chunk_frames, ignore_index=True)
                    frames.append(big_chunk_df)

                current_start = chunk_end

            # Combine data for this city
            if not frames:
                logging.warning(f"No daily data for {city}; skipping.")
                continue

            city_data = pd.concat(frames, ignore_index=True)
            if "TMIN" not in city_data.columns or "TMAX" not in city_data.columns:
                logging.warning(f"Missing TMIN/TMAX columns for {city}. Skipping.")
                continue

            city_data["TAVG"] = (city_data["TMAX"] + city_data["TMIN"]) / 2.0
            city_data["HDD"] = np.maximum(0, HDD_BASE_TEMP - city_data["TAVG"])

            # We'll do population weighting later if needed. For correlation, we may treat each region's
            # total HDD as a population average. Let's just store for each date & city.
            for idx, row_data in city_data.iterrows():
                rows.append({
                    "date": row_data["date"],
                    "region": region_name,
                    "city": city,
                    "HDD": row_data["HDD"]
                })

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        logging.error("No daily NOAA data returned at all.")
        return pd.DataFrame(columns=["date", "region", "HDD"])

    # Next: get population-weighted daily HDD for each region
    df_all = df_all.merge(
        pd.DataFrame([
            {"city": c, "population": CITY_INFO[c]["population"]} for c in CITY_INFO
        ]),
        on="city",
        how="left"
    )

    # sum population by region
    region_pop = df_all.groupby("region")["population"].transform("sum")
    df_all["Weighted_HDD"] = df_all["HDD"] * (df_all["population"] / region_pop)

    # Now group by date & region to get total Weighted HDD
    df_region_daily = df_all.groupby(["date", "region"], as_index=False).agg({
        "Weighted_HDD": "sum"
    })
    df_region_daily.rename(columns={"Weighted_HDD": "HDD"}, inplace=True)

    return df_region_daily


# -----------------------------
# Replace yFinance with IB API for NG Futures
# -----------------------------

class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.historical_data = []
        self.done = False

    def historicalData(self, reqId, bar):
        self.historical_data.append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        logging.info("Historical data request complete")
        self.done = True

    def error(self, reqId, errorCode, errorString):
        logging.error(f"IB error {errorCode} on reqId {reqId}: {errorString}")

def create_ng_futures_contract():
    """
    Create an IB Contract for NG futures using your provided contract details.
    """
    contract = Contract()
    contract.symbol = "NG"
    contract.secType = "FUT"
    contract.exchange = "NYMEX"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = "20250528"  # Expiry date as YYYYMMDD
    contract.localSymbol = "NGM5"
    contract.multiplier = "10000"
    return contract

def fetch_ng_futures_data(start_date, end_date):
    """
    Connect to IB, request historical data for the NG futures contract over a
    user-specified period, wait for data, then return it in a DataFrame.

    NOTE: If you want a full year, you might set durationStr to "365 D".
    IB has limits, so you may need multiple calls for longer data.
    """
    app = IBApp()
    try:
        app.connect("127.0.0.1", 7496, clientId=123)
    except Exception as e:
        logging.error(f"Could not connect to IB API: {e}")
        return pd.DataFrame()

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    contract = create_ng_futures_contract()
    end_dt_str = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    # Example: request 180 days of daily bars to overlap with NOAA data
    # (You can adjust as needed, e.g. "365 D", etc.)
    app.reqHistoricalData(
        reqId=1,
        contract=contract,
        endDateTime=end_dt_str,
        durationStr="2 Y",   # <--- ADJUST AS DESIRED
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    # Wait for data or timeout
    timeout = 30
    start_wait = time.time()
    while not app.done and (time.time() - start_wait) < timeout:
        time.sleep(0.1)

    if not app.done:
        logging.warning("Historical data request timed out.")

    app.disconnect()

    df = pd.DataFrame(app.historical_data)
    if df.empty:
        logging.warning("No NG futures data returned from IB.")
        return df

    # Rename columns
    df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True
    )

    # Convert Date to datetime if needed
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    return df

def fetch_eia_storage_data():
    """
    Mock EIA storage data per region. In real usage, you'd query EIA's API.
    """
    return {
        "East": -3.0,
        "Midwest": -2.0,
        "South Central": 1.0,
        "Mountain": 0.5,
        "Pacific": -1.0,
    }

# -----------------------------
# 3. Calculation & Aggregation
# -----------------------------
def calculate_hdd(row):
    """
    Calculate daily HDD from min/max temperatures using base 65F.
    """
    avg_temp = (row["temp_min"] + row["temp_max"]) / 2.0
    return max(0, HDD_BASE_TEMP - avg_temp)

def aggregate_region_hdd(city_forecasts):
    """
    Given a dict of city->forecastDF, compute population-weighted HDD
    for the entire region. For forecast (7-day).
    Returns a DataFrame with columns [date, Region_HDD].
    """
    total_pop = sum(CITY_INFO[city]["population"] for city in city_forecasts)
    if total_pop == 0:
        logging.warning("Warning: total population is 0; using equal weighting.")
        city_count = len(city_forecasts)
        for city, df in city_forecasts.items():
            df["HDD"] = df.apply(calculate_hdd, axis=1)
            df["Weighted_HDD"] = df["HDD"] / city_count
        city_dfs = [
            df[["date", "Weighted_HDD"]].rename(columns={"Weighted_HDD": f"{city}_WeightedHDD"})
            for city, df in city_forecasts.items()
        ]
        merged_df = city_dfs[0]
        for df in city_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on="date", how="outer")
        hdd_cols = [col for col in merged_df.columns if col.endswith("_WeightedHDD")]
        merged_df["Region_HDD"] = merged_df[hdd_cols].sum(axis=1)
        return merged_df[["date", "Region_HDD"]]

    # Otherwise, population weighting
    city_dfs = []
    for city, df in city_forecasts.items():
        df = df.copy()
        df["HDD"] = df.apply(calculate_hdd, axis=1)
        city_pop = CITY_INFO[city]["population"]
        weight = city_pop / total_pop
        df["Weighted_HDD"] = df["HDD"] * weight

        city_dfs.append(df[["date", "Weighted_HDD"]].rename(columns={"Weighted_HDD": f"{city}_WeightedHDD"}))

    merged_df = city_dfs[0]
    for df in city_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="date", how="outer")

    hdd_cols = [col for col in merged_df.columns if col.endswith("_WeightedHDD")]
    merged_df["Region_HDD"] = merged_df[hdd_cols].sum(axis=1)
    return merged_df[["date", "Region_HDD"]]

def calculate_gwdd(region_forecast_dfs, region_noaa_avgs):
    """
    1) For each region, take the 'Region_HDD' forecast.
    2) Multiply by the region's NOAA 1-year average HDD (weighted) to get 'Region_GWDD'.
    3) Merge all region frames on 'date', sum across 'Region_GWDD' for a total 'GWDD'.
    """
    df_list = []
    for region_name, forecast_df in region_forecast_dfs.items():
        tmp_df = forecast_df.copy()
        noaa_weight = region_noaa_avgs.get(region_name, 0.0)
        tmp_df[f"{region_name}_GWDD"] = tmp_df["Region_HDD"] * noaa_weight
        df_list.append(tmp_df[["date", f"{region_name}_GWDD"]])

    if not df_list:
        logging.error("No regional forecast data frames found; returning empty.")
        return pd.DataFrame(columns=["date", "GWDD"])

    gwdd_df = df_list[0]
    for df in df_list[1:]:
        gwdd_df = pd.merge(gwdd_df, df, on="date", how="outer")

    gwdd_cols = [col for col in gwdd_df.columns if col.endswith("_GWDD")]
    gwdd_df["GWDD"] = gwdd_df[gwdd_cols].sum(axis=1)
    gwdd_df.sort_values("date", inplace=True)
    return gwdd_df[["date", "GWDD"]]

# -----------------------------
# 4. Heuristic & Signal Generation
# -----------------------------

def generate_signals(region_hdd_df, region_name, noaa_avg_hdd, ng_price_df, eia_storage):
    """
    Simple example signal logic comparing forecast HDD with historical averages,
    price trends, etc.
    """
    signals = {}

    # Compare average forecast HDD vs NOAA historical average
    avg_forecast_hdd = region_hdd_df["Region_HDD"].mean()
    avg_diff = avg_forecast_hdd - noaa_avg_hdd
    if avg_diff > 2.0:
        signals["hdd_vs_5yr"] = "Bullish"
    elif avg_diff < -2.0:
        signals["hdd_vs_5yr"] = "Bearish"
    else:
        signals["hdd_vs_5yr"] = "Neutral"

    # Delta between first and last forecast day
    first_day = region_hdd_df["Region_HDD"].iloc[0]
    last_day = region_hdd_df["Region_HDD"].iloc[-1]
    delta = last_day - first_day
    perc_change = (delta / first_day * 100) if first_day != 0 else 0

    if perc_change > 15:
        signals["delta_forecast"] = "Bullish"
    elif perc_change < -10:
        signals["delta_forecast"] = "Bearish"
    else:
        signals["delta_forecast"] = "Neutral"

    # Rolling 3-day average trend
    region_hdd_df["rolling_trend"] = region_hdd_df["Region_HDD"].rolling(window=3, min_periods=1).mean()
    trend_delta = region_hdd_df["rolling_trend"].iloc[-1] - region_hdd_df["rolling_trend"].iloc[0]

    if trend_delta > 0:
        signals["rolling_trend"] = "Bullish"
    elif trend_delta < 0:
        signals["rolling_trend"] = "Bearish"
    else:
        signals["rolling_trend"] = "Neutral"

    # NG price trend
    if not ng_price_df.empty:
        ng_price_df = ng_price_df.sort_values("Date").reset_index(drop=True)
        price_first = ng_price_df["Close"].iloc[0]
        price_last = ng_price_df["Close"].iloc[-1]
        price_delta = price_last - price_first
        price_perc_change = (price_delta / price_first * 100) if price_first != 0 else 0

        if price_perc_change > 2:
            signals["price_trend"] = "Bullish"
        elif price_perc_change < -2:
            signals["price_trend"] = "Bearish"
        else:
            signals["price_trend"] = "Neutral"
    else:
        signals["price_trend"] = "Neutral"

    # Simple storage-based signal
    region_storage_mapping = {
        "Northeast": "East",
        "Midwest": "Midwest",
        "South Central": "South Central",
        "Mountain": "Mountain",
        "Pacific": "Pacific"
    }
    storage_value = eia_storage.get(region_storage_mapping.get(region_name, "East"), 0)

    if storage_value < -1:
        signals["storage"] = "Bullish"
    elif storage_value > 1:
        signals["storage"] = "Bearish"
    else:
        signals["storage"] = "Neutral"

    # Combine signals into a final simple bias
    bullish_count = sum(1 for v in signals.values() if v == "Bullish")
    bearish_count = sum(1 for v in signals.values() if v == "Bearish")
    if bullish_count > bearish_count:
        signals["final_signal"] = "Bullish Bias"
    elif bearish_count > bullish_count:
        signals["final_signal"] = "Bearish Bias"
    else:
        signals["final_signal"] = "Neutral"

    return signals

# -----------------------------------
# 4b. Correlation & Scenario Analysis
# -----------------------------------
def scenario_matrix(df, regions, hdd_changes):
    """
    Generate a scenario matrix for price move, given possible HDD changes.
    We'll assume a naive linear approximation based on correlation & std dev.
    """
    scenario_results = []
    for region_col in regions:
        slope = estimate_price_move(df, region_col)
        if slope is None:
            continue

        for change in hdd_changes:
            price_move = slope * change
            scenario_results.append({
                "Region": region_col,
                "HDD_Change%": change,
                "Estimated_Price_Move%": price_move
            })

    return pd.DataFrame(scenario_results)

def estimate_price_move(df, hdd_col):
    """
    Approximate slope of Price_Change vs region's HDD_Change using correlation.
    slope = r * (std(Price) / std(HDD)).
    """
    subset = df.dropna(subset=[hdd_col, "Price_Change"])
    if subset.empty:
        return None

    corr = subset[hdd_col].corr(subset["Price_Change"])
    std_hdd = subset[hdd_col].std()
    std_price = subset["Price_Change"].std()
    if std_hdd == 0:
        return None
    slope = corr * (std_price / std_hdd)
    return slope

# ----------------------------
# 5. Main Orchestrating Logic
# ----------------------------

def main():
    logging.info("Starting Heating Degree Days analysis and NG signal generation...")

    # ----------------
    # PART A: NOAA
    # ----------------
    logging.info("Retrieving 1-year+ average HDD from NOAA (population-weighted).")
    region_noaa_avgs = fetch_noaa_historical_data(REGIONS, years=HISTORICAL_YEARS)
    if not region_noaa_avgs:
        logging.error("Failed to retrieve NOAA average data or too many failures. Exiting.")
        return

    # Also fetch daily NOAA data for correlation
    logging.info("Fetching daily NOAA historical HDD data for correlation analysis...")
    df_region_daily_hdd = fetch_noaa_historical_daily_data(REGIONS, years=HISTORICAL_YEARS)
    if df_region_daily_hdd.empty:
        logging.error("No daily NOAA data available; skipping correlation analysis.")
        df_region_daily_hdd = pd.DataFrame()

    # ----------------
    # PART B: EIA Storage
    # ----------------
    logging.info("Fetching EIA storage data (mock).")
    eia_storage_data = fetch_eia_storage_data()

    # ----------------
    # PART C: NG Prices
    # ----------------
    logging.info("Fetching extended NG futures data from IB for daily correlation.")
    ng_prices_df = fetch_ng_futures_data(start_date="20230101", end_date="20240101")
    if ng_prices_df.empty:
        logging.warning("No NG price data returned, correlation analysis will be incomplete.")

    # ----------------
    # PART D: 7-day Forecast
    # ----------------
    CITY_COORDS = {
        "New York City": (40.7128, -74.0060),
        "Boston": (42.3601, -71.0589),
        "Chicago": (41.8781, -87.6298),
        "Detroit": (42.3314, -83.0458),
        "Dallas": (32.7767, -96.7970),
        "Houston": (29.7604, -95.3698),
        "Denver": (39.7392, -104.9903),
        "Salt Lake City": (40.7608, -111.8910),
        "Los Angeles": (34.0522, -118.2437),
    }

    region_signals = {}
    region_forecast_dfs = {}

    for region_name, cities in REGIONS.items():
        city_forecasts = {}
        for city in cities:
            lat, lon = CITY_COORDS[city]
            forecast_df = fetch_forecast_temperatures(lat, lon, days=FORECAST_DAYS)
            if forecast_df.empty:
                logging.warning(f"No forecast data for {city}; skipping.")
                continue
            city_forecasts[city] = forecast_df

        if not city_forecasts:
            logging.info(f"No valid city forecasts for region {region_name}. Skipping signals.")
            continue

        # Population-weighted region HDD for the 7-day forecast
        region_hdd_df = aggregate_region_hdd(city_forecasts)
        region_forecast_dfs[region_name] = region_hdd_df

        # Generate signals
        noaa_avg = region_noaa_avgs.get(region_name, 0)
        region_signal = generate_signals(region_hdd_df, region_name, noaa_avg, ng_prices_df, eia_storage_data)
        region_signals[region_name] = region_signal

    # Calculate total Gas Weighted Degree Days
    gwdd_df = calculate_gwdd(region_forecast_dfs, region_noaa_avgs)

    # ----------------
    # PART E: Correlation Analysis
    # ----------------
    if not df_region_daily_hdd.empty and not ng_prices_df.empty:
        # 1) Pivot region daily HDD => columns: [date, region1, region2, ...]
        df_hdd_pivot = df_region_daily_hdd.pivot(index="date", columns="region", values="HDD").reset_index()
        df_hdd_pivot.columns.name = None

        # Ensure datetime
        df_hdd_pivot["date"] = pd.to_datetime(df_hdd_pivot["date"])
        ng_prices_df["Date"] = pd.to_datetime(ng_prices_df["Date"])

        # --- ADDED: Overlapping Date Range Logic ---
        hdd_min = df_hdd_pivot["date"].min()
        hdd_max = df_hdd_pivot["date"].max()
        price_min = ng_prices_df["Date"].min()
        price_max = ng_prices_df["Date"].max()

        start_common = max(hdd_min, price_min)
        end_common   = min(hdd_max, price_max)

        # Filter both DataFrames to this intersection
        df_hdd_pivot = df_hdd_pivot[
            (df_hdd_pivot["date"] >= start_common) &
            (df_hdd_pivot["date"] <= end_common)
        ]
        # We'll copy so we don't overwrite our original
        df_prices_filtered = ng_prices_df[
            (ng_prices_df["Date"] >= start_common) &
            (ng_prices_df["Date"] <= end_common)
        ].copy()

        # rename "Date" => "date"
        df_prices_filtered.rename(columns={"Date":"date"}, inplace=True)

        # 2) Merge on the same date range
        df_merged = pd.merge(
            df_hdd_pivot,
            df_prices_filtered[["date", "Close"]],
            on="date",
            how="inner"
        )

        # 3) Drop rows that have missing data for Price or HDD columns
        subset_cols = ["Close"] + [r for r in REGIONS if r in df_merged.columns]
        df_merged.dropna(subset=subset_cols, inplace=True)

        # Now we can do the original correlation logic
        df_merged.sort_values("date", inplace=True)

        # Compute daily % change for Price
        df_merged["Price_Change"] = df_merged["Close"].pct_change() * 100

        for region in REGIONS:
            if region in df_merged.columns:
                df_merged[f"{region}_Change"] = df_merged[region].pct_change() * 100

        cols_of_interest = ["Price_Change"] + [f"{r}_Change" for r in REGIONS if f"{r}_Change" in df_merged.columns]

        corr_matrix = df_merged[cols_of_interest].corr()
        logging.info("\nCorrelation Matrix (Daily % Changes):")
        logging.info(f"\n{corr_matrix}")

        # Scenario matrix
        scenario_df = scenario_matrix(
            df_merged,
            [f"{r}_Change" for r in REGIONS if f"{r}_Change" in df_merged.columns],
            hdd_changes=[-20, -10, 0, 10, 20]
        )
        logging.info("\nScenario Matrix (price move vs. hypothetical HDD changes):")
        logging.info(f"\n{scenario_df.to_string(index=False)}")

    else:
        logging.warning("Skipping correlation/scenario matrix because daily data is incomplete.")

    # ----------------
    # PART F: Output
    # ----------------
    logging.info("\n=== NOAA 1-Year+ Avg HDD (Weighted) by Region ===")
    for region_name, val in region_noaa_avgs.items():
        logging.info(f"{region_name}: {val:.2f}")

    logging.info("\n=== 7-Day Forecast HDDs & Signals Per Region ===")
    for region, signals in region_signals.items():
        logging.info(f"\nRegion: {region}")
        logging.info(f"Forecasted HDD Data:\n{region_forecast_dfs[region].to_string(index=False)}")
        logging.info(f"Signals: {signals}")

    logging.info("\n=== Gas Weighted Degree Days (GWDD) - 7-Day Forecast ===")
    if not gwdd_df.empty:
        logging.info(f"\n{gwdd_df.to_string(index=False)}")
    else:
        logging.info("No GWDD data available.")

    logging.info("Analysis complete.")

if __name__ == "__main__":
    main()
