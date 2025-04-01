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

# Practical Scheduling Suggestions (Refined & Summarized)
# ... (Unchanged)...

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ----------------------------
# 1. Configuration & Constants
# ----------------------------
load_dotenv()
NOAA_API_TOKEN = os.getenv('NOAA_API_TOKEN')

if not NOAA_API_TOKEN:
    raise ValueError("NOAA_API_TOKEN is not set in your environment variables.")

CITY_INFO = {
    "New York City": {"station_id": "GHCND:USW00094728", "population": 8804190},
    "Boston":        {"station_id": "GHCND:USW00014739", "population": 684379},
    "Chicago":       {"station_id": "GHCND:USW00094846", "population": 2746388},
    "Detroit":       {"station_id": "GHCND:USW00094847", "population": 632464},
    "Denver":        {"station_id": "GHCND:USW00003017", "population": 715522},
    "Salt Lake City":{"station_id": "GHCND:USW00024127", "population": 200133},
    # Removed warm “South Central” & “Pacific” from the dictionary
}

# We no longer define "South Central" & "Pacific" in REGIONS
REGIONS = {
    "Northeast": ["New York City", "Boston"],
    "Midwest":   ["Chicago", "Detroit"],
    "Mountain":  ["Denver", "Salt Lake City"],
}

HDD_BASE_TEMP = 65
FORECAST_DAYS = 7
HISTORICAL_YEARS = 2  # how many years of data to fetch from NOAA

def safe_request(
    url: str,
    headers=None,
    params=None,
    max_retries=3,
    timeout=30,
    backoff_factor=2
):
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Timeout:
            logging.warning(f"Request timeout on attempt {attempt}/{max_retries} for URL: {url}")
        except RequestException as e:
            logging.warning(f"Request exception on attempt {attempt}/{max_retries}: {e}")

        sleep_time = (backoff_factor ** (attempt - 1))
        logging.info(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)

    logging.error(f"All {max_retries} retries failed for {url}.")
    return None

def fetch_forecast_temperatures(lat: float, lon: float, days: int = FORECAST_DAYS):
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
        logging.error(f"Failed forecast fetch lat={lat}, lon={lon}. Returning empty DataFrame.")
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
                chunk_end = current_start + timedelta(days=180)
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

                    r = safe_request(url, headers=headers, params=params_list, max_retries=3, timeout=30)
                    if r is None:
                        logging.error(f"Failed NOAA request for {city}; skipping chunk {current_start} - {chunk_end}.")
                        break

                    if r.status_code != 200:
                        logging.error(f"NOAA API request failed status {r.status_code} => skipping chunk.")
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
                    logging.info(f"No data returned for {city} in chunk {current_start} - {chunk_end}")

                current_start = chunk_end

            if frames:
                city_data = pd.concat(frames, ignore_index=True)
            else:
                logging.warning(f"No historical data returned for {city}. Setting avg_hdd=0.")
                city_avg_hdds[city] = {"avg_hdd": 0, "population": population}
                fail_count += 1
                if fail_count >= max_failures:
                    logging.critical("Too many failures => Aborting.")
                    return {}
                continue

            if "TMIN" not in city_data.columns or "TMAX" not in city_data.columns:
                logging.warning(f"Missing TMIN/TMAX => city HDD=0 for {city}.")
                city_avg_hdds[city] = {"avg_hdd": 0, "population": population}
                continue

            city_data["TAVG"] = (city_data["TMAX"] + city_data["TMIN"]) / 2.0
            city_data["HDD"] = np.maximum(0, 65 - city_data["TAVG"])
            avg_hdd = city_data["HDD"].mean()
            city_avg_hdds[city] = {"avg_hdd": avg_hdd, "population": population}

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
    end_date = (datetime.now().date() - timedelta(days=1))
    start_date = (end_date - timedelta(days=728 * years))
    rows = []

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

                    r = safe_request(url, headers=headers, params=params_list, max_retries=3, timeout=30)
                    if r is None:
                        logging.error(f"Failed NOAA request daily for {city} => skipping chunk.")
                        break

                    if r.status_code != 200:
                        logging.error(f"NOAA daily request => status {r.status_code}, skipping.")
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
                    frames.append(pd.concat(chunk_frames, ignore_index=True))

                current_start = chunk_end

            if not frames:
                logging.warning(f"No daily data => skipping {city}.")
                continue

            city_data = pd.concat(frames, ignore_index=True)
            if "TMIN" not in city_data.columns or "TMAX" not in city_data.columns:
                logging.warning(f"Missing TMIN/TMAX => skipping city {city}.")
                continue

            city_data["TAVG"] = (city_data["TMAX"] + city_data["TMIN"]) / 2.0
            city_data["HDD"] = np.maximum(0, 65 - city_data["TAVG"])

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

    df_all = df_all.merge(
        pd.DataFrame([{"city": c, "population": CITY_INFO[c]["population"]} for c in CITY_INFO]),
        on="city",
        how="left"
    )
    region_pop = df_all.groupby("region")["population"].transform("sum")
    df_all["Weighted_HDD"] = df_all["HDD"] * (df_all["population"] / region_pop)

    df_region_daily = df_all.groupby(["date", "region"], as_index=False).agg({"Weighted_HDD": "sum"})
    df_region_daily.rename(columns={"Weighted_HDD": "HDD"}, inplace=True)
    return df_region_daily

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
    contract = Contract()
    contract.symbol = "NG"
    contract.secType = "FUT"
    contract.exchange = "NYMEX"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = "20250528"
    contract.localSymbol = "NGM5"
    contract.multiplier = "10000"
    return contract

def fetch_ng_futures_data(start_date, end_date):
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

    app.reqHistoricalData(
        reqId=1,
        contract=contract,
        endDateTime=end_dt_str,
        durationStr="2 Y",      # or "1 Y" if you want a smaller range
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    timeout = 30
    start_wait = time.time()
    while not app.done and (time.time() - start_wait) < timeout:
        time.sleep(0.1)

    # Check if we timed out
    if not app.done:
        logging.warning("Historical data request timed out.")

    # Disconnect from IB regardless
    app.disconnect()

    # Build DataFrame from the collected bars
    df = pd.DataFrame(app.historical_data)
    logging.info(f"IB returned {len(app.historical_data)} bars for NG data.")
    if df.empty:
        logging.warning("No NG futures data returned from IB.")
        return df

    # Rename columns to standard names
    df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }, inplace=True)

    # Convert Date column to datetime and sort
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    logging.info(
        f"Price DataFrame shape: {df.shape}, "
        f"Dates: {df['Date'].min()} -> {df['Date'].max()}"
    )

    return df

def fetch_eia_storage_data():
    return {
        "East": -3.0,
        "Midwest": -2.0,
        "Mountain": 0.5
    }

def calculate_hdd(row):
    avg_temp = (row["temp_min"] + row["temp_max"]) / 2.0
    return max(0, HDD_BASE_TEMP - avg_temp)

def aggregate_region_hdd(city_forecasts):
    total_pop = sum(CITY_INFO[city]["population"] for city in city_forecasts)
    if total_pop == 0:
        logging.warning("Warning: total population=0 => using equal weighting.")
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
    df_list = []
    for region_name, forecast_df in region_forecast_dfs.items():
        tmp_df = forecast_df.copy()
        noaa_weight = region_noaa_avgs.get(region_name, 0.0)
        tmp_df[f"{region_name}_GWDD"] = tmp_df["Region_HDD"] * noaa_weight
        df_list.append(tmp_df[["date", f"{region_name}_GWDD"]])

    if not df_list:
        logging.error("No regional forecast data => returning empty.")
        return pd.DataFrame(columns=["date", "GWDD"])

    gwdd_df = df_list[0]
    for df in df_list[1:]:
        gwdd_df = pd.merge(gwdd_df, df, on="date", how="outer")

    gwdd_cols = [col for col in gwdd_df.columns if col.endswith("_GWDD")]
    gwdd_df["GWDD"] = gwdd_df[gwdd_cols].sum(axis=1)
    gwdd_df.sort_values("date", inplace=True)
    return gwdd_df[["date", "GWDD"]]

def generate_signals(region_hdd_df, region_name, noaa_avg_hdd, ng_price_df, eia_storage):
    signals = {}

    avg_forecast_hdd = region_hdd_df["Region_HDD"].mean()
    avg_diff = avg_forecast_hdd - noaa_avg_hdd
    if avg_diff > 2.0:
        signals["hdd_vs_5yr"] = "Bullish"
    elif avg_diff < -2.0:
        signals["hdd_vs_5yr"] = "Bearish"
    else:
        signals["hdd_vs_5yr"] = "Neutral"

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

    region_hdd_df["rolling_trend"] = region_hdd_df["Region_HDD"].rolling(window=3, min_periods=1).mean()
    trend_delta = region_hdd_df["rolling_trend"].iloc[-1] - region_hdd_df["rolling_trend"].iloc[0]
    if trend_delta > 0:
        signals["rolling_trend"] = "Bullish"
    elif trend_delta < 0:
        signals["rolling_trend"] = "Bearish"
    else:
        signals["rolling_trend"] = "Neutral"

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

    region_storage_mapping = {
        "Northeast": "East",
        "Midwest":   "Midwest",
        "Mountain":  "Mountain",
    }
    storage_value = eia_storage.get(region_storage_mapping.get(region_name, "East"), 0)
    if storage_value < -1:
        signals["storage"] = "Bullish"
    elif storage_value > 1:
        signals["storage"] = "Bearish"
    else:
        signals["storage"] = "Neutral"

    bullish_count = sum(1 for v in signals.values() if v == "Bullish")
    bearish_count = sum(1 for v in signals.values() if v == "Bearish")
    if bullish_count > bearish_count:
        signals["final_signal"] = "Bullish Bias"
    elif bearish_count > bullish_count:
        signals["final_signal"] = "Bearish Bias"
    else:
        signals["final_signal"] = "Neutral"

    return signals

def scenario_matrix(df, regions, hdd_changes):
    scenario_results = []
    for region_col in regions:
        if region_col in df.columns:
            std_val = df[region_col].std()
            if std_val < 0.001:
                logging.info(f"Skipping {region_col} - std dev too small: {std_val}")
                continue

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

def main():
    logging.info("Starting Heating Degree Days analysis and NG signal generation...")

    region_noaa_avgs = fetch_noaa_historical_data(REGIONS, years=HISTORICAL_YEARS)
    if not region_noaa_avgs:
        logging.error("Failed NOAA avg data => exit.")
        return

    df_region_daily_hdd = fetch_noaa_historical_daily_data(REGIONS, years=HISTORICAL_YEARS)
    if df_region_daily_hdd.empty:
        logging.error("No daily NOAA => skip correlation.")
        df_region_daily_hdd = pd.DataFrame()

    eia_storage_data = fetch_eia_storage_data()

    ng_prices_df = fetch_ng_futures_data(start_date="20230101", end_date="20240101")
    if ng_prices_df.empty:
        logging.warning("No NG price data => correlation incomplete.")

    CITY_COORDS = {
        "New York City": (40.7128, -74.0060),
        "Boston":        (42.3601, -71.0589),
        "Chicago":       (41.8781, -87.6298),
        "Detroit":       (42.3314, -83.0458),
        "Denver":        (39.7392, -104.9903),
        "Salt Lake City":(40.7608, -111.8910),
    }

    region_signals = {}
    region_forecast_dfs = {}

    for region_name, cities in REGIONS.items():
        city_forecasts = {}
        for city in cities:
            lat, lon = CITY_COORDS[city]
            forecast_df = fetch_forecast_temperatures(lat, lon, days=FORECAST_DAYS)
            if forecast_df.empty:
                logging.warning(f"No forecast data => skip {city}.")
                continue
            city_forecasts[city] = forecast_df

        if not city_forecasts:
            logging.info(f"No valid city forecasts => skip region {region_name}.")
            continue

        region_hdd_df = aggregate_region_hdd(city_forecasts)
        region_forecast_dfs[region_name] = region_hdd_df
        noaa_avg = region_noaa_avgs.get(region_name, 0)
        region_signal = generate_signals(region_hdd_df, region_name, noaa_avg, ng_prices_df, eia_storage_data)
        region_signals[region_name] = region_signal

    gwdd_df = calculate_gwdd(region_forecast_dfs, region_noaa_avgs)

    # Only proceed if we have NOAA daily HDD + NG price data
    if not df_region_daily_hdd.empty and not ng_prices_df.empty:
        df_hdd_pivot = df_region_daily_hdd.pivot(index="date", columns="region", values="HDD").reset_index()
        df_hdd_pivot.columns.name = None

        df_hdd_pivot["date"] = pd.to_datetime(df_hdd_pivot["date"])
        ng_prices_df["Date"] = pd.to_datetime(ng_prices_df["Date"])

        # Combine NOAA + Price date ranges
        hdd_min = df_hdd_pivot["date"].min()
        hdd_max = df_hdd_pivot["date"].max()
        price_min = ng_prices_df["Date"].min()
        price_max = ng_prices_df["Date"].max()

        start_common = max(hdd_min, price_min)
        end_common   = min(hdd_max, price_max)

        df_hdd_pivot = df_hdd_pivot[
            (df_hdd_pivot["date"] >= start_common) &
            (df_hdd_pivot["date"] <= end_common)
        ]
        df_prices_filtered = ng_prices_df[
            (ng_prices_df["Date"] >= start_common) &
            (ng_prices_df["Date"] <= end_common)
        ].copy()
        df_prices_filtered.rename(columns={"Date":"date"}, inplace=True)

        df_merged = pd.merge(
            df_hdd_pivot,
            df_prices_filtered[["date", "Close"]],
            on="date",
            how="inner"
        )

        # Filter for winter months only
        df_merged = df_merged[df_merged["date"].dt.month.isin([10, 11, 12, 1, 2, 3, 4])]

        # Drop any remaining rows missing key columns
        subset_cols = ["Close"] + [r for r in REGIONS if r in df_merged.columns]
        df_merged.dropna(subset=subset_cols, inplace=True)
        df_merged.sort_values("date", inplace=True)

        # Add Price_Change + region changes
        df_merged["Price_Change"] = df_merged["Close"].pct_change() * 100
        for region in REGIONS:
            if region in df_merged.columns:
                df_merged[f"{region}_Change"] = df_merged[region].pct_change() * 100

        # === NEW: Print how many winter rows remain
        logging.info(f"Final winter subset rows: {len(df_merged)}")

        # === NEW: Print count & std dev for each region's % change
        for region in REGIONS:
            col = f"{region}_Change"
            if col in df_merged.columns:
                region_count = df_merged[col].count()
                region_std = df_merged[col].std()
                logging.info(f"{region} => count= {region_count}, std dev= {region_std}")

        # Now do correlation
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
        logging.warning("Skipping correlation => daily data incomplete.")

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
