import os
import requests
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# IB API imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

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
# 2. Data Retrieval Functions
# -----------------------------

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
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    data = response.json()

    dates = data["daily"]["time"]
    temp_mins = data["daily"]["temperature_2m_min"]
    temp_maxs = data["daily"]["temperature_2m_max"]

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "temp_min": temp_mins,
        "temp_max": temp_maxs
    })
    return df

def fetch_noaa_historical_data(regions_dict, years=1, noaa_token=NOAA_API_TOKEN):
    """
    Fetch ~1 year of NOAA historical daily TMIN/TMAX data for each city,
    compute average HDD (base 65F), then aggregate to a region-weighted average.
    Returns: dict of { region_name: NOAA_1yr_weighted_HDD }
    """
    end_date = (datetime.now().date() - timedelta(days=1))
    start_date = (end_date - timedelta(days=365 * years))

    city_avg_hdds = {}
    for region_name, cities in regions_dict.items():
        for city in cities:
            station_id = CITY_INFO[city]["station_id"]
            population = CITY_INFO[city]["population"]
            print(f"Fetching NOAA historical data for {city} ({station_id}) from {start_date} to {end_date} ...")

            current_start = start_date
            frames = []
            while current_start < end_date:
                chunk_end = current_start + timedelta(days=365)
                if chunk_end > end_date:
                    chunk_end = end_date

                # Build base params for this chunk
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
                    # Convert base_params to list of tuples for requests
                    params_list = list(base_params.items()) + [("offset", offset)]
                    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
                    headers = {"token": noaa_token}

                    r = requests.get(url, headers=headers, params=params_list)
                    if r.status_code != 200:
                        print(f"\nNOAA API request failed with status {r.status_code}. Response text:\n{r.text}\n")
                        break  # Exit the pagination loop, go on to next chunk

                    resp_json = r.json()

                    # If no "results", we are done with pagination for this chunk
                    if "results" not in resp_json or not resp_json["results"]:
                        break

                    # Convert results to a DataFrame and pivot
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

                    # Check metadata to see if more pages remain
                    meta = resp_json.get("metadata", {})
                    resultset = meta.get("resultset", {})
                    total_count = resultset.get("count", 0)
                    limit = resultset.get("limit", 1000)
                    current_offset = resultset.get("offset", offset)

                    # Update offset for next page
                    next_offset = current_offset + limit
                    if next_offset > total_count:
                        # No more pages of data
                        break
                    else:
                        offset = next_offset

                # After pagination for this chunk, combine data
                if chunk_frames:
                    big_chunk_df = pd.concat(chunk_frames, ignore_index=True)
                    frames.append(big_chunk_df)
                else:
                    print(f"No data returned for {city} from {current_start} to {chunk_end}")

                # Move to next chunk
                current_start = chunk_end

            if frames:
                city_data = pd.concat(frames, ignore_index=True)
            else:
                print(f"No historical data returned for {city}.")
                city_avg_hdds[city] = {"avg_hdd": 0, "population": population}
                continue

            if "TMIN" not in city_data.columns or "TMAX" not in city_data.columns:
                print(f"Missing TMIN/TMAX data for {city}. Setting HDD to 0.")
                city_avg_hdds[city] = {"avg_hdd": 0, "population": population}
                continue

            # Calculate HDD = max(0, 65 - avg_temp)
            city_data["TAVG"] = (city_data["TMAX"] + city_data["TMIN"]) / 2.0
            city_data["HDD"] = np.maximum(0, HDD_BASE_TEMP - city_data["TAVG"])
            avg_hdd = city_data["HDD"].mean()

            city_avg_hdds[city] = {"avg_hdd": avg_hdd, "population": population}

    # Aggregate city-level avg HDDs into region-weighted average
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
        print("Historical data request complete")
        self.done = True

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
    Connect to IB, request historical data for the NG futures contract,
    wait for data, then return it in a DataFrame.
    """
    app = IBApp()
    app.connect("127.0.0.1", 7496, clientId=123)
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    contract = create_ng_futures_contract()
    end_dt_str = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    # Example: 14 days of daily bars
    app.reqHistoricalData(
        reqId=1,
        contract=contract,
        endDateTime=end_dt_str,
        durationStr="14 D",
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
        print("Historical data request timed out.")

    app.disconnect()

    # Convert the list of dicts into a DataFrame
    df = pd.DataFrame(app.historical_data)

    # Rename the columns to align with usage
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

    # (Optional) parse Date if needed
    # df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

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
    for the entire region.
    Returns a DataFrame with columns [date, Region_HDD].
    """
    total_pop = sum(CITY_INFO[city]["population"] for city in city_forecasts)
    if total_pop == 0:
        print("Warning: total population is 0; using equal weighting.")
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

# -----------------------------
# 3b. Gas Weighted Degree Days
# -----------------------------

def calculate_gwdd(region_forecast_dfs, region_noaa_avgs):
    """
    1) For each region, take the 'Region_HDD' forecast.
    2) Multiply by the region's NOAA 1-year average HDD (weighted) to get 'Region_GWDD'.
    3) Merge all region frames on 'date', sum across 'Region_GWDD' for a total 'GWDD'.
    """
    df_list = []
    for region_name, forecast_df in region_forecast_dfs.items():
        # Copy to avoid changing the original
        tmp_df = forecast_df.copy()
        # Region's 1-year Weighted HDD
        noaa_weight = region_noaa_avgs.get(region_name, 0.0)
        # Gas-weighted HDD for that region
        tmp_df[f"{region_name}_GWDD"] = tmp_df["Region_HDD"] * noaa_weight
        df_list.append(tmp_df[["date", f"{region_name}_GWDD"]])

    # Merge them all into one DataFrame by 'date'
    gwdd_df = df_list[0]
    for df in df_list[1:]:
        gwdd_df = pd.merge(gwdd_df, df, on="date", how="outer")

    # Sum across each region's GWDD column
    gwdd_cols = [col for col in gwdd_df.columns if col.endswith("_GWDD")]
    gwdd_df["GWDD"] = gwdd_df[gwdd_cols].sum(axis=1)

    # Sort by date if needed
    gwdd_df.sort_values("date", inplace=True)
    return gwdd_df[["date", "GWDD"]]

# --------------------------------
# 4. Heuristic & Signal Generation
# --------------------------------

def generate_signals(region_hdd_df, region_name, noaa_avg_hdd, ng_price_df, eia_storage):
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

    # NG price trend from the IB data
    if not ng_price_df.empty:
        ng_price_df = ng_price_df.reset_index(drop=True).sort_values("Date")
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

# ----------------------------
# 5. Main Orchestrating Logic
# ----------------------------

def main():
    print("Starting Heating Degree Days analysis and NG signal generation...")

    # 1) Get region-level 1-year avg HDD from NOAA
    print("Retrieving 1-year average HDD from NOAA for each region (weighted by population)...")
    region_noaa_avgs = fetch_noaa_historical_data(REGIONS, years=HISTORICAL_YEARS, noaa_token=NOAA_API_TOKEN)

    # 2) Fetch EIA storage data
    print("Fetching EIA storage data...")
    eia_storage_data = fetch_eia_storage_data()

    # 3) Fetch NG futures data from IB
    print("Fetching NG futures data from IB API...")
    ng_prices_df = fetch_ng_futures_data(None, None)

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

    # 4) For each region, fetch city forecasts and compute region-level HDD
    region_signals = {}
    region_forecast_dfs = {}

    for region_name, cities in REGIONS.items():
        print(f"\nProcessing forecast for region: {region_name} ...")

        city_forecasts = {}
        for city in cities:
            lat, lon = CITY_COORDS[city]
            forecast_df = fetch_forecast_temperatures(lat, lon, days=FORECAST_DAYS)
            city_forecasts[city] = forecast_df

        # Aggregate city forecasts into a region-level HDD forecast
        region_hdd_df = aggregate_region_hdd(city_forecasts)

        # Generate signals for each region
        noaa_avg = region_noaa_avgs.get(region_name, 0)
        region_signal = generate_signals(
            region_hdd_df, region_name, noaa_avg, ng_prices_df, eia_storage_data
        )

        region_signals[region_name] = region_signal
        region_forecast_dfs[region_name] = region_hdd_df

    # 5) Calculate total Gas Weighted Degree Days across all regions
    gwdd_df = calculate_gwdd(region_forecast_dfs, region_noaa_avgs)

    # 6) Print out results
    print("\n=== NOAA 1-Year Avg HDD (Weighted) by Region ===")
    for region_name, val in region_noaa_avgs.items():
        print(f"{region_name}: {val:.2f}")

    print("\n=== 7-Day Forecast HDDs & Signals Per Region ===")
    for region, signals in region_signals.items():
        print(f"\nRegion: {region}")
        print(f"Forecasted HDD Data:\n{region_forecast_dfs[region].to_string(index=False)}")
        print(f"Signals: {signals}")

    print("\n=== Gas Weighted Degree Days (GWDD) ===")
    print(gwdd_df.to_string(index=False))
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()

