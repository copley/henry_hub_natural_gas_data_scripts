import os
import requests
import pandas as pd
import numpy as np
import threading
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import openai  # Make sure your OpenAI API key is configured

# IB API imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

"""
Below is a high-level explanation of what each section of that data represents and how you can interpret it:

1. Region-Level Forecast HDD & CDD
For each region (e.g., Northeast, Midwest, South Central, etc.), you see a table with columns:

date
The forecast date (from your 7-day forecast period).

Region_HDD (Heating Degree Days)
A population-weighted daily measure of how far average temperatures dip below 65°F.

Higher HDD typically means colder weather → higher heating demand.

Region_CDD (Cooling Degree Days)
Similar concept but indicates how far temperatures rise above 65°F.

Higher CDD typically means hotter weather → higher cooling (AC) demand.

In Northeast or Midwest, you see mostly Region_HDD values because it’s cooler, while in Pacific or South Central, you sometimes see more Region_CDD values due to warmer temperatures.

2. Signals
Below each region’s forecast, you see a dictionary of calculated signals. These are heuristics summarizing temperature forecasts, price trends, and storage data into a single “bias” for each region.

hdd_vs_1yr:
Compares the average forecast HDD to the region’s 1-year historical HDD average.
"Bullish" here means the forecast HDD is significantly above the historical average (colder than usual).
"Bearish" would mean much lower HDD than usual (warmer than usual).
"Neutral" indicates no strong deviation from typical HDD.

delta_forecast:
Looks at the 3-day rolling average of forecast HDD and checks how the last day’s rolling average compares to the first day’s.
If it jumps more than a threshold, it’s Bullish (getting colder). If it drops below a threshold, it’s Bearish (warming up). Otherwise, Neutral.

rolling_trend:
Also uses a rolling(3) mean of HDD but specifically checks the difference between the earliest rolling value vs. the latest rolling value. If it’s positive (temps trending colder), you get Bullish, negative is Bearish, and near zero is Neutral.

price_trend:
Checks how the natural gas futures price has moved from the start to the end of the data you retrieved.
Bullish if it increased by more than +2%, Bearish if it fell by more than -2%, else Neutral.

storage:
Uses EIA storage data for that region: if storage is significantly below normal, it’s Bullish for prices (indicating possible shortages). If it’s above normal, it’s Bearish (excess supply). Otherwise, it’s Neutral.

final_signal:
The script sums up how many Bullish vs. Bearish signals you have and picks the overall “bias.”
"Bullish Bias" if bullish signals outnumber bearish ones.
"Bearish Bias" if bearish signals outnumber bullish ones.
"Neutral" if they’re roughly even or all moderate.

3. Interpretation for Each Region
Northeast:
Forecast Region_HDD values are fairly high (18 to 27 HDD), suggesting cold temperatures.
The script labels hdd_vs_1yr as “Bullish” (colder than normal).
However, the day-to-day trend is slightly downward (the last few days from 25 HDD down to 10 HDD), so rolling_trend is “Bearish.”
Storage is “Bullish” (implying below-average storage or deficits).
Price is trending down (“Bearish”), leading to a final_signal of “Neutral.”

Midwest:
Similar story to Northeast: high HDD, but an overall negative rolling trend.
Storage is also “Bullish,” price trend is “Bearish,” netting a final_signal of “Neutral.”

South Central:
There’s a mixture of HDD (some cooler days) and CDD (warmer days).
hdd_vs_1yr is “Neutral” because the average HDD isn’t far off from historical.
Day-to-day forecast isn’t moving drastically (delta_forecast: “Neutral”), but the short-term average is trending down → “Bearish.”
Price is “Bearish,” storage is “Neutral,” so overall “Bearish Bias.”

Mountain:
Moderately high HDD (30 → 2 by the end), but again it’s trending downward.
Storage is “Neutral,” price is “Bearish,” so the net effect is “Bearish Bias.”

Pacific:
Mostly CDD values (0.3 up to around 10.95), meaning it’s quite warm or warming up.
hdd_vs_1yr is “Bearish,” which in this case means it’s less cold than normal (or more warm relative to historical HDD).
That leads to a “Bearish Bias” overall.

4. How You Might Use This Data
Forecasted HDD/CDD gives direct insight into how cold or hot the region is likely to be over the next week, affecting heating or cooling demand.
Signals let you quickly see if conditions favor bullish or bearish pressure on natural gas prices due to:
Weather deviations (colder/warmer than historical).
Changes from early to later forecast days.
Rolling trends.
Storage levels.
Market price movement itself.
This overall helps you decide if natural gas prices might be pushed up or down based on short-term demand, supply, and general market signals.
"""

# ----------------------------
# 1. Configuration & Constants
# ----------------------------

# --- NEW: Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# (Optional) file logging
# file_handler = logging.FileHandler('noaa_coverage.log')
# file_handler.setFormatter(formatter)
# file_handler.setLevel(logging.INFO)
# logger.addHandler(file_handler)
# --- END Logger Setup ---

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

# --------------------------------------
# New Functions: Dynamic Prompt & AI Analysis
# --------------------------------------

def build_ng_market_analysis_prompt(region_forecast_dfs, region_signals, gwdd_df, region_noaa_avgs):
    """
    Builds a dynamic prompt that includes the most recent forecast, signals, GWDD data,
    and NOAA averages. The prompt is designed to instruct the AI to produce a detailed
    analysis report for a natural gas trader.
    """
    prompt = (
        "You are ChatGPT, an expert in natural gas market analysis. I will provide you with "
        "a dataset that includes a 7-day forecast of Heating Degree Days (HDD) and Cooling Degree Days (CDD) "
        "for various regions, along with multiple signals, Gas Weighted Degree Days (GWDD), and NOAA 1-Year Average "
        "HDD & CDD values. Your task is to generate a comprehensive, detailed analysis that a natural gas trader can "
        "use to inform trading decisions.\n\n"

        "Your analysis should include:\n\n"
        "**Regional Breakdown:**\n"
        "- For each region (Northeast, Midwest, South Central, Mountain, Pacific), compare the forecasted HDD and CDD values "
        "against the NOAA 1-year averages.\n"
        "- Highlight significant deviations, trends, or anomalies, and discuss their potential impact on natural gas demand.\n\n"

        "**Signal Evaluation:**\n"
        "- Analyze the various signals provided for each region (e.g., hdd_vs_1yr, delta_forecast, rolling_trend, price_trend, storage, final_signal).\n"
        "- Explain what bullish or bearish signals might imply for each region and for the overall market.\n\n"

        "**Aggregated Analysis with GWDD:**\n"
        "- Evaluate the Gas Weighted Degree Days (GWDD) data over the 7-day period.\n"
        "- Discuss how this aggregate metric relates to the regional forecasts and what it indicates about overall natural gas consumption trends.\n\n"

        "**Trading Implications:**\n"
        "- Based on your analysis, summarize the potential trading impacts, such as expected changes in natural gas prices, demand shifts, and storage implications.\n"
        "- Provide any recommendations or cautions that might be relevant for natural gas traders.\n\n"

        "**Structured Reporting:**\n"
        "- Organize your analysis with clear headers and bullet points where appropriate.\n"
        "- Ensure the report is thorough and detailed, extracting the maximum amount of actionable insights from the provided data.\n\n"

        "### Data Provided:\n\n"
        "=== 7-Day Forecast HDD & CDD & Signals Per Region ===\n\n"
    )
    # Append each region's dynamic forecast and signals
    for region in region_forecast_dfs.keys():
        prompt += f"Region: {region}\n"
        prompt += "Forecasted HDD/CDD:\n"
        forecast_str = region_forecast_dfs[region].to_string(index=False)
        prompt += forecast_str + "\n"
        prompt += f"Signals: {region_signals[region]}\n\n"

    prompt += "=== Gas Weighted Degree Days (GWDD) ===\n"
    prompt += gwdd_df.to_string(index=False) + "\n\n"

    prompt += "=== NOAA 1-Year Avg HDD & CDD (Weighted) by Region ===\n"
    for region, values in region_noaa_avgs.items():
        prompt += f"{region}: HDD={values['hdd']:.2f}, CDD={values['cdd']:.2f}\n"

    prompt += "\n### TASK:\n"
    prompt += "Produce a detailed analysis report that covers all the points mentioned above. Your final output "
    prompt += "should be structured, clear, and actionable for someone trading natural gas.\n"

    return prompt

def get_ai_analysis(prompt):
    response = openai.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
# --- NEW: Coverage-Checking Method ---
def verify_noaa_coverage(station_id, city_data, start_date, end_date):
    """
    Checks how many unique dates we received vs. how many dates were expected
    in the NOAA retrieval period. Logs a warning if there's a shortfall.
    """
    if city_data.empty:
        logger.warning(
            f"No data at all for station {station_id} "
            f"from {start_date} to {end_date}."
        )
        return

    expected_days = (end_date - start_date).days + 1
    actual_days = city_data['date'].nunique()

    if actual_days < expected_days:
        gap = expected_days - actual_days
        logger.warning(
            f"Missing {gap} day(s) of data for station {station_id} "
            f"within {start_date} to {end_date}. "
            f"(Expected {expected_days}, got {actual_days})"
        )
    else:
        logger.info(
            f"Station {station_id}: Found {actual_days} days of data for "
            f"{expected_days} expected days — good coverage."
        )
# --- END Coverage-Checking Method ---

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
    compute average HDD & CDD (base 65F), then aggregate to a region-weighted avg.
    Returns: dict of { region_name: {"hdd": float, "cdd": float} }
    """
    end_date = (datetime.now().date() - timedelta(days=1))
    start_date = (end_date - timedelta(days=365 * years))

    city_avgs = {}  # store avg_hdd, avg_cdd, population per city
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

                    r = requests.get(url, headers=headers, params=params_list)
                    if r.status_code != 200:
                        print(f"\nNOAA API request failed with status {r.status_code}. Response:\n{r.text}\n")
                        break

                    resp_json = r.json()

                    # If no "results", we are done with pagination
                    if "results" not in resp_json or not resp_json["results"]:
                        break

                    results = resp_json["results"]
                    if not results:
                        break

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

                    # Update offset
                    next_offset = current_offset + limit
                    if next_offset > total_count:
                        break
                    else:
                        offset = next_offset

                if chunk_frames:
                    big_chunk_df = pd.concat(chunk_frames, ignore_index=True)
                    frames.append(big_chunk_df)
                else:
                    print(f"No data returned for {city} from {current_start} to {chunk_end}")

                current_start = chunk_end

            # Combine all chunks for this city
            if frames:
                city_data = pd.concat(frames, ignore_index=True)

                # (2) Verify coverage
                verify_noaa_coverage(station_id, city_data, start_date, end_date)

                # (3) Check partial missing TMIN/TMAX rows (only if columns exist)
                if "TMIN" in city_data.columns and "TMAX" in city_data.columns:
                    missing_tmin = city_data["TMIN"].isna().sum()
                    missing_tmax = city_data["TMAX"].isna().sum()
                    if missing_tmin > 0 or missing_tmax > 0:
                        logger.warning(
                            f"{station_id} has {missing_tmin} missing TMIN and "
                            f"{missing_tmax} missing TMAX rows."
                        )
            else:
                print(f"No historical data returned for {city}.")
                city_avgs[city] = {"avg_hdd": 0, "avg_cdd": 0, "population": population}
                continue

            # (4) Check if TMIN/TMAX exist at all
            if "TMIN" not in city_data.columns or "TMAX" not in city_data.columns:
                print(f"Missing TMIN/TMAX data for {city}. Setting HDD & CDD to 0.")
                city_avgs[city] = {"avg_hdd": 0, "avg_cdd": 0, "population": population}
                continue

            # (5) Now compute HDD & CDD
            city_data["TAVG"] = (city_data["TMAX"] + city_data["TMIN"]) / 2.0
            city_data["HDD"] = np.maximum(0, HDD_BASE_TEMP - city_data["TAVG"])
            city_data["CDD"] = np.maximum(0, city_data["TAVG"] - HDD_BASE_TEMP)

            avg_hdd = city_data["HDD"].mean()
            avg_cdd = city_data["CDD"].mean()
            city_avgs[city] = {
                "avg_hdd": avg_hdd,
                "avg_cdd": avg_cdd,
                "population": population
            }

    # Aggregate city-level to region-level, weighting by population
    region_averages = {}
    for region_name, cities in regions_dict.items():
        total_pop = sum(city_avgs[c]["population"] for c in cities)
        if total_pop == 0:
            region_averages[region_name] = {"hdd": 0.0, "cdd": 0.0}
            continue

        # Weighted average
        weighted_hdd = 0
        weighted_cdd = 0
        for c in cities:
            pop = city_avgs[c]["population"]
            weighted_hdd += city_avgs[c]["avg_hdd"] * (pop / total_pop)
            weighted_cdd += city_avgs[c]["avg_cdd"] * (pop / total_pop)

        region_averages[region_name] = {"hdd": weighted_hdd, "cdd": weighted_cdd}

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
    # Example date/symbol - adjust to a valid future
    contract.lastTradeDateOrContractMonth = "20250428"
    contract.localSymbol = "NGK5"  # e.g. June 2025 contract
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

def calculate_hdd(temp_min, temp_max):
    """Calculate daily HDD from min/max using base 65F."""
    avg_temp = (temp_min + temp_max) / 2.0
    return max(0, HDD_BASE_TEMP - avg_temp)

def calculate_cdd(temp_min, temp_max):
    """Calculate daily CDD from min/max using base 65F."""
    avg_temp = (temp_min + temp_max) / 2.0
    return max(0, avg_temp - HDD_BASE_TEMP)

def aggregate_region_weather(city_forecasts):
    """
    Given a dict of city->forecastDF, compute population-weighted HDD and CDD
    for the entire region. Returns a DataFrame with [date, Region_HDD, Region_CDD].
    """
    total_pop = sum(CITY_INFO[city]["population"] for city in city_forecasts)
    if total_pop == 0:
        print("Warning: total population is 0; using equal weighting for HDD and CDD.")
        city_count = len(city_forecasts)
        city_dfs = []
        for city, df in city_forecasts.items():
            df = df.copy()
            df["HDD"] = df.apply(lambda row: calculate_hdd(row["temp_min"], row["temp_max"]), axis=1)
            df["CDD"] = df.apply(lambda row: calculate_cdd(row["temp_min"], row["temp_max"]), axis=1)
            # Equal weighting
            df["Weighted_HDD"] = df["HDD"] / city_count
            df["Weighted_CDD"] = df["CDD"] / city_count
            city_dfs.append(df[["date", "Weighted_HDD", "Weighted_CDD"]].rename(
                columns={"Weighted_HDD": f"{city}_WeightedHDD", "Weighted_CDD": f"{city}_WeightedCDD"}))

        merged_df = city_dfs[0]
        for df_ in city_dfs[1:]:
            merged_df = pd.merge(merged_df, df_, on="date", how="outer")
        hdd_cols = [col for col in merged_df.columns if col.endswith("_WeightedHDD")]
        cdd_cols = [col for col in merged_df.columns if col.endswith("_WeightedCDD")]
        merged_df["Region_HDD"] = merged_df[hdd_cols].sum(axis=1)
        merged_df["Region_CDD"] = merged_df[cdd_cols].sum(axis=1)
        return merged_df[["date", "Region_HDD", "Region_CDD"]]

    # Otherwise, population-weighted approach
    city_dfs = []
    for city, df in city_forecasts.items():
        df = df.copy()
        df["HDD"] = df.apply(lambda row: calculate_hdd(row["temp_min"], row["temp_max"]), axis=1)
        df["CDD"] = df.apply(lambda row: calculate_cdd(row["temp_min"], row["temp_max"]), axis=1)
        city_pop = CITY_INFO[city]["population"]
        weight = city_pop / total_pop
        df["Weighted_HDD"] = df["HDD"] * weight
        df["Weighted_CDD"] = df["CDD"] * weight

        city_dfs.append(df[["date", "Weighted_HDD", "Weighted_CDD"]].rename(
            columns={"Weighted_HDD": f"{city}_WeightedHDD", "Weighted_CDD": f"{city}_WeightedCDD"}))

    merged_df = city_dfs[0]
    for df_ in city_dfs[1:]:
        merged_df = pd.merge(merged_df, df_, on="date", how="outer")

    hdd_cols = [col for col in merged_df.columns if col.endswith("_WeightedHDD")]
    cdd_cols = [col for col in merged_df.columns if col.endswith("_WeightedCDD")]
    merged_df["Region_HDD"] = merged_df[hdd_cols].sum(axis=1)
    merged_df["Region_CDD"] = merged_df[cdd_cols].sum(axis=1)
    return merged_df[["date", "Region_HDD", "Region_CDD"]]

# -----------------------------
# 3b. Gas Weighted Degree Days
# -----------------------------

def calculate_gwdd(region_forecast_dfs, region_noaa_avgs):
    """
    Now uses both HDD & CDD in the GWDD calculation:
      GWDD = Region_HDD * noaa_hdd + Region_CDD * noaa_cdd
    """
    df_list = []
    for region_name, forecast_df in region_forecast_dfs.items():
        tmp_df = forecast_df.copy()
        noaa_hdd = region_noaa_avgs.get(region_name, {}).get("hdd", 0.0)
        noaa_cdd = region_noaa_avgs.get(region_name, {}).get("cdd", 0.0)

        # Updated GWDD calculation using both HDD & CDD
        tmp_df[f"{region_name}_GWDD"] = (
            tmp_df["Region_HDD"] * noaa_hdd + tmp_df["Region_CDD"] * noaa_cdd
        )
        df_list.append(tmp_df[["date", f"{region_name}_GWDD"]])

    # Merge them all into one DataFrame
    gwdd_df = df_list[0]
    for df_ in df_list[1:]:
        gwdd_df = pd.merge(gwdd_df, df_, on="date", how="outer")

    gwdd_cols = [col for col in gwdd_df.columns if col.endswith("_GWDD")]
    gwdd_df["GWDD"] = gwdd_df[gwdd_cols].sum(axis=1)
    gwdd_df.sort_values("date", inplace=True)
    return gwdd_df[["date", "GWDD"]]

# --------------------------------
# 4. Heuristic & Signal Generation
# --------------------------------

def generate_signals(region_weather_df, region_name, noaa_avg_hdd, ng_price_df, eia_storage):
    """
    Generates example signals for a region. We've updated the day-to-day 'delta_forecast'
    to use a rolling(3) mean, and we copy the DataFrame before adding columns to
    avoid SettingWithCopyWarning.
    """
    signals = {}

    # 1) Compare average forecast HDD vs NOAA historical average
    avg_forecast_hdd = region_weather_df["Region_HDD"].mean()
    avg_diff = avg_forecast_hdd - noaa_avg_hdd
    if avg_diff > 2.0:
        signals["hdd_vs_1yr"] = "Bullish"
    elif avg_diff < -2.0:
        signals["hdd_vs_1yr"] = "Bearish"
    else:
        signals["hdd_vs_1yr"] = "Neutral"

    # 2) Delta between first & last forecast day, now using rolling(3) mean
    rolling_mean = region_weather_df["Region_HDD"].rolling(3).mean()
    first_day = rolling_mean.iloc[0]
    last_day = rolling_mean.iloc[-1]
    delta = last_day - first_day
    perc_change = (delta / first_day * 100) if first_day != 0 else 0

    if perc_change > 15:
        signals["delta_forecast"] = "Bullish"
    elif perc_change < -10:
        signals["delta_forecast"] = "Bearish"
    else:
        signals["delta_forecast"] = "Neutral"

    # 3) Rolling 3-day average trend (copy the DataFrame to avoid warnings)
    region_weather_df = region_weather_df.copy()
    region_weather_df["rolling_trend"] = region_weather_df["Region_HDD"].rolling(window=3, min_periods=1).mean()
    trend_delta = region_weather_df["rolling_trend"].iloc[-1] - region_weather_df["rolling_trend"].iloc[0]
    if trend_delta > 0:
        signals["rolling_trend"] = "Bullish"
    elif trend_delta < 0:
        signals["rolling_trend"] = "Bearish"
    else:
        signals["rolling_trend"] = "Neutral"

    # 4) NG price trend from the IB data
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

    # 5) Simple storage-based signal
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
    print("Starting Heating/Cooling Degree Days analysis and NG signal generation...")

    # 1) Get region-level 1-year avg HDD & CDD from NOAA
    print("Retrieving NOAA 1-year avg HDD & CDD for each region (weighted by population)...")
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

    # 4) For each region, fetch city forecasts, compute region-level HDD & CDD
    region_signals = {}
    region_forecast_dfs = {}

    for region_name, cities in REGIONS.items():
        print(f"\nProcessing forecast for region: {region_name} ...")

        city_forecasts = {}
        for city in cities:
            lat, lon = CITY_COORDS[city]
            forecast_df = fetch_forecast_temperatures(lat, lon, days=FORECAST_DAYS)
            city_forecasts[city] = forecast_df

        # Aggregate city forecasts -> Region-level HDD & CDD
        region_weather_df = aggregate_region_weather(city_forecasts)

        # Generate signals (now includes the rolling mean improvement for delta)
        noaa_avg_hdd = region_noaa_avgs.get(region_name, {}).get("hdd", 0.0)
        region_signal = generate_signals(
            region_weather_df, region_name, noaa_avg_hdd, ng_prices_df, eia_storage_data
        )

        region_signals[region_name] = region_signal
        region_forecast_dfs[region_name] = region_weather_df

    # 5) Calculate total Gas Weighted Degree Days across all regions
    gwdd_df = calculate_gwdd(region_forecast_dfs, region_noaa_avgs)

    # 6) Print out results
    print("\n=== 7-Day Forecast HDD & CDD & Signals Per Region ===")
    for region, signals in region_signals.items():
        print(f"\nRegion: {region}")
        print("Forecasted HDD/CDD:")
        print(region_forecast_dfs[region].to_string(index=False))
        print(f"Signals: {signals}")

    print("\n=== Gas Weighted Degree Days (GWDD) ===")
    print(gwdd_df.to_string(index=False))
    print("\nAnalysis complete.")
    print("\n=== NOAA 1-Year Avg HDD & CDD (Weighted) by Region ===")
    for region_name, val_dict in region_noaa_avgs.items():
        print(f"{region_name}: HDD={val_dict['hdd']:.2f}, CDD={val_dict['cdd']:.2f}")

    # -------------------------------
    # Build dynamic prompt and get AI analysis
    # -------------------------------
    prompt = build_ng_market_analysis_prompt(region_forecast_dfs, region_signals, gwdd_df, region_noaa_avgs)
    analysis_report = get_ai_analysis(prompt)
    print("\nNatural Gas Market Analysis Report:")
    print(analysis_report)

if __name__ == "__main__":
    main()


