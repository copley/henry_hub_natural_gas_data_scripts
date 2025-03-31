#!/usr/bin/env python3
"""
End-to-end EIA Weekly Natural Gas Storage Trend Logic

What This Script Does:
----------------------
1) Loads weekly U.S. Lower 48 natural gas storage data from the EIA API (v2).
2) Automatically calculates:
     - Actual Storage Change (most recent vs. prior week)
     - Current Inventory (most recent)
     - Year-ago Inventory (closest matching week last year)
     - 5-year Average Inventory (closest matching weeks for the last 5 years)
3) Stubs the Analyst Expectations for the weekly storage change. (Replace with real data source.)
4) Uses the pseudo-code logic to:
     - Compare actual vs. expected change (Bullish/Bearish/Neutral)
     - Compare current inventory vs. 5-year average
     - Compare current inventory vs. last year
5) Determines a final "Trend" (UP / DOWN / SIDEWAYS).

Requirements:
-------------
- `requests` library for HTTP calls to the EIA API.
- An environment variable `EIA_API_KEY` (or set in code) with your EIA API token.
- Basic Python 3 environment.

Notes:
------
- This script fetches ~6 years of data to handle 5-year comparisons.
- 'Closest date' matching is used for year-ago and each of the 5 years. 
- The thresholds (±5% deviation, ±50 Bcf difference) are from your pseudo-code. 
  Adjust if your strategy demands different sensitivity.
"""

import os
import math
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# 0) Configuration
# ----------------------------------------------------------------------

load_dotenv()  # if you keep your EIA_API_KEY in a .env file
API_KEY = os.getenv('EIA_API_KEY', 'REPLACE_ME_WITH_YOUR_API_KEY')

BASE_URL = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
# We'll fetch data for the last ~6 years to handle yoy and 5-year lookback
START_DATE = "2019-01-01"
END_DATE   = None  # or leave as None for "latest available"

# The "series-description" we want to filter for:
#   e.g. "Working Gas in Underground Storage, Lower 48 states"
TARGET_PHRASE = "lower 48 states"

# ----------------------------------------------------------------------
# 1) Helper: Load EIA Weekly Storage Data
# ----------------------------------------------------------------------

def load_storage_data():
    """
    Fetches weekly natural gas storage data from the EIA 'wkly' endpoint
    for the Lower 48 states only, from START_DATE to END_DATE.
    
    Returns:
        dict: A dictionary { date_obj: float_value_Bcf }
              where date_obj is a datetime.date, 
              and float_value_Bcf is the reported inventory for that week.
    """
    if not API_KEY or API_KEY == 'REPLACE_ME_WITH_YOUR_API_KEY':
        raise ValueError("No valid EIA_API_KEY provided. Please set it in your environment or script.")

    params = {
        "frequency": "weekly",
        "data[0]": "value",       # we want the 'value' field
        "start": START_DATE,
        "end": END_DATE,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
        "api_key": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    result = response.json()

    storage_by_date = {}
    entries = result.get("response", {}).get("data", [])
    for e in entries:
        desc   = e.get("series-description", "").lower()
        period = e.get("period", "")
        val    = e.get("value", None)

        # We only want the "Lower 48" total working gas
        if (TARGET_PHRASE in desc) and (val is not None):
            # Convert "2025-03-21" -> datetime.date(2025, 3, 21)
            try:
                dt = datetime.strptime(period, "%Y-%m-%d").date()
            except ValueError:
                # skip if period not a valid date
                continue
            # Store in dictionary
            storage_by_date[dt] = float(val)

    return storage_by_date


def find_latest_two_weeks(storage_data):
    """
    Returns the two most recent weekly data points from 'storage_data'
      sorted in ascending date order.

    Returns:
        ( (date1, value1), (date2, value2) )
        Where date2 is the latest. 
    """
    if not storage_data:
        return None, None

    # Sort the dictionary keys (dates) descending
    sorted_dates = sorted(storage_data.keys(), reverse=True)
    if len(sorted_dates) < 2:
        raise ValueError("Not enough weekly data to compute the actual storage change (need >= 2).")

    latest_date = sorted_dates[0]
    prev_date   = sorted_dates[1]
    return (prev_date, storage_data[prev_date]), (latest_date, storage_data[latest_date])


def find_closest_week_value(storage_data, target_date, max_delta_days=7):
    """
    Find the storage value for the date in 'storage_data' that is closest
    to 'target_date' (within ±7 days by default).
    Returns the (closest_date, value) or (None, None) if no suitable match.
    """
    if not storage_data:
        return None, None

    best_date = None
    best_val  = None
    min_diff  = timedelta(days=9999)

    for d in storage_data:
        diff = abs(d - target_date)
        if diff < min_diff:
            min_diff = diff
            best_date = d
            best_val  = storage_data[d]

    # Check if the best match is within our max_delta limit
    if best_date and min_diff.days <= max_delta_days:
        return best_date, best_val
    else:
        return None, None


# ----------------------------------------------------------------------
# 2) The Five "Fetch" Functions from Your Pseudo-Code
#    Now with real EIA-based logic + a stub for analyst expectations
# ----------------------------------------------------------------------

# We'll load all data once, at module level, 
# so we don't fetch from the API multiple times.
STORAGE_DATA = load_storage_data()

def fetch_actual_storage_change():
    """
    Return the actual weekly storage injection (+) or withdrawal (–), in Bcf.
    Computed from the difference between the last two available weekly data points.
    """
    (prev_d, prev_val), (latest_d, latest_val) = find_latest_two_weeks(STORAGE_DATA)
    return latest_val - prev_val  # Positive => injection, Negative => withdrawal

def fetch_analyst_expectations():
    """
    Return the consensus forecast (in Bcf).
    EIA does NOT provide consensus forecasts, so this is a STUB.
    Replace with your real data source if available.
    """
    # Example: -100.0 means analysts expect a 100 Bcf withdrawal.
    return -100.0

def fetch_current_inventory():
    """
    Return the current (latest available) total Lower 48 inventory, in Bcf.
    """
    # The most recent date is the first in the sorted list (descending)
    sorted_dates = sorted(STORAGE_DATA.keys(), reverse=True)
    if not sorted_dates:
        raise ValueError("No storage data found to determine current inventory.")
    latest = sorted_dates[0]
    return STORAGE_DATA[latest]

def fetch_five_year_average_inventory():
    """
    Return the approximate 5-year average inventory for the "current week".
    We do this by:
      - Finding the latest data date
      - For each of the past 5 years, find the 'closest date' within ±7 days
      - Average them if found
    """
    sorted_dates = sorted(STORAGE_DATA.keys(), reverse=True)
    if not sorted_dates:
        raise ValueError("No storage data for computing 5-year average.")
    latest_date = sorted_dates[0]

    # We'll accumulate valid matches
    matched_values = []
    for i in range(1, 6):  # 1..5
        past_year = latest_date.replace(year=latest_date.year - i)
        match_d, match_val = find_closest_week_value(STORAGE_DATA, past_year)
        if match_val is not None:
            matched_values.append(match_val)

    if not matched_values:
        # fallback: if we couldn't find any year, just return the latest as a fallback
        return STORAGE_DATA[latest_date]

    return sum(matched_values) / len(matched_values)

def fetch_year_ago_inventory():
    """
    Return the approximate inventory for the same week last year (±7 days).
    """
    # Latest date
    sorted_dates = sorted(STORAGE_DATA.keys(), reverse=True)
    if not sorted_dates:
        raise ValueError("No storage data for computing year-ago inventory.")
    latest_date = sorted_dates[0]

    # Check ~1 year earlier
    year_ago_date = latest_date.replace(year=latest_date.year - 1)
    match_d, match_val = find_closest_week_value(STORAGE_DATA, year_ago_date)
    if match_val is None:
        # If not found, fallback to 2 weeks old, etc. (rarely needed)
        return 0.0
    return match_val


# ----------------------------------------------------------------------
# 3) The Trading Logic from your Pseudo-code
# ----------------------------------------------------------------------

def get_eia_report():
    """
    Retrieves all needed variables: 
      - actual_storage_change
      - expected_storage_change
      - current_inventory
      - five_year_average
      - year_ago_inventory
    """
    actual = fetch_actual_storage_change()
    expected = fetch_analyst_expectations()
    current_inv = fetch_current_inventory()
    five_year_avg = fetch_five_year_average_inventory()
    year_ago_inv = fetch_year_ago_inventory()

    return (actual, expected, current_inv, five_year_avg, year_ago_inv)


def analyze_storage_report(actual, expected, current_inv, five_year_avg, year_ago_inv):
    """
    Applies your logic to produce three sentiment values:
      - sentiment (actual vs. expected)
      - sentiment_5yr (current vs. 5-year avg)
      - sentiment_yoy (current vs. last year)
    """
    difference = actual - expected
    # Avoid division by zero if expected=0
    if abs(expected) < 1e-9:
        deviation_percent = 0
    else:
        deviation_percent = (difference / abs(expected)) * 100

    # Determine sentiment for actual vs. expected
    if deviation_percent <= -5:
        sentiment = "Bullish"    # actual is more negative => bigger withdrawal => bullish
    elif deviation_percent >= 5:
        sentiment = "Bearish"    # actual is more positive => bigger injection => bearish
    else:
        sentiment = "Neutral"

    # Compare current to 5-year average
    inventory_vs_5yr = current_inv - five_year_avg
    if inventory_vs_5yr < -50:
        sentiment_5yr = "Bullish"
    elif inventory_vs_5yr > 50:
        sentiment_5yr = "Bearish"
    else:
        sentiment_5yr = "Neutral"

    # Compare current to year-ago
    inventory_vs_last_year = current_inv - year_ago_inv
    if inventory_vs_last_year < -50:
        sentiment_yoy = "Bullish"
    elif inventory_vs_last_year > 50:
        sentiment_yoy = "Bearish"
    else:
        sentiment_yoy = "Neutral"

    return (sentiment, sentiment_5yr, sentiment_yoy)


def determine_trend(sentiments):
    """
    Final trend direction based on number of Bullish/Bearish signals:
      - If >=2 are "Bullish" => "UP"
      - If >=2 are "Bearish" => "DOWN"
      - Otherwise => "SIDEWAYS"
    """
    bullish_score = sentiments.count("Bullish")
    bearish_score = sentiments.count("Bearish")

    if bullish_score >= 2:
        return "UP"
    elif bearish_score >= 2:
        return "DOWN"
    else:
        return "SIDEWAYS"


# ----------------------------------------------------------------------
# 4) Main script logic
# ----------------------------------------------------------------------

def main():
    # Retrieve all relevant data
    actual, expected, current_inv, five_year_avg, year_ago_inv = get_eia_report()

    # Analyze the storage report
    sentiments = analyze_storage_report(actual, expected, current_inv, five_year_avg, year_ago_inv)
    trend = determine_trend(sentiments)

    # Print results
    print("\n=== EIA Weekly Natural Gas Storage Report Analysis ===")
    print(f"Latest Actual Storage Change (Bcf):  {actual:+.1f}")
    print(f"Analyst Expectation (Bcf):           {expected:+.1f}")
    print(f"Deviation from Expectation (%):      ( {((actual-expected)/abs(expected)*100) if abs(expected)>1e-9 else 0:.1f}% )")
    print("------------------------------------------------------")
    print(f"Current Inventory (Bcf):             {current_inv:,.1f}")
    print(f"5-Year Average Inventory (Bcf):      {five_year_avg:,.1f}")
    print(f"Year-ago Inventory (Bcf):            {year_ago_inv:,.1f}")
    print("------------------------------------------------------")
    print(f"Storage vs 5-Year Avg (Bcf):         {current_inv - five_year_avg:+,.1f}")
    print(f"Storage vs Year-ago (Bcf):           {current_inv - year_ago_inv:+,.1f}")
    print("------------------------------------------------------")

    print(f"Sentiments: {sentiments}")
    print(f"Trend Direction: {trend}")
    print("======================================================\n")


if __name__ == "__main__":
    main()

