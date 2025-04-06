#!/usr/bin/env python3
"""
📊 Natural Gas Monthly Consumption Summary Tool
----------------------------------------------

This script queries the U.S. Energy Information Administration (EIA) API
to retrieve natural gas consumption statistics by state and by sector
for a specified date range (default: January to December 2024).

🔍 What It Does:
---------------
- Connects to the EIA's `/v2/natural-gas/cons/sum/data/` endpoint.
- Retrieves monthly natural gas consumption data (in MMcf = million cubic feet).
- Filters data based on the defined start and end months (currently: 2024-01 to 2024-12).
- Outputs each data point as:
  "State + Sector + Consumption Description: Value MMcf"

✅ What the Output Includes:
---------------------------
- Residential, Commercial, Industrial, and Electric Power sectors
- U.S. total consumption and by-state breakdowns
- Special consumer categories like:
    - Vehicle fuel usage
    - Lease and plant fuel
    - Pipeline and distribution use

⚠️ Output may include 'None' values where data is missing or delayed.

💡 Units Explained:
-------------------
- MMcf = Million Cubic Feet of natural gas
- These values are used widely in industry for analyzing trends, demand, and energy usage.

🛠️ Requirements:
----------------
- Environment variable `EIA_API_KEY` must be set (e.g. in a `.env` file).
- Python package `requests` for HTTP calls.
- Python package `dotenv` for loading environment variables.

💬 Example Use Cases:
---------------------
- Compare gas usage across states/sectors
- Identify seasonal demand spikes
- Analyze trends in vehicle or electric power gas consumption
- Integrate into dashboards or analytics pipelines

"""


"""
Natural Gas Monthly Consumption Summary (by sector/state)

This script connects to the EIA API to retrieve U.S. natural gas
consumption data for a specific month (default: Dec 2024).
"""
#!/usr/bin/env python3
"""
Natural Gas Consumption Bias Script
-----------------------------------

This script queries the U.S. Energy Information Administration (EIA) API
to retrieve monthly natural gas consumption data (in MMcf). It then:
1. Structures the data in Python objects (list of dicts).
2. Aggregates consumption by month across all states/sectors.
3. Compares the two latest months to generate a simple
   Bullish/Bearish/Neutral bias based on % change in consumption.

Environment:
  - Requires an EIA_API_KEY environment variable set,
    e.g., in a .env file or your shell.

Author: Your Name
"""
#!/usr/bin/env python3
"""
Natural Gas Consumption Bias Script - Updated
---------------------------------------------

Pulls monthly natural gas consumption data from the EIA for 2024.
Accumulates sums by 'period' (YYYY-MM). Compares the two most recent
months that actually have numeric values to generate a Bullish/Bearish
bias. Prints debug info so you can see what months were recognized.

"""

import os
import requests
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()


def get_ng_consumption_data():
    """
    Retrieves monthly U.S. Natural Gas consumption data from EIA
    for 2024-01 to 2024-12. Returns a list of dicts:
      [
         { "period": "YYYY-MM", "description": "...", "value": float or None, "units": "MMcf" },
         ...
      ]
    """
    EIA_API_KEY = os.getenv('EIA_API_KEY')
    if not EIA_API_KEY:
        raise ValueError("ERROR: EIA_API_KEY environment variable not set.")

    url = "https://api.eia.gov/v2/natural-gas/cons/sum/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "monthly",
        "data[0]": "value",
        "start": "2024-01",
        "end":   "2024-12",
        # We’ll request in descending order, but we’ll re-sort in code anyway:
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    result = response.json()

    # Validate structure
    if "response" not in result or "data" not in result["response"]:
        print("No data returned.")
        return []

    raw_list = result["response"]["data"]
    data_list = []
    for row in raw_list:
        desc   = row.get("series-description", "Unknown")
        value  = row.get("value")  # numeric or None
        units  = row.get("units", "MMcf")
        period = row.get("period") # e.g. "2024-11"

        data_list.append({
            "period": period,
            "description": desc,
            "value": value,
            "units": units
        })

    return data_list


def generate_trading_bias(consumption_data, threshold=5.0):
    """
    Generates a simple Bullish/Bearish/Neutral bias from
    month-over-month % change in total consumption.

    1) Sum consumption by 'period' for all rows with a numeric value.
    2) Sort months ascending: 2024-01, 2024-02, ...
    3) Look at the *two most recent* (largest) months.
    4) If % change >= +threshold => Bullish
       If % change <= -threshold => Bearish
       Else => Neutral

    If there's only 1 or 0 months recognized, returns a message.
    """
    if not consumption_data:
        return "No data at all"

    # Summation by month
    monthly_sums = defaultdict(float)
    for row in consumption_data:
        period = row["period"]
        val    = row["value"]
        if isinstance(val, (int, float)):
            monthly_sums[period] += val

    if not monthly_sums:
        return "No numeric data found"

    # Sort months ascending by period string
    # Example periods: '2024-01', '2024-02' -> standard lexicographical sort works fine
    sorted_months = sorted(monthly_sums.items(), key=lambda x: x[0])  # list of (period, sum) tuples

    print("\n[DEBUG] Found these month sums:")
    for (prd, total) in sorted_months:
        print(f"   {prd} => {int(total):,} MMcf")

    if len(sorted_months) < 2:
        return "Not enough monthly data to determine trend"

    # The last two (largest) months are at the end
    prev_period, prev_val = sorted_months[-2]
    last_period, last_val = sorted_months[-1]

    # Avoid dividing by zero
    if abs(prev_val) < 1e-9:
        return f"No prior month volume (period {prev_period} is zero)"

    pct_change = ((last_val - prev_val) / prev_val) * 100.0

    if pct_change >= threshold:
        return f"Bullish (+{pct_change:.1f}% from {prev_period} to {last_period})"
    elif pct_change <= -threshold:
        return f"Bearish ({pct_change:.1f}% from {prev_period} to {last_period})"
    else:
        return f"Neutral ({pct_change:.1f}% from {prev_period} to {last_period})"


def main():
    # 1) Fetch data
    consumption_data = get_ng_consumption_data()
    if not consumption_data:
        print("No data or request failed.")
        return

    # 2) Print raw consumption data (optional, comment out if too big)
    print("\n=== EIA Natural Gas Consumption Data (Jan–Dec 2024) ===")
    for row in consumption_data:
        val_str = f"{row['value']:,}" if isinstance(row['value'], (int, float)) else str(row['value'])
        print(f"  {row['period']} | {row['description']}: {val_str} {row['units']}")

    # 3) Generate bias
    bias = generate_trading_bias(consumption_data, threshold=5.0)
    print(f"\n==> Trading Bias Based on Consumption Trend: {bias}")


if __name__ == "__main__":
    main()
