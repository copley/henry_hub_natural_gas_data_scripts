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

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def main():
    EIA_API_KEY = os.getenv('EIA_API_KEY')
    if not EIA_API_KEY:
        print("ERROR: EIA_API_KEY environment variable not set.")
        return

    url = "https://api.eia.gov/v2/natural-gas/cons/sum/data/"

    params = {
        "api_key": EIA_API_KEY,
        "frequency": "monthly",
        "data[0]": "value",
        "start": "2024-01",
        "end": "2024-12",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()

        if "response" not in result or "data" not in result["response"]:
            print("No data returned.")
            return

        print("\n📅 Natural Gas Consumption Summary (Dec 2024):\n")
        for row in result["response"]["data"]:
            desc = row.get("series-description", "Unknown")
            value = row.get("value")
            units = row.get("units", "MMcf")
            if isinstance(value, (int, float)):
                print(f"  - {desc}: {value:,} {units}")
            else:
                print(f"  - {desc}: {value} {units}")

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

if __name__ == "__main__":
    main()

