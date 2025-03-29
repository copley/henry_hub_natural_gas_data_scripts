#!/usr/bin/env python3

"""
📦 This script pulls data from the EIA's Weekly Natural Gas Storage report via API.

What this script does:
- Connects to the EIA API (https://api.eia.gov)
- Retrieves **weekly U.S. underground natural gas storage levels**
- Starts from a user-defined date (currently set to 2025-03-01)
- Groups the storage data by weekly period (e.g., March 21, March 14, etc.)
- Displays reported values in **BCF (Billion Cubic Feet)**
- Sorts storage data alphabetically by region/description within each week

What this data includes:
- Weekly working gas in underground storage
- Regions such as:
    - East, Midwest, Mountain, Pacific, South Central, and Lower 48 States
    - Salt and Non-Salt storage breakdowns
- Reported values match what’s in the official **Weekly Natural Gas Storage Report** published by the EIA every Thursday

What this script does NOT include (but could be added):
- Historical comparisons (e.g., same week last year)
- Week-over-week changes
- Charts or graphs
- Export to CSV/Excel
- Price or production data

✅ Summary:
This script is a direct line to the EIA’s weekly storage data. It prints a clear summary of how much gas is stored underground in each region of the U.S., broken down by week.

🛠️ You can expand this to:
- Calculate weekly changes or YoY comparisons
- Visualize storage levels
- Filter to show only specific regions
- Add recent prices or weather overlays
"""

import os
import requests
import json
from collections import defaultdict
from dotenv import load_dotenv

def sum_weekly_scores(data_list):
    """
    Sums all BCF values from a list of tuples (description, value).
    """
    total = 0
    for description, value in data_list:
        try:
            total += float(value)
        except ValueError:
            print(f"Warning: Unable to convert {value} to a float for {description}. Skipping...")
    return total

def main():
    # Load environment variables from the .env file
    load_dotenv()

    # Read token from environment
    EIA_API_KEY = os.getenv('EIA_API_KEY')
    if not EIA_API_KEY:
        print("ERROR: EIA_API_KEY environment variable not set.")
        print("Please set it with:")
        print('  export EIA_API_KEY="YOUR_API_KEY_HERE"')
        return

    # EIA API base URL
    base_url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    # Query parameters
    params = {
        "frequency": "weekly",
        "data[0]": "value",
        "start": "2024-12-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
        "api_key": EIA_API_KEY
    }

    headers = {
        "X-Params": json.dumps({
            "frequency": "weekly",
            "data": ["value"],
            "facets": {},
            "start": "2024-12-01",
            "end": None,
            "sort": [
                {"column": "period", "direction": "desc"}
            ],
            "offset": 0,
            "length": 5000
        })
    }

    try:
        # Make the GET request
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Organize data by period
        grouped = defaultdict(list)
        for entry in data["response"]["data"]:
            period = entry["period"]
            description = entry["series-description"]
            value = entry["value"]
            grouped[period].append((description, value))

        # Print sorted output
        for period in sorted(grouped.keys(), reverse=True):
            print(f"\n📅 {period}")
            weekly_data = sorted(grouped[period], key=lambda x: x[0])
            for description, value in weekly_data:
                print(f"  - {description}: {value} BCF")
            total = sum_weekly_scores(weekly_data)
            print(f"  Total: {total} BCF")

    except requests.exceptions.RequestException as e:
        print("Request failed:")
        print(e)

if __name__ == "__main__":
    main()
