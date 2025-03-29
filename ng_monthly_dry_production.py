#!/usr/bin/env python3

"""
📦 This script pulls data from the EIA Natural Gas Monthly report via API.

✅ EIA production data is typically delayed compared to some other sources.
Here’s a breakdown of why and what alternatives you can use:

🕒 EIA Data Delay — Why It Happens:

1. Accuracy Focus:
   - The EIA prioritizes verified, reported data from operators and government agencies.
   - It does not rely on estimates or real-time sensor feeds.

2. Data Collection Process:
   - Monthly surveys are collected from energy producers.
   - These take time to compile, clean, and validate before release.

3. Publication Lag:
   - Most monthly production data (e.g., dry gas or gross withdrawals) is released
     1.5 to 2 months after the end of the reporting month.
   - Example: 📆 December data → published late February or March.

What this script does:
- Connects to the EIA API (https://api.eia.gov)
- Pulls monthly U.S. natural gas production data
- Filters to show only:
    - Dry Natural Gas Production
    - Marketed Production
    - Gross Withdrawals
- Prints only the latest available month ("front month")
- Displays U.S.-level totals only

What the full Natural Gas Monthly report includes (not covered here):
- State-by-state production
- Consumption by sector (residential, commercial, industrial, electric)
- Imports/Exports
- Underground storage levels
- Prices (wellhead, citygate, residential, etc.)
- LNG volumes and capacity
- Pipeline flows and capacity

✅ So yes:
This script shows just the U.S.-level monthly production part of the NGM — the same figures you'd find in Table 1 of the PDF/Excel report or on the EIA’s data browser.

🛠️ You can expand this to:
- Compare multiple months (e.g., trends or changes)
- Include state-level breakdowns
- Pull in consumption or pricing data
- Export to CSV/JSON or visualize with matplotlib

🔥 This is a great foundation to build your own energy dashboard or analytics pipeline.
"""

import os
import requests
import json

def main():
    # Read token from environment
    EIA_TOKEN = os.environ.get("EIA_TOKEN")
    if not EIA_TOKEN:
        print("ERROR: EIA_TOKEN environment variable not set.")
        return

    # API endpoint
    url = "https://api.eia.gov/v2/natural-gas/prod/sum/data/"

    # Parameters
    params = {
        "frequency": "monthly",
        "data[0]": "value",
        "start": "2024-12",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
        "api_key": EIA_TOKEN
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Get the most recent month (front month)
        latest_period = None
        filtered_entries = []

        for entry in data["response"]["data"]:
            period = entry["period"]
            description = entry["series-description"]
            value = entry["value"]

            # Only include U.S.-level Dry/Marketed/Gross Production
            if (
                description.startswith("U.S. ") and
                any(key in description.lower() for key in ["dry", "marketed", "gross"])
            ):
                if latest_period is None:
                    latest_period = period
                if period == latest_period:
                    filtered_entries.append((description, value))

        # Print the clean, filtered output
        if filtered_entries:
            print(f"\n📅 Front Month: {latest_period}\n")
            for desc, val in sorted(filtered_entries):
                if isinstance(val, (int, float)):
                    print(f"  - {desc}: {val:,} MMcf")
                else:
                    print(f"  - {desc}: {val} MMcf")
        else:
            print("No data found for U.S. dry/marketed/gross gas production.")

    except requests.exceptions.RequestException as e:
        print("Request failed:")
        print(e)

if __name__ == "__main__":
    main()

