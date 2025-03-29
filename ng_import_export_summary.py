#!/usr/bin/env python3

"""
📦 U.S. Natural Gas Trade Summary (Imports & Exports)

This script pulls monthly data from the EIA API for:
1. Imports by Country
2. Exports by Country
3. Imports & Exports by State
4. Imports by Point of Entry
5. Exports by Point of Exit


✅ What It Does:
- Fetches clean and structured natural gas trade data from multiple EIA endpoints.
- Focuses on a specific month or range of months (default: full year 2024).
- Prints results in a legible format (e.g., grouped summaries showing country, state, or port-level flows).
- Automatically reports if no data is available or if endpoints are down.
- Handles all common errors gracefully (API key issues, connectivity, missing data).

Usage:
- Requires your EIA API Key set in a `.env` file as EIA_API_KEY.
- Prints clean summaries to terminal for December 2024.
- Includes debugging and graceful handling of API errors.

Data shown in Million Cubic Feet (MMcf).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Setup
API_KEY = os.getenv("EIA_API_KEY")
MONTH = "2024-12"
BASE_URLS = {
    "Imports by Country": "https://api.eia.gov/v2/natural-gas/move/impc/data/",
    "Exports by Country": "https://api.eia.gov/v2/natural-gas/move/expc/data/",
    "Imports & Exports by State": "https://api.eia.gov/v2/natural-gas/move/state/data/",
    "Imports by Point of Entry": "https://api.eia.gov/v2/natural-gas/move/poe1/data/",
    "Exports by Point of Exit": "https://api.eia.gov/v2/natural-gas/move/poe2/data/",
}

COMMON_PARAMS = {
    "frequency": "monthly",
    "data[0]": "value",
    "start": "2024-01",
    "end": "2024-12",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": 5000,
    "api_key": API_KEY
}

def fetch_data(name, url):
    print(f"\n🔎 Fetching: {name}")
    try:
        response = requests.get(url, params=COMMON_PARAMS)
        response.raise_for_status()
        data = response.json()
        series = data.get("response", {}).get("data", [])
        if not series:
            print("⚠️ No data available.")
            return

        # Print header
        print(f"\n📅 Period: {MONTH}")
        for entry in series:
            desc = entry.get("series-description", "Unknown")
            val = entry.get("value")
            units = entry.get("units", "MMcf")

            if isinstance(val, (int, float)):
                print(f"  - {desc}: {val:,} {units}")
            else:
                print(f"  - {desc}: {val} {units}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as ex:
        print(f"❗ Unexpected error: {ex}")

def main():
    if not API_KEY:
        print("❌ ERROR: EIA_API_KEY environment variable not set.")
        return

    for label, url in BASE_URLS.items():
        fetch_data(label, url)

if __name__ == "__main__":
    main()

