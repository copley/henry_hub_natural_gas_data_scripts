#!/usr/bin/env python3

"""
📦 U.S. Monthly Natural Gas Supply & Disposition Balance

This script queries the EIA API for:
1. Macro-level supply and disposition data
   (Production, Imports, Exports, Consumption, Storage, etc.)
2. Line-item summary data (if available)

It retrieves and displays values for a specific month (default: Dec 2024),
organized and printed to the terminal in a clean, readable format.

API Key must be set in a `.env` file as: EIA_API_KEY=your_key_here
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Setup
API_KEY = os.getenv("EIA_API_KEY")
MONTH = "2024-12"

BASE_URLS = {
    "Supply & Disposition Summary": "https://api.eia.gov/v2/natural-gas/sum/sndm/data/",
    "Line-Item Summary": "https://api.eia.gov/v2/natural-gas/sum/lsum/data/"
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
    print(f"\n🔍 Fetching: {name}")
    try:
        response = requests.get(url, params=COMMON_PARAMS)
        response.raise_for_status()
        result = response.json()
        records = result.get("response", {}).get("data", [])

        if not records:
            print("⚠️ No data available.")
            return

        print(f"\n📅 Period: {MONTH}")
        for entry in records:
            desc = entry.get("series-description", "Unknown")
            value = entry.get("value")
            units = entry.get("units", "MMcf")

            if isinstance(value, (int, float)):
                print(f"  - {desc}: {value:,} {units}")
            else:
                print(f"  - {desc}: {value} {units}")
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

