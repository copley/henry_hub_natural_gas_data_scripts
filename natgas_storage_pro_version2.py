#!/usr/bin/env python3

import os
import requests
import json
from collections import defaultdict

def main():
    # Read token from environment
    EIA_TOKEN = os.environ.get("EIA_TOKEN")
    if not EIA_TOKEN:
        print("ERROR: EIA_TOKEN environment variable not set.")
        print("Please set it with:")
        print('  export EIA_TOKEN="YOUR_API_KEY_HERE"')
        return

    # EIA API base URL
    base_url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    # Query parameters
    params = {
        "frequency": "weekly",
        "data[0]": "value",
        "start": "2025-03-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
        "api_key": EIA_TOKEN
    }

    headers = {
        "X-Params": json.dumps({
            "frequency": "weekly",
            "data": ["value"],
            "facets": {},
            "start": "2025-03-01",
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
            for description, value in sorted(grouped[period], key=lambda x: x[0]):
                print(f"  - {description}: {value} BCF")

    except requests.exceptions.RequestException as e:
        print("Request failed:")
        print(e)

if __name__ == "__main__":
    main()

