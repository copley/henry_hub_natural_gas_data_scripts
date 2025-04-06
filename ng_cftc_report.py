#!/usr/bin/env python3

import os
import requests
import pandas as pd
from dotenv import load_dotenv

def main():
    """
    Main entry point. Loads .env, tries to fetch COT data from the
    'publicreporting.cftc.gov' open data portal for Natural Gas (NYMEX).
    If that fails, fallback to Nasdaq Data Link with your NASDAQ_API_KEY.
    """
    load_dotenv()
    nasdaq_api_key = os.getenv("NASDAQ_API_KEY")

    # 1) Try the CFTC Public Reporting Open Data Portal
    if not fetch_cftc_public_report():
        # 2) If that fails, fallback to Nasdaq Data Link
        fetch_from_nasdaq(nasdaq_api_key)


def fetch_cftc_public_report():
    """
    Fetch data from the CFTC "Public Reporting" resource for NATURAL GAS - NYMEX.
    For example, '29v7-k7v4' is one Disaggregated CIT FUTURES ONLY resource ID.
    Check the CFTC open data portal for the correct ID.
    """
    print("\n🔍 Trying to fetch from CFTC Public Reporting API...")

    # Example: "Disaggregated CIT FUTURES ONLY – NATURAL GAS – NYMEX"
    resource_id = "29v7-k7v4"

    base_url = f"https://publicreporting.cftc.gov/resource/{resource_id}.json"

    # Try no "$where" or a minimal filter first, to ensure we get data
    params = {
        # '$where': "market_and_exchange_name='NATURAL GAS - NYMEX'",
        '$limit': 1,
        '$order': 'report_date desc'
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.ok:
            data = response.json()
            if data:
                print("✅ SUCCESS! COT from publicreporting.cftc.gov:\n")
                print("🔎 Data (latest record):", data[0])
            else:
                print("⚠️ Received an empty list from CFTC. Possibly no matching records or filter mismatch.")
            return True
        else:
            print(f"❌ FAILED: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ ERROR fetching from publicreporting.cftc.gov: {e}")
        return False


def fetch_from_nasdaq(api_key):
    """
    Fallback to Nasdaq Data Link (ex‐Quandl) for NATURAL GAS COT data.
    Must confirm you have access to 'CFTC/067651_F_ALL' or choose
    a dataset code you can access on your subscription.
    """
    print("\n🔁 Trying fallback to Nasdaq Data Link (Quandl)...")

    if not api_key:
        print("❌ No NASDAQ_API_KEY found in .env. Can't fetch from Nasdaq.")
        return

    # Check your plan for an accessible code:
    dataset_code = "CFTC/067651_F_ALL"
    url = f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}/data.json"

    params = {
        "api_key": api_key,
        "limit": 1
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.ok:
            data = response.json()
            dataset_data = data.get("dataset_data", {})
            records = dataset_data.get("data", [])
            if records:
                print("✅ SUCCESS! NG COT Data from Nasdaq Data Link:\n")
                print("🔎 Latest record:", records[0])
            else:
                print("⚠️ Received no records from Nasdaq. Possibly restricted or empty.")
        else:
            print("❌ FAILED from Nasdaq Data Link:", response.text[:200])
    except Exception as e:
        print("❌ ERROR from Nasdaq Data Link:", str(e))


if __name__ == "__main__":
    main()
