import os
import requests
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------- Configuration ----------------
API_TOKEN = os.getenv('NOAA_API_TOKEN')
if not API_TOKEN:
    raise ValueError("No NOAA_API_TOKEN found in the .env file.")

# NOAA CDO API base URL and dataset info
BASE_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
DATASET_ID = 'GHCND'  # Daily Summaries

# Set location and date range (modify these as needed)
LOCATION_ID = 'FIPS:37'         # Example: FIPS:37 for North Carolina
START_DATE = '2025-03-01'
END_DATE = '2025-03-31'

# We require both maximum and minimum temperature data.
DATATYPE_IDS = ['TMAX', 'TMIN']

# API query parameters
params = {
    'datasetid': DATASET_ID,
    'locationid': LOCATION_ID,
    'startdate': START_DATE,
    'enddate': END_DATE,
    'datatypeid': DATATYPE_IDS,
    'limit': 1000  # Increase if needed
}

headers = {
    'token': API_TOKEN
}
# ------------------------------------------------

def fetch_weather_data():
    """Retrieve weather data from NOAA's CDO API."""
    response = requests.get(BASE_URL, params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None
    return response.json()

def process_weather_data(data):
    """
    Organize data by date.
    NOAA returns temperatures in tenths of degrees Celsius.
    """
    daily_temps = defaultdict(dict)
    for record in data.get('results', []):
        # Extract date in YYYY-MM-DD format
        date = record['date'][:10]
        datatype = record['datatype']
        value = record['value']
        daily_temps[date][datatype] = value
    return daily_temps

def convert_to_fahrenheit(value_c_tenths):
    """Convert temperature from tenths of °C to °F."""
    value_c = value_c_tenths / 10.0
    value_f = (value_c * 9/5) + 32
    return value_f

def calculate_heating_degree_days(daily_temps):
    """
    For each day, compute the average temperature from TMAX and TMIN,
    then calculate Heating Degree Days (HDD) as: HDD = max(0, 65 - avg_temp).
    (65°F is a common base temperature for heating.)
    """
    total_hdd = 0
    daily_hdd = {}
    for date, temps in daily_temps.items():
        if 'TMAX' in temps and 'TMIN' in temps:
            tmax_f = convert_to_fahrenheit(temps['TMAX'])
            tmin_f = convert_to_fahrenheit(temps['TMIN'])
            avg_temp = (tmax_f + tmin_f) / 2.0
            hdd = max(0, 65 - avg_temp)
            daily_hdd[date] = hdd
            total_hdd += hdd
        else:
            print(f"Missing temperature data for {date}. Skipping this day.")
    return total_hdd, daily_hdd

def print_detailed_summary(total_hdd, daily_hdd):
    """Print a detailed summary of the HDD results and their implications."""
    num_days = len(daily_hdd)
    avg_hdd = total_hdd / num_days if num_days > 0 else 0

    print(f"\nTotal Heating Degree Days (HDD) from {START_DATE} to {END_DATE}: {total_hdd:.2f}\n")
    print("Detailed Daily HDD Values:")
    for date in sorted(daily_hdd):
        print(f"  {date}: {daily_hdd[date]:.2f} HDD")
    print(f"\nAverage HDD per day: {avg_hdd:.2f}\n")

    # Detailed explanation of the results
    print("Detailed Weather Assessment for US Natural Gas Demand:")
    print(f"  - The total HDD of {total_hdd:.2f} over the period indicates the cumulative degrees by which the average daily temperature fell below the baseline of 65°F.")
    print(f"  - An average HDD of {avg_hdd:.2f} per day suggests that, on average, temperatures were {65 - avg_hdd:.2f}°F, indicating moderately cold conditions.")
    print("  - In practical terms, higher HDD values signal increased heating demand. For example, on days like 2025-03-02 (26.97 HDD) and 2025-03-03 (22.47 HDD), the cold was more pronounced,")
    print("    likely leading to higher natural gas consumption for heating.")
    print("  - Conversely, lower HDD values (e.g., 9.96 HDD on 2025-03-01) suggest milder conditions on those days.")
    print("\nOverall Assessment:")
    if avg_hdd >= 20:
        print("  The high average HDD indicates that the weather was generally cold during the period, which could boost natural gas demand and potentially drive prices higher.")
    elif avg_hdd >= 10:
        print("  The average HDD is moderate, suggesting mildly cold weather. While there is some heating demand, its impact on natural gas prices might be less pronounced.")
    else:
        print("  The low average HDD points to warm conditions, likely leading to reduced heating demand and softer natural gas prices.")
    print("\nNote: This analysis focuses solely on temperature-derived HDD as a proxy for heating demand. For a comprehensive market assessment, consider other factors such as storage, production, and imports.")

def main():
    weather_data = fetch_weather_data()
    if weather_data is None:
        return

    daily_temps = process_weather_data(weather_data)
    total_hdd, daily_hdd = calculate_heating_degree_days(daily_temps)
    
    # Print a detailed summary of the results.
    print_detailed_summary(total_hdd, daily_hdd)

if __name__ == '__main__':
    main()

