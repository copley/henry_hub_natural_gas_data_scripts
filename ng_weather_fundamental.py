import os
import requests
import datetime
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
# Example: FIPS:37 for North Carolina; you may use other location IDs as required.
LOCATION_ID = 'FIPS:37'
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

def assess_weather(total_hdd, num_days):
    """
    Use a simple logic based on average HDD per day.
    Higher HDD suggests colder weather and increased heating demand,
    which can drive up natural gas usage.
    """
    avg_hdd = total_hdd / num_days if num_days > 0 else 0
    print(f"Average HDD per day: {avg_hdd:.2f}")
    
    if avg_hdd >= 20:
        return "Cold weather expected. Increased heating demand may drive up natural gas prices."
    elif avg_hdd >= 10:
        return "Mild to moderately cold weather expected. Monitor trends for potential impact on natural gas demand."
    else:
        return "Warm weather expected. Lower heating demand may lead to softer natural gas prices."

def main():
    weather_data = fetch_weather_data()
    if weather_data is None:
        return

    daily_temps = process_weather_data(weather_data)
    total_hdd, daily_hdd = calculate_heating_degree_days(daily_temps)
    num_days = len(daily_hdd)
    
    print(f"\nTotal Heating Degree Days (HDD) from {START_DATE} to {END_DATE}: {total_hdd:.2f}")
    print("Daily HDD values:")
    for date in sorted(daily_hdd):
        print(f"{date}: {daily_hdd[date]:.2f}")
    
    assessment = assess_weather(total_hdd, num_days)
    print("\nWeather Assessment for US Natural Gas Demand:")
    print(assessment)

if __name__ == '__main__':
    main()

