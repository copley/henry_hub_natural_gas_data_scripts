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
        date = record['date'][:10]  # Format: YYYY-MM-DD
        datatype = record['datatype']
        value = record['value']
        daily_temps[date][datatype] = value
    return daily_temps

def convert_temperature(value_c_tenths):
    """
    Convert a temperature reading (in tenths of °C) to both Fahrenheit and °C.
    Returns a tuple: (temp_f, temp_c)
    """
    temp_c = value_c_tenths / 10.0
    temp_f = (temp_c * 9/5) + 32
    return temp_f, temp_c

def calculate_heating_degree_days(daily_temps):
    """
    For each day, compute the average temperature from TMAX and TMIN,
    then calculate Heating Degree Days (HDD) as: HDD = max(0, 65 - avg_temp_F).
    (65°F is a common base temperature for heating.)
    """
    total_hdd = 0
    daily_hdd = {}
    for date, temps in daily_temps.items():
        if 'TMAX' in temps and 'TMIN' in temps:
            tmax_f, _ = convert_temperature(temps['TMAX'])
            tmin_f, _ = convert_temperature(temps['TMIN'])
            avg_temp_f = (tmax_f + tmin_f) / 2.0
            hdd = max(0, 65 - avg_temp_f)
            daily_hdd[date] = hdd
            total_hdd += hdd
        else:
            print(f"Missing temperature data for {date}. Skipping this day.")
    return total_hdd, daily_hdd

def calculate_market_move(avg_hdd, baseline=15, sensitivity=0.2):
    """
    Calculate an estimated market move in cents per MMBtu based on HDD.
    
    - baseline: a 'neutral' average HDD value (default 15).
    - sensitivity: estimated cents per MMBtu movement per HDD point deviation (default 0.2).
    
    For example, if the average HDD is 20.85, then the difference is 5.85, which
    translates to an estimated move of 5.85 * 0.2 ≈ 1.17 cents per MMBtu.
    """
    delta_hdd = avg_hdd - baseline
    move = delta_hdd * sensitivity
    return move

def print_temperature_details(daily_temps):
    """Print TMAX and TMIN for each day with both Fahrenheit and Celsius values."""
    print("Daily Temperature Details (TMAX and TMIN):")
    for date in sorted(daily_temps):
        temps = daily_temps[date]
        if 'TMAX' in temps and 'TMIN' in temps:
            tmax_f, tmax_c = convert_temperature(temps['TMAX'])
            tmin_f, tmin_c = convert_temperature(temps['TMIN'])
            print(f"  {date}:")
            print(f"    TMAX: {tmax_f:.2f}°F ({tmax_c:.2f}°C)")
            print(f"    TMIN: {tmin_f:.2f}°F ({tmin_c:.2f}°C)")
        else:
            print(f"  {date}: Incomplete temperature data.")

def print_interpretation_text(market_move):
    """
    Print the interpretation text with examples.
    This text explains what a market move of +X cents means on an MHNG chart.
    """
    print("""
Interpretation on a Monthly MHNG Chart (Revised Examples)

If the Current Price Is $3.35 per MMBtu
  In “cents” terms, $3.35 = 335.0 cents.
  A move of +1.17 cents would take you from 335.0 cents to 336.17 cents.
  Converting back to dollars, that’s $3.3617 per MMBtu.
  Relative to $3.35, that’s an increase of roughly 0.35%.

If the Current Price Is $4.00 per MMBtu
  $4.00 = 400.0 cents.
  Adding +1.17 cents brings it to 401.17 cents, or $4.0117.
  That’s a small absolute move, but it indicates mild upward pressure 
  due to colder-than-expected weather.
  At such a high price level, +1.17 cents is a very small fraction of the total (0.0585%).

Why Are We Talking About “Cents”?
  Futures markets often reference natural gas prices in dollars per MMBtu (e.g., $3.35).
  A “cent” move in this context refers to 1 cent per MMBtu = $0.01 per MMBtu.
  So, if your chart shows $3.35, that is 335 cents per MMBtu. 
  A 1.17‑cent move would add $0.0117 to $3.35, giving $3.3617.

Summary
  A +1.17 cent move represents a $0.0117 increase per MMBtu.
  At lower prices (like $3–$4), that’s around a 0.3% shift—enough to note, but not huge.
  
Context Matters: This heuristic indicates a slight upward pressure on prices 
from colder weather (higher HDD), but you should still consider storage levels, 
production data, and broader market fundamentals when forming a trading strategy.

Remember: The “1.17 cents” figure comes from the script’s assumption 
that for every Heating Degree Day above (or below) a 15 HDD baseline, 
we see a 0.2-cent move. In reality, the market can be influenced by 
many other variables, so always treat this as one piece of a larger puzzle.
""")

def print_detailed_summary(total_hdd, daily_hdd, daily_temps):
    """Print a comprehensive summary including HDD values, temperature details, weather assessment, market impact, and interpretation."""
    num_days = len(daily_hdd)
    avg_hdd = total_hdd / num_days if num_days > 0 else 0
    market_move = calculate_market_move(avg_hdd)
    
    # Print detailed HDD values
    print("\nDetailed Daily HDD Values:")
    for date in sorted(daily_hdd):
        print(f"  {date}: {daily_hdd[date]:.2f} HDD")
    print(f"\nAverage HDD per day: {avg_hdd:.2f}")
    
    # Print detailed temperature readings
    print("\nDaily Temperature Readings:")
    print_temperature_details(daily_temps)
    
    # Detailed weather assessment
    print("\nDetailed Weather Assessment for US Natural Gas Demand:")
    print(f"  - The total HDD of {total_hdd:.2f} represents the cumulative shortfall in temperature below 65°F over the period.")
    print(f"  - An average HDD of {avg_hdd:.2f} per day indicates that, on average, temperatures were about {65 - avg_hdd:.2f}°F (in terms of the deficit) relative to the 65°F baseline.")
    print("  - Higher HDD values (for example, above 20) indicate significantly colder days, likely leading to higher natural gas consumption for heating.")
    print("  - Lower HDD values suggest milder conditions on those days.")
    
    # Market impact metric
    print("\nMarket Impact Metric:")
    print("  - Our heuristic model uses a baseline of 15 HDD, where conditions are considered neutral.")
    print("  - For each HDD point above (or below) this baseline, we assume an approximate impact of 0.2 cents per MMBtu on natural gas prices.")
    if market_move >= 0:
        print(f"  - With an average HDD of {avg_hdd:.2f}, this equates to an estimated upward move of +{market_move:.2f} cents per MMBtu.")
    else:
        print(f"  - With an average HDD of {avg_hdd:.2f}, this equates to an estimated downward move of {market_move:.2f} cents per MMBtu.")
    
    # Print the interpretation text
    print_interpretation_text(market_move)

def main():
    weather_data = fetch_weather_data()
    if weather_data is None:
        return

    daily_temps = process_weather_data(weather_data)
    total_hdd, daily_hdd = calculate_heating_degree_days(daily_temps)
    
    # Print the complete detailed summary which is generated on the fly
    print_detailed_summary(total_hdd, daily_hdd, daily_temps)

if __name__ == '__main__':
    main()

