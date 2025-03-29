import os
import io
import requests
import contextlib
from collections import defaultdict
from dotenv import load_dotenv
import openai

# ---------------- Load Environment Variables ----------------
load_dotenv()
NOAA_API_TOKEN = os.getenv('NOAA_API_TOKEN')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not NOAA_API_TOKEN:
    raise ValueError("NOAA_API_TOKEN not found in the .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in the .env file.")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# ---------------- NOAA API Configuration ----------------
BASE_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
DATASET_ID = 'GHCND'  # Daily Summaries

# Example location and date range
LOCATION_ID = 'FIPS:37'  # e.g., North Carolina
START_DATE = '2025-03-01'
END_DATE = '2025-03-31'
DATATYPE_IDS = ['TMAX', 'TMIN']

params = {
    'datasetid': DATASET_ID,
    'locationid': LOCATION_ID,
    'startdate': START_DATE,
    'enddate': END_DATE,
    'datatypeid': DATATYPE_IDS,
    'limit': 1000
}

headers = {
    'token': NOAA_API_TOKEN
}

def fetch_weather_data():
    """Retrieve weather data from NOAA's CDO API."""
    response = requests.get(BASE_URL, params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None
    return response.json()

def process_weather_data(data):
    """Organize NOAA data by date."""
    daily_temps = defaultdict(dict)
    for record in data.get('results', []):
        date = record['date'][:10]  # YYYY-MM-DD
        datatype = record['datatype']
        value = record['value']
        daily_temps[date][datatype] = value
    return daily_temps

def convert_temperature(value_c_tenths):
    """
    Convert a temperature reading (in tenths of °C) to both °F and °C.
    Returns (temp_f, temp_c).
    """
    temp_c = value_c_tenths / 10.0
    temp_f = (temp_c * 9/5) + 32
    return temp_f, temp_c

def calculate_heating_degree_days(daily_temps):
    """
    Compute Heating Degree Days (HDD) per day, then total.
    HDD = max(0, 65 - avg_temp_F).
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
    For each HDD above/below 'baseline', assume a 0.2-cent move.
    E.g., if avg_hdd=20.85 => (20.85 - 15)*0.2 ≈ +1.17 cents.
    """
    delta_hdd = avg_hdd - baseline
    return delta_hdd * sensitivity

def print_temperature_details(daily_temps):
    """Print TMAX/TMIN for each day in °F and °C."""
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
    """Print the revised interpretation text for the MHNG chart."""
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

def print_weather_implication_rules():
    """Print additional weather implication rules for short-term and medium-term outlooks."""
    print("\n2. Weather")
    print("Short-Term Implication Rules:")
    print("  - Sudden Temperature Swings: Abrupt increases in heating or cooling degree days can drive immediate changes in demand,")
    print("    pushing prices up if supply doesn’t react quickly.")
    print("  - Localized vs. Widespread Weather Events: A cold snap or heatwave concentrated in major demand centers (e.g., Northeast)")
    print("    can boost regional gas usage and spot prices.")
    print("\nMedium-Term Implication Rules:")
    print("  - Seasonal Forecasts: Extended forecasts of colder-than-normal or hotter-than-normal weather patterns raise the")
    print("    expectation of sustained demand, putting upward pressure on future or forward prices.")
    print("  - Climate Anomalies (e.g., El Niño/La Niña): These can alter weather patterns over multiple months. Analysts")
    print("    anticipate changes in gas consumption trends, influencing hedging and investment decisions.")

def print_detailed_summary(total_hdd, daily_hdd, daily_temps):
    """Print summary including HDD values, daily temps, market impact, and interpretation."""
    num_days = len(daily_hdd)
    avg_hdd = total_hdd / num_days if num_days else 0
    market_move = calculate_market_move(avg_hdd)

    # Average temperature deficit (just for display)
    avg_temp_deficit_f = 65 - avg_hdd
    avg_temp_deficit_c = (avg_temp_deficit_f - 32) * 5.0/9.0

    print("\nDetailed Daily HDD Values:")
    for date in sorted(daily_hdd):
        print(f"  {date}: {daily_hdd[date]:.2f} HDD")
    print(f"\nAverage HDD per day: {avg_hdd:.2f}\n")

    print("Daily Temperature Readings:")
    print_temperature_details(daily_temps)

    print("\nDetailed Weather Assessment for US Natural Gas Demand:")
    print(f"  - The total HDD of {total_hdd:.2f} represents the cumulative shortfall in temperature below 65°F (18.33°C).")
    print(f"  - An average HDD of {avg_hdd:.2f} per day => about {avg_temp_deficit_f:.2f}°F ({avg_temp_deficit_c:.2f}°C)")
    print("    relative to the 65°F baseline.")
    print("  - Higher HDD values (e.g., >20) => significantly colder days => higher nat gas consumption for heating.")
    print("  - Lower HDD => milder conditions on those days.")

    print("\nMarket Impact Metric:")
    print("  - Baseline: 15 HDD => neutral.")
    print("  - Each HDD above/below that => ~0.2 cents per MMBtu.")
    if market_move >= 0:
        print(f"  - Average HDD {avg_hdd:.2f} => ~+{market_move:.2f} cents per MMBtu upward pressure.")
    else:
        print(f"  - Average HDD {avg_hdd:.2f} => ~{market_move:.2f} cents per MMBtu downward pressure.")

    print_interpretation_text(market_move)
    print_weather_implication_rules()

def build_ai_prompt(captured_text):
    """
    Build the prompt to send to GPT for analysis.
    """
    prompt = f"""
You are an advanced trading assistant analyzing the MHNG futures contract.

### TASK
1. Provide a trade recommendation (Buy/Sell/Hold).
2. Estimate the probability (as a single integer between 40 and 80) that MHNG will move in the recommended direction over the next 2 weeks based on the provided weather analysis. Search other sources too.
3. Give a brief fundamental justification.

Rules:
- Keep your answer under 100 words total.

Below is the detailed weather analysis captured from the terminal output:

{captured_text}
"""
    return prompt.strip()

# ----------------------------------------------------------------
# NOTE: The crucial part the user requested: use openai.chat.completions.create
# ----------------------------------------------------------------
def get_ai_analysis(prompt):
    """
    Use openai.chat.completions.create(...) exactly as requested.
    """
    # Make sure your library actually supports this syntax. 
    # If you still get the APIRemovedInV1 error, it means your local library 
    # does not match the method name you are using.

    response = openai.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[
            {"role": "system", "content": "You are a helpful trading assistant."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )
    # Return the text from the first choice
    return response.choices[0].message.content.strip()

def main():
    # Fetch NOAA data
    weather_data = fetch_weather_data()
    if weather_data is None:
        return

    # Process daily temps, compute HDD
    daily_temps = process_weather_data(weather_data)
    total_hdd, daily_hdd = calculate_heating_degree_days(daily_temps)

    # Capture output from summary
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        print_detailed_summary(total_hdd, daily_hdd, daily_temps)
    captured_text = output_buffer.getvalue()

    # Also print it to console
    print(captured_text)

    # Build the AI prompt
    ai_prompt = build_ai_prompt(captured_text)

    # Call GPT-4.5-preview using the syntax you demanded
    ai_analysis = get_ai_analysis(ai_prompt)

    print("\nAI Analysis:")
    print(ai_analysis)

if __name__ == '__main__':
    main()

