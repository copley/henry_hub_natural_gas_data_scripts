📘 Natural Gas Market Python Scripts (EIA API)
This repo contains a set of Python scripts to pull and display U.S. natural gas market data directly from the EIA Open Data API. Each script focuses on a different fundamental area, from production to consumption, storage, and trade.

🔍 Script Descriptions
Filename	Description
ng_monthly_dry_production.py	Retrieves monthly U.S. natural gas production figures (Dry Gas, Gross Withdrawals, Marketed Production). Shows only national-level data for the latest available month. Great for macro supply tracking.
ng_consumption_by_sector.py	Pulls monthly consumption data by sector and state (Residential, Commercial, Industrial, Electric). Helps track regional demand shifts and seasonal patterns.
ng_import_export_summary.py	Fetches import/export data for natural gas by country, state, and point of entry/exit. Gives a comprehensive picture of pipeline and LNG flows in/out of the U.S.
ng_supply_disposition_balance.py	Displays the monthly supply and disposition balance of the U.S. natural gas market, including storage changes, pipeline use, exports, and more. Ideal for understanding overall supply/demand dynamics.
ng_weekly_storage_by_region.py	Pulls weekly underground storage levels by region (East, Midwest, Pacific, etc.). Matches the data in EIA’s Weekly Storage Report. Useful for tracking inventory health and injection/withdrawal cycles.
⚙️ Shared Features
All scripts use the official EIA v2 API

API Key loaded securely via .env file (EIA_API_KEY)

Output is printed cleanly to the terminal with units (MMcf or BCF)

Graceful error handling for timeouts, invalid responses, or missing data

📁 Requirements
Python 3.7+

Install dependencies:

bash
Edit
pip install requests python-dotenv
🧪 Example .env File

env
Edit
EIA_API_KEY=your_actual_eia_api_key_here

