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

Natural Gas Analytics & Forecasting Suite
This project is a comprehensive toolkit for analyzing and forecasting the natural gas market. It integrates multiple data sources and methodologies—including technical analysis, machine learning, energy fundamentals, and weather-driven insights—to help traders, analysts, and researchers make informed decisions.

Key Features
Technical Analysis & Trade Signals:
Calculate a wide array of technical indicators (moving averages, RSI, stochastics, MACD, etc.) using IB API data and generate trade signals based on proprietary logic (e.g., Rogue Radar thresholds).

AI-Enhanced Forecasting:
Combine technical metrics with AI-driven insights by generating custom prompts for OpenAI’s GPT-4.5-preview model to receive trade recommendations and market analysis.

Energy Consumption & Trade Analysis:
Retrieve and summarize monthly natural gas consumption by sector as well as import/export statistics from the EIA API to understand demand and trade flows.

Production & Supply/Disposition Insights:
Query the EIA API for U.S. monthly dry production data and supply/disposition balances, comparing current levels against historical averages and analyst expectations.

Machine Learning Forecasting:
Use a hybrid model that decomposes price series with CEEMDAN, optimizes Support Vector Regression (SVR) hyperparameters via Harris Hawks Optimization, and produces multi-step forecasts with prediction intervals.

Weather & Fundamental Analysis:
Fetch NOAA weather data to compute Heating Degree Days (HDD) and assess weather impacts on the market. Some modules also combine weather forecasts with AI analysis for trade signals.

Weekly Storage Report Automation:
Automatically pull and compare weekly U.S. natural gas storage data, contrasting actual changes with expectations, 5-year averages, and year-ago inventories to produce a trader’s report.

File Overview
barchart_technicals.py
Implements a full technical analysis framework using IB API data. It calculates multiple indicators and generates a final trade opinion based on traditional “Barchart” methods and Rogue Radar thresholds.

ng_ai_and_technicals_forecast.py
Builds on technical analysis by integrating AI. After computing key technical metrics, it creates a prompt and calls OpenAI’s GPT-4.5-preview model to provide a trade recommendation and analysis.

ng_consumption_by_sector.py
Connects to the EIA API to pull and print a monthly summary of natural gas consumption by sector (residential, commercial, industrial, electric power) for 2024.

ng_import_export_summary.py
Fetches monthly natural gas trade data from the EIA API—covering imports/exports by country, state, and points of entry/exit—and prints a clear, grouped summary.

ng_machine_learning_forcaster.py
Implements an end-to-end machine learning pipeline for forecasting natural gas futures (NGM5). It uses CEEMDAN for signal decomposition, SVR (optimized with Harris Hawks Optimization) within a Bagging ensemble, and outputs forecasts with prediction intervals.

ng_monthly_dry_production.py
Retrieves U.S. monthly natural gas production figures (dry production, marketed production, gross withdrawals) from the EIA API, focusing on the front month’s data.

ng_supply_disposition_balance.py
Analyzes supply and disposition data from the EIA API by comparing actual storage changes against consensus expectations, current inventory versus historical averages, and produces an overall market trend.

ng_weather_ai.py
Combines NOAA weather data with AI analysis. It calculates heating degree days, builds an AI prompt, and calls OpenAI’s GPT-4.5-preview model to offer weather-based trade recommendations.

ng_weather_fundamental.py
Focuses on fundamental weather analysis by fetching NOAA temperature data, computing heating degree days, and interpreting the weather impact on natural gas prices.

ng_weather_model.py
Integrates weather forecasts (via Open-Meteo), historical NOAA data, IB futures data, and EIA storage metrics to generate regional natural gas price signals based on weighted heating degree days and other key indicators.

weekly_report_updates.py
Automates weekly analysis of U.S. natural gas storage by retrieving storage data from the EIA API, calculating actual versus expected changes, comparing against 5-year averages and year-ago levels, and outputting a trader-friendly report with a handy cheat sheet.

Data Sources
Interactive Brokers (IB) API:
Provides historical and real-time market data for natural gas futures.

U.S. Energy Information Administration (EIA) API:
Supplies monthly production, consumption, trade, and storage data.

National Oceanic and Atmospheric Administration (NOAA) API:
Delivers weather data used to compute heating degree days and assess weather impacts.

OpenAI API:
Powers AI-driven analysis and trade recommendations through GPT-4.5-preview.

Usage
Each module is designed to run as a standalone script for specific analyses—from technical signal generation to machine learning forecasts and weather-based evaluations. Together, they form an extensible framework for a holistic view of the natural gas market.

Feel free to adjust any details to better match your project specifics. This “About” section helps users understand the project’s scope, individual module responsibilities, and the diverse data sources driving the analysis.
