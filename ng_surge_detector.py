#!/usr/bin/env python3

import argparse
import time
from datetime import datetime, timedelta
import pandas as pd

from ib_insync import IB, Contract, util

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull NG futures tick data from IB and test basic surge logic."
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="IB API host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=7496,
        help="IB API port (default: 7497, or 7496 for live TWS)."
    )
    parser.add_argument(
        "--clientId", type=int, default=9,
        help="Client ID to avoid collisions with other apps."
    )
    parser.add_argument(
        "--days", type=int, default=2,
        help="Number of days of tick data to fetch (max ~7 recommended)."
    )
    parser.add_argument(
        "--symbol", default="NG",
        help="Futures symbol, e.g. NG for Natural Gas."
    )
    parser.add_argument(
        "--exchange", default="NYMEX",
        help="Exchange for the futures contract."
    )
    parser.add_argument(
        "--expiry", default="20250528",
        help="LastTradeDateOrContractMonth for the NG contract (YYYYMMDD or YYYYMM)."
    )
    parser.add_argument(
        "--localSymbol", default="NGM5",
        help="Local symbol if needed (like 'NGM5')."
    )
    return parser.parse_args()

def create_ng_contract(symbol, exchange, expiry, localSymbol=None):
    """
    Create a simple NG Futures contract. Adjust multiplier, localSymbol if needed.
    """
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "FUT"
    contract.exchange = exchange
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = expiry
    if localSymbol:
        contract.localSymbol = localSymbol
    contract.multiplier = "10000"  # typical for NG
    return contract

def fetch_tick_data(ib, contract, days=2):
    """
    Fetch up to `days` worth of historical ticks from IB.
    IB often limits the max # of ticks you can download in one request, so we may need to chunk it.
    We'll try a naive approach with 'MIDPOINT' or 'TRADES' ticks.
    """
    print(f"Requesting ~{days} days of tick data. This can take a while...")

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    # Interactive Brokers only provides ~6 days of 'historical ticks' data, 
    # and they often limit each chunk to about 1000 ticks, so we may do a loop.
    # We'll do a single chunk approach; if it's too big, it might fail or be partial:
    #   If partial, you can chunk it in smaller windows (like 1 day at a time).
    #   This is a simplified example.

    ticks = ib.reqHistoricalTicks(
        contract,
        startDateTime=start_dt,
        endDateTime=end_dt,
        numberOfTicks=1000,      # IB often caps at 1000 per request for 'historicalTicks'
        whatToShow="TRADES",     # or "MIDPOINT"
        useRth=False
    )
    if not ticks:
        print("No tick data returned. Possibly out of IB's range or no permission.")
        return pd.DataFrame()

    # Convert to DataFrame
    data = []
    for t in ticks:
        row = {
            "time": t.time,
            "price": t.price,
            "size": t.size
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Fetched {len(df)} ticks from IB.")
    return df

def detect_surge_events(df, price_threshold=0.001, volume_factor=1.5, window_sec=5):
    """
    Example 'surge detection' that checks if, within a small rolling window:
      - Price changed by >= 0.4% (0.004) 
      - Volume is >= 3x the rolling average (or something similar)
    
    This is *very naive* and just a demonstration. 
    You'd likely do a more robust approach (like a queue of ticks, rolling sums, etc.).
    """

    # Convert times to numeric (secs) for rolling window
    df["timestamp"] = df["time"].astype("int64") // 1_000_000_000  # seconds
    # We'll do a simple loop to see if the last N seconds had a big jump.

    # We'll store events in a list
    surge_events = []
    # We'll do an incremental approach:
    # For each tick i, look backwards for up to window_sec to see if price jumped enough.
    
    # A more robust approach is to do a rolling approach, but let's keep it simple.
    # We'll also do a rough volume check: # of ticks or sum of sizes in that window, vs. some average.

    # Pre-calc an average size
    avg_size = df["size"].mean() if not df.empty else 1

    # The idea: for each row, find the row in the last 'window_sec' and compare price min/max, volume, etc.
    # We'll do a naive approach: for each row i, define window start time = row_i.timestamp - window_sec,
    # slice the sub-DF, check if there's >=0.4% price change from min->max, or sum volumes, etc.

    for i in range(len(df)):
        current_time = df.loc[i, "timestamp"]
        current_price = df.loc[i, "price"]
        window_start = current_time - window_sec
        # slice:
        sub = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= current_time)]
        if sub.empty:
            continue

        price_min = sub["price"].min()
        price_max = sub["price"].max()
        vol_sum = sub["size"].sum()

        # Basic check: if (price_max - price_min) / price_min >= 0.4% => surge
        # But we'll see if the current price is up or down from sub's earliest
        earliest_price = sub.iloc[0]["price"]
        price_change = (current_price - earliest_price) / (earliest_price + 1e-9)

        # volume check: compare sub's total volume to (window_sec * avg_size) for naive factor
        # or do vol_sum / (sub.shape[0]) vs. avg_size for per-tick comparison
        # let's do total volume check:
        volume_thresh = volume_factor * (avg_size * (sub.shape[0]))  # naive guess

        if abs(price_change) >= price_threshold and vol_sum >= volume_thresh:
            # We say we detected a surge
            direction = "UP" if price_change > 0 else "DOWN"
            # record event:
            surge_events.append({
                "event_time": df.loc[i, "time"],
                "last_price": current_price,
                "price_change_pct": price_change * 100,
                "vol_window": vol_sum,
                "num_ticks_window": sub.shape[0],
                "direction": direction
            })

    return pd.DataFrame(surge_events)

def main():
    args = parse_args()

    ib = IB()
    print(f"Connecting to IB at {args.host}:{args.port} with clientId={args.clientId} ...")
    ib.connect(args.host, args.port, clientId=args.clientId)
    print("Connection established.")

    # Build contract
    contract = create_ng_contract(
        symbol=args.symbol,
        exchange=args.exchange,
        expiry=args.expiry,
        localSymbol=args.localSymbol
    )

    # We fetch tick data for ~ X days
    df_ticks = fetch_tick_data(ib, contract, days=args.days)
    if df_ticks.empty:
        print("No data. Exiting.")
        return

    print("Preview of fetched ticks:")
    print(df_ticks.head(10))

    # Example surge detection
    # We'll do a naive approach: 0.4% threshold, 3x volume factor, 5-second window
    surge_df = detect_surge_events(df_ticks, price_threshold=0.004, volume_factor=3, window_sec=5)

    print("\n=== DETECTED SURGE EVENTS ===")
    if surge_df.empty:
        print("No surge events found with these parameters. Try adjusting your thresholds.")
    else:
        print(surge_df)

    # Done
    ib.disconnect()
    print("Finished. Disconnected from IB.")


if __name__ == "__main__":
    main()

