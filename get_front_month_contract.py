#!/usr/bin/env python3

from ib_insync import IB, Future, Contract
from datetime import datetime

def parse_expiry(expiry_str):
    """
    Parse IB expiry strings, which might be YYYYMM or YYYYMMDD.
    """
    try:
        return datetime.strptime(expiry_str, '%Y%m%d')
    except ValueError:
        return datetime.strptime(expiry_str, '%Y%m')

def main():
    # 1) Connect to Interactive Brokers (adjust host/port/clientId as needed)
    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=8)
    print("[INFO] Connected to IB API.")

    try:
        # 2) Request all details for Natural Gas (NG) on NYMEX
        all_details = ib.reqContractDetails(Future(symbol='NG', exchange='NYMEX'))
        if not all_details:
            print("[ERROR] No NG futures contracts returned.")
            return

        # 3) Sort by earliest expiry => front month is the first
        sorted_details = sorted(
            all_details,
            key=lambda d: parse_expiry(d.contract.lastTradeDateOrContractMonth)
        )
        front_month = sorted_details[0].contract

        # 4) Print in the desired Contract() format
        print("\n--- FRONT-MONTH CONTRACT DEFINITION ---\n")
        print("contract = Contract()")
        print(f'contract.symbol = "{front_month.symbol}"  # Symbol')
        print(f'contract.secType = "{front_month.secType}"  # Security Type')
        print(f'contract.exchange = "{front_month.exchange}"  # Exchange')
        print(f'contract.currency = "{front_month.currency}"  # Currency')
        print(f'contract.lastTradeDateOrContractMonth = "{front_month.lastTradeDateOrContractMonth}"  # Expiry')
        print(f'contract.localSymbol = "{front_month.localSymbol}"  # Local Symbol')
        print(f'contract.multiplier = "{front_month.multiplier}"  # Multiplier\n')

    finally:
        # 5) Disconnect
        ib.disconnect()
        print("[INFO] Disconnected from IB API.")

if __name__ == '__main__':
    main()

