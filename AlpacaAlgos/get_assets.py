#JOHANSEN TEST TWO

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.timeframe import *
from alpaca.trading.client import *
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Alpaca API credentials
api_key_id = 'PKQRDDO1GX8JTPLXX0BJ'
api_secret = 'sbgjHq2yrYvKByWnkXThFmTcENYSooJ1jNNSzgUv'
base_url = 'https://paper-api.alpaca.markets/v2'

trading_client = TradingClient(api_key_id, api_secret, paper=True)


def get_pairs():
    print('Getting Current ETF Pair Data...')

    # Load ETF symbols from CSV file
    try:
        etf_symbols = pd.read_csv('etf.csv')['Symbols'].tolist()
        print(f"Loaded {len(etf_symbols)} ETF symbols from CSV.")
    except Exception as e:
        print(f"Error loading ETF symbols: {e}")
        return []

    # Check which symbols are tradable
    print("Checking tradable symbols...")
    tradable_assets = []
    try:
        search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
        assets = trading_client.get_all_assets(search_params)
        tradable_assets = {asset.symbol for asset in assets if asset.tradable}
        print(f"Found {len(tradable_assets)} tradable symbols.")
    except Exception as e:
        print(f"Error fetching tradable assets: {e}")
        return []

    etf_symbols = [symbol for symbol in etf_symbols if symbol in tradable_assets]
    if not etf_symbols:
        print("No tradable ETF symbols found. Exiting.")
        return []

    pairs_to_test = [(etf_symbols[i], etf_symbols[j]) for i in range(len(etf_symbols)) for j in
                     range(i + 1, len(etf_symbols))]

    client = StockHistoricalDataClient(api_key_id, api_secret)
    start_date = pd.to_datetime("2024-01-01").tz_localize('America/New_York')
    end_date = pd.to_datetime("2024-12-31").tz_localize('America/New_York')

    cointegration_results = []

    for symbol1, symbol2 in pairs_to_test:
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol1, symbol2], timeframe=TimeFrame.Day, start=start_date, end=end_date
            )
            data = client.get_stock_bars(request_params)

            if data.df.empty:
                print(f"   No data for {symbol1} or {symbol2}. Skipping.")
                continue

            df = data.df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d')
            df = df.pivot(index='timestamp', columns='symbol', values='close')

            if df.isnull().any().any() or df.shape[0] < 50:
                print(f"Insufficient data for {symbol1} and {symbol2}.")
                continue

            print(f"   Performing Johansen test on {symbol1} and {symbol2}...")
            result = coint_johansen(df[[symbol1, symbol2]].values, det_order=0, k_ar_diff=1)

            cointegration_strength = result.eig[0]
            eigenvalues = result.eig
            eigenvectors = result.evec
            trace_statistics = result.lr1
            crit_values = result.cvt[:, 1]  # 5% significance level

            if np.any(trace_statistics > crit_values):
                cointegration_results.append((symbol1, symbol2, cointegration_strength, eigenvalues, eigenvectors,
                                              trace_statistics, crit_values))
            else:
                print(f"   No significant cointegration found for this pair.")

        except Exception as e:
            print(f"Error processing {symbol1} and {symbol2}: {e}")
            continue

    unique_results = []
    seen_symbols = set()
    cointegration_results.sort(key=lambda x: x[2], reverse=True)

    for entry in cointegration_results:
        symbol1, symbol2, strength, eigvals, eigvecs, trace_stats, crit_vals = entry
        if symbol1 not in seen_symbols and symbol2 not in seen_symbols:
            unique_results.append(entry)
            seen_symbols.update([symbol1, symbol2])

    tradable_pairs = unique_results[:10] if len(unique_results) >= 10 else unique_results

    print(f"\nTop Cointegrated ETF Pairs:")
    for rank, (symbol1, symbol2, strength, eigvals, eigvecs, trace_stats, crit_vals) in enumerate(tradable_pairs,
                                                                                                  start=1):
        print(f"{rank}. Pair: {symbol1} & {symbol2}, Strength: {strength}")
        print(f"   Eigenvalues: {eigvals}")
        print(f"   Eigenvectors: {eigvecs}")
        print(f"   Trace Statistics: {trace_stats}")
        print(f"   Critical Values (5% level): {crit_vals}")

    return [(pair[0], pair[1]) for pair in tradable_pairs]


if __name__ == "__main__":
    tradable_pairs = get_pairs()
    print("\nFinal Output: Top Cointegrated ETF Pairs")
    for idx, (symbol1, symbol2) in enumerate(tradable_pairs, start=1):
        print(f"{idx}. {symbol1} & {symbol2}")


#JOHANSEN TEST ONE
# from alpaca.data.historical.stock import StockHistoricalDataClient
#
# from alpaca.data.timeframe import *
# from alpaca.trading.client import *
#
# from alpaca.data.requests import (
#     StockBarsRequest
# )
# from alpaca.trading.requests import GetAssetsRequest
# from alpaca.trading.enums import AssetClass
#
# import pandas as pd
# import numpy as np
#
# from statsmodels.tsa.vector_ar.vecm import coint_johansen
#
# # Alpaca API credentials
# api_key_id = 'PKON7XLHTRMPTZDJ6JB3'
# api_secret = '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL'
# base_url = 'https://paper-api.alpaca.markets/v2'
#
# trading_client = TradingClient(api_key_id, api_secret, paper=True)
#
# def get_pairs():
#     print('Getting Current ETF Pair Data...')
#
#     # Load ETF symbols from CSV file
#     try:
#         etf_symbols = pd.read_csv('etf_symbols.csv')['Symbols'].tolist()
#         print(f"Loaded {len(etf_symbols)} ETF symbols from CSV.")
#     except Exception as e:
#         print(f"Error loading ETF symbols: {e}")
#         return []
#
#     # Check which symbols are tradable
#     print("Checking tradable symbols...")
#     tradable_assets = []
#     try:
#         search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
#         assets = trading_client.get_all_assets(search_params)
#         tradable_assets = {asset.symbol for asset in assets if asset.tradable}
#         print(f"Found {len(tradable_assets)} tradable symbols.")
#     except Exception as e:
#         print(f"Error fetching tradable assets: {e}")
#         return []
#
#     # Filter ETF symbols to include only tradable ones
#     etf_symbols = [symbol for symbol in etf_symbols if symbol in tradable_assets]
#     if not etf_symbols:
#         print("No tradable ETF symbols found. Exiting.")
#         return []
#
#     # Generate all unique pairs of tradable ETFs
#     pairs_to_test = [(etf_symbols[i], etf_symbols[j]) for i in range(len(etf_symbols)) for j in
#                      range(i + 1, len(etf_symbols))]
#
#     # Initialize historical data client
#     client = StockHistoricalDataClient('PKON7XLHTRMPTZDJ6JB3',
#                                                              '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')
#
#     # Set the start time
#     start_date = pd.to_datetime("2024-01-01").tz_localize('America/New_York')
#     end_date = pd.to_datetime("2024-12-31").tz_localize('America/New_York')
#
#     cointegration_results = []
#
#     for symbol1, symbol2 in pairs_to_test:
#         try:
#             # Request historical data for both symbols
#             request_params = StockBarsRequest(
#                 symbol_or_symbols=[symbol1, symbol2], timeframe=TimeFrame.Day, start=start_date, end=end_date
#             )
#             data = client.get_stock_bars(request_params)
#
#             if data.df.empty:
#                 print(f"   No data for {symbol1} or {symbol2}. Skipping.")
#                 continue
#
#             # Reset index and pivot data for Johansen Test
#             df = data.df.reset_index()
#             df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d')
#             df = df.pivot(index='timestamp', columns='symbol', values='close')
#
#             if df.isnull().any().any() or df.shape[0] < 50:
#                 print(f"Insufficient data for {symbol1} and {symbol2}.")
#                 continue
#
#             # Perform Johansen Test
#             print(f"   Performing Johansen test on {symbol1} and {symbol2}...")
#             result = coint_johansen(df[[symbol1, symbol2]].values, det_order=0, k_ar_diff=1)
#
#             cointegration_strength = result.eig[0]
#
#             if np.any(result.lr1 > result.cvt[:, 1]):  # 5% significance level
#                 cointegration_results.append((symbol1, symbol2, cointegration_strength))
#             else:
#                 print(f"   No significant cointegration found for this pair.")
#
#         except Exception as e:
#             print(f"Error processing {symbol1} and {symbol2}: {e}")
#             continue
#
#     # Sort and filter results
#     unique_results = []
#     seen_symbols = set()
#     cointegration_results.sort(key=lambda x: x[2], reverse=True)
#
#     for symbol1, symbol2, strength in cointegration_results:
#         if symbol1 not in seen_symbols and symbol2 not in seen_symbols:
#             unique_results.append((symbol1, symbol2, strength))
#             seen_symbols.update([symbol1, symbol2])
#
#     tradable_pairs = unique_results[:10] if len(unique_results) >= 10 else unique_results
#
#     print(f"\nTop Cointegrated ETF Pairs:")
#     for rank, pair in enumerate(tradable_pairs, start=1):
#         print(f"{rank}. Pair: {pair[0]} & {pair[1]}, Strength: {pair[2]}")
#
#     return [(pair[0], pair[1]) for pair in tradable_pairs]
#
# if __name__ == "__main__":
#     tradable_pairs = get_pairs()
#     print("\nFinal Output: Top Cointegrated ETF Pairs")
#     for idx, (symbol1, symbol2) in enumerate(tradable_pairs, start=1):
#         print(f"{idx}. {symbol1} & {symbol2}")


# from alpaca.data.historical.stock import StockHistoricalDataClient
#
# from alpaca.data.timeframe import *
# from alpaca.trading.client import *
# from alpaca.trading.requests import *
# from alpaca.data.requests import (
#     StockBarsRequest
# )
# from alpaca.trading.requests import (
#     GetAssetsRequest
# )
#
# api_key_id = 'PKON7XLHTRMPTZDJ6JB3'
# api_secret = '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL'
#
# trading_client = TradingClient('PKON7XLHTRMPTZDJ6JB3', '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')
#
# # search for US equities
# search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, exchange=AssetExchange.ARCA)
#
# assets = trading_client.get_all_assets(search_params)
#
# symbols = [asset.symbol for asset in assets if asset.tradable]
#
# print(symbols)
#
# # setup stock historical data client
# stock_historical_data_client = StockHistoricalDataClient('PKON7XLHTRMPTZDJ6JB3',
#                                                          '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')
#
# # Set the start time
# start_date = pd.to_datetime("2024-07-01").tz_localize('America/New_York')
# end_date = pd.to_datetime("2024-08-16").tz_localize('America/New_York')
#
# # Get data for the specified symbols
# request_params = StockBarsRequest(
#     symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_date
#     )
#
# data = stock_historical_data_client.get_stock_bars(request_params)
#
# data = data.df
#
# print(data)

# Crypto

# from alpaca.trading.client import TradingClient
#
# from alpaca.data.requests import CryptoBarsRequest
#
# from alpaca.data.timeframe import TimeFrame
#
# from alpaca.trading.requests import (
#     GetAssetsRequest
# )
#
# from alpaca.trading.enums import (
#     AssetClass,
#     AssetStatus
# )
#
# from alpaca.data.historical import CryptoHistoricalDataClient
#
# import pandas as pd
#
# # Alpaca API credentials
# api_key_id = 'PKON7XLHTRMPTZDJ6JB3'
# api_secret = '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL'
# base_url = 'https://paper-api.alpaca.markets/v2'
#
# trading_client = TradingClient(api_key_id, api_secret, paper=True)
#
# # get list of crypto pairs
#
# req = GetAssetsRequest(
#   asset_class=AssetClass.CRYPTO,
#   status=AssetStatus.ACTIVE
# )
# assets = trading_client.get_all_assets(req)
#
# symbols = [asset.symbol for asset in assets if asset.tradable]
#
# print(symbols)
#
# # no keys required for crypto data
# client = CryptoHistoricalDataClient()
#
# # Set the start time
# start_date = pd.to_datetime("2024-07-01").tz_localize('America/New_York')
# end_date = pd.to_datetime("2024-08-19").tz_localize('America/New_York')
#
# # Get data for the specified symbols
# request_params = CryptoBarsRequest(
#     symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_date, end=end_date
# )
#
# data = client.get_crypto_bars(request_params)
#
# data = data.df
#
# print(data)
#
# # Calculate 20-day moving average of closing prices
# data['20_day_ma'] = data['close'].rolling(window=20).mean()
#
# # Narrow down the list based on open prices being higher than the 20-day moving average
# momentum_cryptos = data[data['open'] > data['20_day_ma']]
#
# print(momentum_cryptos)
#
# # Specify the file path and name for the CSV file
# file_path = "crypto_data_two.csv"
#
# # Save the DataFrame to a CSV file
# data.to_csv(file_path, index=True)  # 'index=True' will include the index in the CSV file
#
# momentum_cryptos.to_csv(file_path, index=True)
#
# print(f"Data saved to {file_path}")
