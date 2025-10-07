from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical import CryptoHistoricalDataClient

from datetime import datetime

import datetime

# Alpaca API credentials
api_key_id = 'PKON7XLHTRMPTZDJ6JB3'
api_secret = '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL'
base_url = 'https://paper-api.alpaca.markets/v2'

trading_client = TradingClient(api_key_id, api_secret, paper=True)

# Get Tickers
def get_tickers():
    print('Getting Current Ticker Data...')

    symbols = ["AAVE/USD", "AVAX/USD", "BAT/USD", "BCH/USD", "CRV/USD",
               "DOGE/USD", "DOT/USD", "GRT/USD", "LINK/USD", "LTC/USD",
               "MKR/USD", "SHIB/USD", "SUSHI/USD", "UNI/USD","USDC/USD",
               "USDT/USD", "XTZ/USD", "YFI/USD"]

    # no keys required for crypto data
    client = CryptoHistoricalDataClient()

    # Get data for the specified symbols
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Minute, start=datetime.datetime(2020, 1, 1),
        end=datetime.datetime(2023, 12, 31)
    )

    data = client.get_crypto_bars(request_params)

    data = data.df

    # Save DataFrame with index to CSV
    data.to_csv('output_file.csv', index=True)

# Run the function
get_tickers()
