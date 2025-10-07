import math

from api_interface import Main, BarData, PriceInformation

from datetime import datetime

from pytz import timezone

from threading import Timer

import numpy as np
import pandas as pd
import datetime, time

import networkx as nx
import matplotlib.pyplot as plt

from datetime import timedelta

import yfinance as yf

num = 7

class App(Main):
    def __init__(self, ip, port, client):
        Main.__init__(self, ip, port, client)

        self.portfolio_value = 0.0
        self.macd_size = 1
        self.bar_size = '1 min'
        self.historical_bar_size = self.bar_size

        if int(self.bar_size.split()[0]) * self.macd_size >= 60:
            self.historical_data_duration = "{} S".format(int(self.bar_size.split()[0]) * self.macd_size)
        else:
            self.historical_data_duration = "{} S".format(60)

        self.prices = {}

        self.ts = None
        self.price = None

        self.lots_to_buy = 0

        self.default_stop = 0.99950

        self.risk = 0.000002

        # Get when the market opens or opened today
        self.nyc = timezone('America/New_York')
        self.today = datetime.datetime.today().astimezone(self.nyc)

        self.current_dt = datetime.datetime.today().astimezone(self.nyc)

        self.start_time = self.today.replace(hour=1, minute=00, second=0, microsecond=0)
        self.start_time = self.start_time.astimezone(self.nyc)

        self.stop_time = self.today.replace(hour=16, minute=30, second=0, microsecond=0)
        self.stop_time = self.today.replace(hour=16, minute=25, second=0, microsecond=0)
        # self.stop_time = self.stop_time.astimezone(self.nyc)

        self.pnl_time = self.stop_time.astimezone(self.nyc)

        labels = ['USD', 'EUR', 'AUD', 'GBP', 'CAD', 'JPY', 'CHF']

        # self.symbols = ['USD.EUR', 'USD.AUD', 'USD.GBP', 'USD.CAD', 'USD.JPY', 'USD.CHF', 'EUR.USD',
        #                 'EUR.AUD', 'EUR.GBP', 'EUR.CAD', 'EUR.JPY', 'EUR.CHF', 'AUD.USD', 'AUD.EUR',
        #                 'AUD.GBP', 'AUD.CAD', 'AUD.JPY', 'AUD.CHF', 'GBP.USD', 'GBP.EUR', 'GBP.AUD',
        #                 'GBP.CAD', 'GBP.JPY', 'GBP.CHF', 'CAD.USD', 'CAD.EUR', 'CAD.AUD', 'CAD.GBP',
        #                 'CAD.JPY', 'CAD.CHF', 'JPY.USD', 'JPY.EUR', 'JPY.AUD', 'JPY.GBP', 'JPY.CAD',
        #                 'JPY.CHF', 'CHF.USD', 'CHF.EUR', 'CHF.AUD', 'CHF.GBP', 'CHF.CAD', 'CHF.JPY']

        self.symbols = ['EUR.USD', 'AUD.USD', 'GBP.USD']

        self.currency_pairs = [
            self.EUR_USD_FX(),
            self.AUD_USD_FX(),
            self.GBP_USD_FX()
        ]

        self.historical_data = {}

        self.stop_prices = {}

        self.vertices = num
        self.arches = []
        self.tickers = labels
        self.currency_matrix = np.zeros((num, num))

        self.dataframe = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close'])

    def get_historical_data(self):
        print('Getting historical data...')

        # Load the currency pairs from the csv file
        fx_pairs_df = pd.read_csv("C:/Users/STEVE/Desktop/fx_pairs.csv")
        available_pairs = set(fx_pairs_df["Symbol"])

        count = 0

        for contract in self.symbols:
            if contract in available_pairs:
                # If the currency pair is available, directly retrieve historical data
                bars = self.get_historical_fx_data(contract, bar_size=self.historical_bar_size,
                                                   duration=self.historical_data_duration,
                                                   data_type="MIDPOINT")
                data = {'Date': [bar.date for bar in bars],
                        'Open': [bar.open for bar in bars],
                        'High': [bar.high for bar in bars],
                        'Low': [bar.low for bar in bars],
                        'Close': [bar.close for bar in bars]}
            else:
                # If the currency pair is not available, get its reciprocal
                base_currency, quote_currency = contract.split(".")
                reciprocal_contract = f"{quote_currency}.{base_currency}"

                # Check if the reciprocal pair is available
                if reciprocal_contract in available_pairs:
                    # Retrieve historical data for the reciprocal pair
                    bars = self.get_historical_fx_data(reciprocal_contract, bar_size=self.historical_bar_size,
                                                       duration=self.historical_data_duration,
                                                       data_type="MIDPOINT")
                    # Calculate reciprocal values for Open, High, Low, Close
                    data = {'Date': [bar.date for bar in bars],
                            'Open': [1 / bar.open for bar in bars],
                            'High': [1 / bar.high for bar in bars],
                            'Low': [1 / bar.low for bar in bars],
                            'Close': [1 / bar.close for bar in bars]}
                else:
                    print(f"Reciprocal pair {reciprocal_contract} not available.")
                    continue

            # Store historical data in the dictionary
            self.historical_data[contract] = pd.DataFrame(data)
            self.dataframe = pd.concat([self.dataframe, self.historical_data[contract]], ignore_index=True)

            count += 1
            # print('{}/{}'.format(count, len(currency_pairs)))

        print('Success.')
        return self.historical_data

    def main(self):
        if self.current_dt < self.start_time:
            seconds_to_wait = (self.start_time - self.current_dt).total_seconds()
            Timer(seconds_to_wait, self.main).start()
            print("Starting later today at: {}".format(self.start_time.time()))
            return None

        seconds_to_wait = (self.start_time + timedelta(days=1) - self.current_dt).total_seconds()
        Timer(seconds_to_wait, self.main).start()

        if datetime.datetime.now().weekday() in [5, 6]:
            print("It's the weekend, no trading today")
            return None

        seconds_to_wait = (self.pnl_time - self.current_dt).total_seconds()
        Timer(seconds_to_wait, self.get_pnl).start()

        if not self.isConnected():
            self.reconnect()

        time.sleep(2)

        contract = None  # Default value

        # # Fetch historical data
        # historical_data = self.get_historical_data(self.symbols)
        #
        # # Print out historical data
        # for symbol, data in historical_data.items():
        #     print(f"\nHistorical data for {symbol}:")
        #     print(data)

        # Load the currency pairs from the csv file
        fx_pairs_df = pd.read_csv("C:/Users/STEVE/Desktop/fx_pairs.csv")
        available_pairs = set(fx_pairs_df["Symbol"])

        # for i, pair in enumerate(self.symbols):
        #
        #     if contract in available_pairs:
        #
        #         # Creating contract instance using fxPair method
        #         contract = self.fxPair(pair)
        #
        #         req_id = self.get_unique_id() + i
        #
        #         self.market_data[req_id] = PriceInformation(contract)
        #
        #         self.reqMarketDataType(1)
        #         self.reqMktData(req_id, contract, "", False, False, [])

            # # Get the rounded prices for the contract
            # rounded_prices = self.get_midpoint_price(contract)
            #
            # # Print out the rounded prices
            # print("Rounded Prices for {}: {}".format(contract, rounded_prices))

        for i, contract in enumerate(self.currency_pairs):
            req_id = self.get_unique_id() + i

            self.market_data[req_id] = PriceInformation(contract)

            self.reqMarketDataType(1)
            self.reqMktData(req_id, contract, "", False, False, [])

        self.get_midpoint_price()

        # self.Bellman_Ford()

    # Get Midpoint Data
    def get_midpoint_price(self):

        for i, contract in enumerate(self.currency_pairs):

            req_id = self.get_unique_id() + i

            print('The next unique id is: ', req_id)
            # Print the contract symbol to check its value
            print(f"Contract: {contract}")

            current_datetime = datetime.datetime.today().astimezone(self.nyc)

            # Convert current_datetime to a new datetime object with seconds and microseconds set to zero
            current_datetime = current_datetime.replace(second=0, microsecond=0)

            ts = current_datetime.strftime("%Y%m%d %H:%M:%S")

            # Check if the req_id exists in the self.market_data dictionary
            if req_id in self.market_data:
                # Access the market data using req_id and calculate the price
                market_data = self.market_data[req_id]
                if market_data.Bid is not None and market_data.Ask is not None:
                    midpoint_price = (market_data.Bid + market_data.Ask) / 2

                    if contract in ['EUR/USD', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'USD/CNH', 'NZD/USD', 'USD/CHF']:
                        rounded_price = round(midpoint_price, 5)
                    elif contract == 'USD/JPY':
                        rounded_price = round(midpoint_price, 3)
                    else:
                        rounded_price = round(midpoint_price, 5)

                    print("Midpoint price:", rounded_price)

                    # Add the relevant information to self.prices dictionary
                    if midpoint_price is not None:
                        self.prices[contract] = {
                            'req_id': req_id,
                            'ts': ts,
                            'price': midpoint_price
                        }

                    time.sleep(1)

        # # Load the currency pairs from the csv file
        # fx_pairs_df = pd.read_csv("C:/Users/STEVE/Desktop/fx_pairs.csv")
        # available_pairs = set(fx_pairs_df["Symbol"])
        #
        # for i, pair in enumerate(self.symbols):
        #
        #     contract = self.fxPair(pair)
        #
        #     req_id = self.get_unique_id() + i
        #
        #     print('The next unique id is: ', req_id)
        #     # Print the contract symbol to check its value
        #     print(f"Contract: {contract}")
        #
        #     current_datetime = datetime.datetime.today().astimezone(self.nyc)
        #
        #     # Convert current_datetime to a new datetime object with seconds and microseconds set to zero
        #     current_datetime = current_datetime.replace(second=0, microsecond=0)
        #
        #     ts = current_datetime.strftime("%Y%m%d %H:%M:%S")
        #
        #     # Check if the req_id exists in the self.market_data dictionary
        #     if req_id in self.market_data:
        #         # Access the market data using req_id and calculate the price
        #         market_data = self.market_data[req_id]
        #         if market_data.Bid is not None and market_data.Ask is not None:
        #             midpoint_price = (market_data.Bid + market_data.Ask) / 2
        #
        #             if contract in ['EUR.USD', 'AUD.USD', 'GBP.USD', 'USD.CAD', 'USD.CNH', 'NZD.USD', 'USD.CHF']:
        #                 rounded_price = round(midpoint_price, 5)
        #             elif contract == 'USD.JPY':
        #                 rounded_price = round(midpoint_price, 3)
        #             else:
        #                 rounded_price = round(midpoint_price, 5)
        #
        #             print("Midpoint price:", rounded_price)
        #
        #             # Add the relevant information to self.prices dictionary
        #             if midpoint_price is not None:
        #                 self.prices[contract] = {
        #                     'req_id': req_id,
        #                     'ts': ts,
        #                     'price': midpoint_price
        #                 }
        #
        #             time.sleep(1)


app = App("127.0.0.1", 7497, 1)
try:
    app.main()
except KeyboardInterrupt:
    app.disconnect()