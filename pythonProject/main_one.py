import math

from api_interface import Main, BarData, PriceInformation

from datetime import datetime

from pytz import timezone

from threading import Timer

import pandas as pd
import numpy as np
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

        # Get when the market opens or opened today
        self.nyc = timezone('America/New_York')
        self.today = datetime.datetime.today().astimezone(self.nyc)

        self.current_dt = datetime.datetime.today().astimezone(self.nyc)

        self.start_time = self.today.replace(hour=1, minute=00, second=0, microsecond=0)
        self.start_time = self.start_time.astimezone(self.nyc)

        self.stop_time = self.today.replace(hour=16, minute=30, second=0, microsecond=0)
        self.pnl_time = self.today.replace(hour=16, minute=25, second=0, microsecond=0)
        # self.stop_time = self.stop_time.astimezone(self.nyc)

        self.lots_to_buy = 0

        self.default_stop = 0.99950

        self.risk = 0.000002

        labels = ['USD', 'EUR', 'AUD', 'GBP', 'CAD', 'JPY', 'CHF']

        self.currency_pairs = [
            self.EUR_USD_FX(), self.AUD_USD_FX(), self.GBP_USD_FX(), self.USD_CAD_FX(), self.USD_JPY_FX(),
            self.USD_CHF_FX(), self.EUR_AUD_FX(), self.GBP_AUD_FX(), self.EUR_GBP_FX(), self.EUR_CAD_FX(),
            self.GBP_CAD_FX(), self.EUR_JPY_FX(), self.AUD_JPY_FX(), self.GBP_JPY_FX(), self.CAD_JPY_FX(),
            self.CHF_JPY_FX(), self.EUR_CHF_FX(), self.AUD_CHF_FX(), self.GBP_CHF_FX(), self.CAD_CHF_FX(),
            self.USD_CAD_FX()]

        self.historical_data = {}

        self.vertices = num
        self.arches = []
        self.tickers = labels
        self.currency_matrix = np.zeros((num, num))

        self.dataframe = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close'])

        self.target_prices = {}

        self.stop_prices = {}

        self.historical_data = {}

        self.market_data_dict = {}

        self.latest_cost_basis = {}

        self.open_orders = []

    def get_historical_data(self, currency_pairs):
        print('Getting historical data...')

        count = 0

        for contract in currency_pairs:
            if contract in self.currency_pairs:
                # If the contract is available, retrieve data as usual
                bars = self.get_historical_market_data(contract,
                                                       bar_size=self.historical_bar_size,
                                                       duration=self.historical_data_duration,
                                                       data_type="MIDPOINT")
            else:
                # If the contract is not available, reciprocate the rate
                reciprocated_contract, bars = self.reciprocate_currency_pair(contract)
                print(f"Reciprocated rate for {contract} to {reciprocated_contract}")

            data = {'Date': [bar.date for bar in bars],
                    'Open': [bar.open for bar in bars],
                    'High': [bar.high for bar in bars],
                    'Low': [bar.low for bar in bars],
                    'Close': [bar.close for bar in bars]}

            self.historical_data[contract] = pd.DataFrame(data)
            self.dataframe = pd.concat([self.dataframe, self.historical_data[contract]], ignore_index=True)

            count += 1
            print('{}/{}'.format(count, len(currency_pairs)))

        print('Success.')
        return self.historical_data

    def reciprocate_currency_pair(self, contract):
        # Reciprocate the currency pair
        contract_symbol, contract_currency = contract.split('.')
        reciprocated_contract = f"{contract_currency}.{contract_symbol}"

        # Reciprocating the rate
        reciprocated_data = self.historical_data[reciprocated_contract].copy()
        reciprocated_data['Close'] = 1 / reciprocated_data['Close']
        reciprocated_data['High'] = 1 / reciprocated_data['High']
        reciprocated_data['Open'] = 1 / reciprocated_data['Open']
        reciprocated_data['Low'] = 1 / reciprocated_data['Low']

        return reciprocated_contract, reciprocated_data

    def find_stop(self, current_value, historical_data, now):
        # Filter historical data for the specific currency pair (contract)
        historical_data = self.dataframe

        series = historical_data['Low'][-100:].dropna()

        # Convert 'series' index to a DatetimeIndex if it is not already
        if not isinstance(series.index, pd.DatetimeIndex) and len(series) > 0:
            series.index = pd.to_datetime(series.index)

        # Ensure 'now' is a pandas Timestamp object with UTC time zone
        if not isinstance(now, pd.Timestamp):
            now = pd.Timestamp(now).tz_localize('UTC')

        # Convert 'series' index to UTC time zone
        series = series.tz_localize('UTC')

        # Get the slice of 'series' from 'now' to the end of the day
        series_slice = series[now.floor('1D'):]

        # Resample and get the minimum value
        resampled_series = series_slice.resample('5min').min()

        diff = np.diff(resampled_series.values)
        low_index = np.where((diff[:-1] <= 0) & (diff[1:] > 0))[0] + 1

        if len(low_index) > 0:
            return resampled_series.iloc[low_index[-1]] - pd.Timedelta(seconds=1)

        return current_value * self.default_stop

    def define_arches(self, s, e, v):  # s=start, e=end, v=value
        self.arches.append([s, e, v])
        print(f"Defined edge: {s} -> {e} with value {v}")

    # Building a matrix data structure and appending edges to graph
    def build_graph(self):
        for i in range(self.vertices):
            for j in range(self.vertices):
                if i == j:
                    # The diagonal is always equal to 1
                    data = 1
                else:
                    # Concatenating --> ex. "USDEUR=X"
                    aux = str(self.tickers[i] + self.tickers[j] + '=X')
                    print(aux)

                    ticker = self.tickers[i] + self.tickers[j] + '=X'
                    fx_data = yf.download(tickers=ticker, period='1d', interval='1m')

                    # Check if fx_data is not empty
                    if not fx_data.empty:
                        # Assuming the close price represents the exchange rate
                        data = fx_data['Close'].iloc[-1]
                    else:
                        # Handle the case where fx_data is empty
                        print(f"No data available for {ticker}. Using default value.")
                        data = 1  # Set a default value or handle it according to your requirements

                # Keeping track of exchange rates
                self.currency_matrix[i][j] = data
                self.define_arches(i, j, round(-math.log(data, 10), 5))

    def Negative_Cycle(self, dist, path):
        Neg_cycles = []
        flag = False
        for s, e, v in self.arches:
            # Verifying distance after the algo has converged
            if dist[s] + v < dist[e] and dist[s] != float("Inf"):
                neg_cycle = [e, s]
                aux = s  # auxiliary variable

                while path[aux] not in neg_cycle:  # Going backwards in original path
                    neg_cycle.append(path[aux])
                    aux = path[aux]
                neg_cycle.append(path[aux])

                # Selecting valid cycle
                if neg_cycle[0] == neg_cycle[-1] and len(neg_cycle) > 3:
                    Neg_cycles.append(neg_cycle)
                    flag = True

        if(flag):
            return Neg_cycles
        else:
            return False

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

        for i, contract in enumerate(self.currency_pairs):
            req_id = self.get_unique_id() + i

            self.market_data[req_id] = PriceInformation(contract)

            self.reqMarketDataType(1)
            self.reqMktData(req_id, contract, "", False, False, [])

        self.loop()

    def loop(self):

        while self.current_dt < self.stop_time:

            self.get_historical_data(self.currency_pairs)

            time.sleep(1)

            # Get the latest portfolio value here
            self.fetchAccountUpdates()

            portfolio_value = self.portfolio_value

            print("Current Portfolio Value:", portfolio_value)

            for i, contract in enumerate(self.currency_pairs):
                self.get_positions()
                existing_positions = self.positions

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

                        # Reciprocate the price if necessary
                        if contract not in self.currency_pairs:
                            midpoint_price = self.reciprocate_price(midpoint_price)

                        if contract in ['EUR/USD', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'USD/CNH', 'NZD/USD', 'USD/CHF']:
                            rounded_price = round(midpoint_price, 5)
                        elif contract == ['USD/JPY', 'EUR/JPY', 'AUD/JPY', 'GBP/JPY', 'CAD/JPY', 'CHF/JPY']:
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

                        price = self.prices[contract]['price']
                        ts = self.prices[contract]['ts']

                        stop_price = self.find_stop(
                            price, self.historical_data[contract], ts
                        )
                        self.stop_prices[contract] = stop_price

                        target_price = price + (
                                (price - stop_price) * 2
                        )

                        self.target_prices[contract] = target_price

                        # Contract has passed all checks; figure out how much to buy

                        portfolio_value = self.portfolio_value

                        print("Portfolio Value:", portfolio_value)

                        # Calculate quantity based on formula
                        if portfolio_value is not None:  # Check if portfolio_value is not None

                            self.lots_to_buy = portfolio_value * self.risk // (price - stop_price)

                            if self.lots_to_buy == 0:
                                self.lots_to_buy = 1

                            # Iterate through positions to check contract and update lots_to_buy accordingly
                            for position in self.positions:
                                if position.contract == contract:
                                    self.lots_to_buy -= position.pos

                            # Check if lots_to_buy exceeds available funds
                            if self.lots_to_buy * price > portfolio_value:
                                print(f"Not enough funds to buy for {contract}")
                            else:
                                print(f"Lots to buy for {contract}: {self.lots_to_buy}")

                        # 1° Creating graph
                        print(
                            '\nCollecting data, computing Bellman Ford algorithm, searching for arbitrage opportunity...')
                        self.build_graph()

                        # 2° Initializing distances between vertices
                        dist = [float("Inf")] * self.vertices
                        path = [float("Inf")] * self.vertices
                        dist[0] = 0
                        path[0] = 0
                        profit = 1

                        # 3° Relaxing all edges and checking for short distance with nested loops
                        for _ in range(self.vertices - 1):
                            for s, e, v in self.arches:
                                if dist[s] != float("Inf") and dist[s] + v < dist[e]:
                                    dist[e] = dist[s] + v
                                    path[e] = s

                        # 4° Detecting negative cycles
                        Neg_cycles = self.Negative_Cycle(dist, path)

                        # 5° Results, if there is a negative cycle --> computing possible profit
                        if not Neg_cycles:
                            print("\nNo arbitrage opportunity.")
                            # self.Display_Graph(path, 0, 0)

                        else:

                            for neg_cycle in Neg_cycles:
                                print("\nFound negative cycle:")
                                print('  ' + " --> ".join([self.tickers[i]
                                                           for i in neg_cycle[::-1]]))
                                prec = neg_cycle[-1]
                                for i in neg_cycle[-2::-1]:
                                    profit *= self.currency_matrix[prec][i]
                                    prec = i
                                profit = round(profit, 4)
                                print("  Profit: ", profit)

                                # Place trades based on the negative cycle
                                for i in range(len(neg_cycle) - 1):
                                    from_currency = self.tickers[neg_cycle[i]]
                                    to_currency = self.tickers[neg_cycle[i + 1]]
                                    contract = f"{from_currency}/{to_currency}"
                                    action = "BUY" if self.currency_matrix[neg_cycle[i]][
                                                          neg_cycle[i + 1]] > 1 else "SELL"
                                    price = self.prices[contract]['price']
                                    # Place the trade
                                    self.mkt_trade(contract, action, price)

    def reciprocate_price(self, price):
        return 1 / price

    def mkt_trade(self, contract, action, price):

        if contract in ['EUR/USD', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'USD/CNH', 'NZD/USD', 'USD/CHF']:
            rounded_price = round(price, 5)
        elif contract == ['USD/JPY', 'EUR/JPY', 'AUD/JPY', 'GBP/JPY', 'CAD/JPY', 'CHF/JPY']:
            rounded_price = round(price, 3)
        else:
            rounded_price = round(price, 5)

        print(
            "=========================================================== trading {} with action: {} price: {}"
            .format(contract, action, rounded_price))

        self.quantity = self.lots_to_buy

        order = self.market_order(action, self.quantity)

        self.placeOrder(self.get_order_id(), contract, order)

    @staticmethod
    def min_price_increment(price: float, contract: str):
        if contract in ['EUR/USD', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'USD/CNH', 'NZD/USD', 'USD/CHF']:
            return round(round(price / 0.00005) * 0.00005, 5)
        elif contract == ['USD/JPY', 'EUR/JPY', 'AUD/JPY', 'GBP/JPY', 'CAD/JPY', 'CHF/JPY']:
            return round(round(price / 0.005) * 0.005, 3)
        else:
            return price  # or raise an exception indicating an unsupported currency pair


app = App("127.0.0.1", 7497, 1)
try:
    app.main()
except KeyboardInterrupt:
    app.disconnect()
