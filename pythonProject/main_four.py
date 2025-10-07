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

        self.symbols = ['USD.EUR', 'USD.AUD', 'USD.GBP', 'USD.CAD', 'USD.JPY', 'USD.CHF', 'EUR.USD',
                        'EUR.AUD', 'EUR.GBP', 'EUR.CAD', 'EUR.JPY', 'EUR.CHF', 'AUD.USD', 'AUD.EUR',
                        'AUD.GBP', 'AUD.CAD', 'AUD.JPY', 'AUD.CHF', 'GBP.USD', 'GBP.EUR', 'GBP.AUD',
                        'GBP.CAD', 'GBP.JPY', 'GBP.CHF', 'CAD.USD', 'CAD.EUR', 'CAD.AUD', 'CAD.GBP',
                        'CAD.JPY', 'CAD.CHF', 'JPY.USD', 'JPY.EUR', 'JPY.AUD', 'JPY.GBP', 'JPY.CAD',
                        'JPY.CHF', 'CHF.USD', 'CHF.EUR', 'CHF.AUD', 'CHF.GBP', 'CHF.CAD', 'CHF.JPY']

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

        for i, pair in enumerate(self.symbols):
            req_id = self.get_unique_id() + i

            # Creating contract instance using fxPair method
            contract = self.fxPair(pair)

            # Check if the contract is available
            if contract in available_pairs:
                self.market_data[req_id] = PriceInformation(contract)

                self.reqMarketDataType(1)
                self.reqMktData(req_id, contract, "", False, False, [])
            else:
                print(f"Contract {contract} is not available on Interactive Brokers. Trying reciprocal symbol.")

                # Get the reciprocal symbol
                # base_currency, quote_currency = contract.split(".")
                reciprocal_contract = f"{contract.currency}.{contract.symbol}"

                # Check if the reciprocal contract is available
                if reciprocal_contract in available_pairs:
                    # Create a Contract object for the reciprocal symbol using fxPair method
                    reciprocal_contract_obj = self.fxPair(reciprocal_contract)

                    self.market_data[req_id] = PriceInformation(reciprocal_contract)

                    self.reqMarketDataType(1)
                    self.reqMktData(req_id, reciprocal_contract_obj, "", False, False, [])
                else:
                    print(
                        f"Reciprocal contract {reciprocal_contract} is also not available on Interactive Brokers. Skipping market data request.")
            # Get the rounded prices for the contract
            rounded_prices = self.get_midpoint_price(contract)

            # Print out the rounded prices
            print("Rounded Prices for {}: {}".format(contract, rounded_prices))
        # self.get_midpoint_price(contract)

        # self.Bellman_Ford()

    # Get Midpoint Data
    def get_midpoint_price(self, contract):

        # self.get_historical_data()

        # Load the currency pairs from the csv file
        fx_pairs_df = pd.read_csv("C:/Users/STEVE/Desktop/fx_pairs.csv")
        available_pairs = set(fx_pairs_df["Symbol"])

        if contract in available_pairs:

            req_id = self.get_unique_id()

            print('The next unique id is: ', req_id)
            # Print the contract symbol to check its value
            print(f"Contract: {contract}")

            current_datetime = datetime.datetime.today().astimezone(self.nyc)

            # Convert current_datetime to a new datetime object with seconds and microseconds set to zero
            current_datetime = current_datetime.replace(second=0, microsecond=0)

            ts = current_datetime.strftime("%Y%m%d %H:%M:%S")

            # Check if the contract is a reciprocal pair
            # base_currency, quote_currency = contract.split(".")
            reciprocal_contract = f"{contract.currency}.{contract.symbol}"
            if reciprocal_contract in available_pairs:
                # If it's a reciprocal pair, get the midpoint price of the reciprocal pair
                if req_id in self.market_data:
                    # Access the market data using req_id and calculate the price
                    market_data = self.market_data[req_id]
                    if market_data.Bid is not None and market_data.Ask is not None:
                        midpoint_price = (market_data.Bid + market_data.Ask) / 2
                        reciprocal_midpoint_price = 1 / midpoint_price

                        if contract in ['USD.JPY', 'EUR.JPY', 'AUD.JPY', 'GBP.JPY', 'CAD.JPY', 'CHF.JPY']:
                            rounded_price = round(1 / reciprocal_midpoint_price, 3)
                        else:
                            rounded_price = round(1 / reciprocal_midpoint_price, 5)

                        print("Reciprocal Midpoint price:", rounded_price)

                        # Add the relevant information to self.prices dictionary under the reciprocal contract name
                        self.prices[reciprocal_contract] = {
                            'req_id': req_id,
                            'ts': ts,
                            'price': rounded_price
                        }

                        time.sleep(1)

                        self.price = self.prices[reciprocal_contract]['price']
                        self.ts = self.prices[reciprocal_contract]['ts']

                        # Additional code
                        stop_price = self.find_stop(
                            self.price, self.historical_data[contract], self.ts
                        )
                        self.stop_prices[contract] = stop_price

                        # Contract has passed all checks; figure out how much to buy

                        portfolio_value = self.portfolio_value

                        print("Portfolio Value:", portfolio_value)

                        # Calculate quantity based on formula
                        if portfolio_value is not None:  # Check if portfolio_value is not None

                            self.lots_to_buy = portfolio_value * self.risk // (self.price - stop_price)

                            if self.lots_to_buy == 0:
                                self.lots_to_buy = 1

                            # Iterate through positions to check contract and update lots_to_buy accordingly
                            for position in self.positions:
                                if position.contract == contract:
                                    self.lots_to_buy -= position.pos

                            # Check if lots_to_buy exceeds available funds
                            if self.lots_to_buy * self.price > portfolio_value:
                                print(f"Not enough funds to buy for {contract}")
                            else:
                                print(f"Lots to buy for {contract}: {self.lots_to_buy}")

                        return rounded_price  # Return the calculated rounded_price

            else:
                # If it's not a reciprocal pair, continue with the original logic
                if req_id in self.market_data:
                    # Access the market data using req_id and calculate the price
                    market_data = self.market_data[req_id]
                    if market_data.Bid is not None and market_data.Ask is not None:
                        midpoint_price = (market_data.Bid + market_data.Ask) / 2

                        if contract in ['USD.JPY', 'EUR.JPY', 'AUD.JPY', 'GBP.JPY', 'CAD.JPY', 'CHF.JPY']:
                            rounded_price = round(midpoint_price, 3)
                        else:
                            rounded_price = round(midpoint_price, 5)

                        print("Midpoint price:", rounded_price)

                        # Add the relevant information to self.prices dictionary under the original contract name
                        self.prices[contract] = {
                            'req_id': req_id,
                            'ts': ts,
                            'price': midpoint_price
                        }

                        time.sleep(1)

                        self.price = self.prices[contract]['price']
                        self.ts = self.prices[contract]['ts']

                        # Additional code
                        stop_price = self.find_stop(
                            self.price, self.historical_data[contract], self.ts
                        )
                        self.stop_prices[contract] = stop_price

                        # Contract has passed all checks; figure out how much to buy
                        portfolio_value = self.portfolio_value

                        print("Portfolio Value:", portfolio_value)

                        # Calculate quantity based on formula
                        if portfolio_value is not None:  # Check if portfolio_value is not None

                            self.lots_to_buy = portfolio_value * self.risk // (self.price - stop_price)

                            if self.lots_to_buy == 0:
                                self.lots_to_buy = 1

                            # Iterate through positions to check contract and update lots_to_buy accordingly
                            for position in self.positions:
                                if position.contract == contract:
                                    self.lots_to_buy -= position.pos

                            # Check if lots_to_buy exceeds available funds
                            if self.lots_to_buy * self.price > portfolio_value:
                                print(f"Not enough funds to buy for {contract}")
                            else:
                                print(f"Lots to buy for {contract}: {self.lots_to_buy}")

                        return rounded_price  # Return the calculated rounded_price

    def build_graph(self):
        for i, source_currency in enumerate(self.tickers):
            for j, target_currency in enumerate(self.tickers):
                if i == j:
                    self.currency_matrix[i][j] = 1
                else:
                    pair = f"{source_currency}.{target_currency}"
                    if pair in self.symbols:
                        contract = self.fxPair(pair)
                        data = self.get_midpoint_price(contract)
                        if data is not None:
                            self.currency_matrix[i][j] = data
                            self.define_arches(i, j, round(-math.log(data, 10), 5))
                        else:
                            print(f"Data for pair {pair} is None.")
                    else:
                        reciprocal_pair = f"{target_currency}.{source_currency}"
                        reciprocal_contract = self.fxPair(reciprocal_pair)
                        if reciprocal_contract in self.symbols:
                            data = self.get_midpoint_price(reciprocal_contract)
                            if data is not None:
                                reciprocal_data = 1 / data
                                self.currency_matrix[i][j] = reciprocal_data
                                self.define_arches(i, j, round(-math.log(reciprocal_data, 10), 5))
                            else:
                                print(f"Data for reciprocal pair {reciprocal_pair} is None.")
                        else:
                            print(f"Reciprocal pair {reciprocal_pair} not available.")
                            continue

                    # self.define_arches(i, j, round(-math.log(data, 10), 5))
                    # print(f"Defined edge: {source_currency} -> {target_currency} with value {data}")

    def Bellman_Ford(self):

        # 1° Creating graph
        print('\nCollecting data, computing Bellman Ford algorithm, searching for arbitrage opportunity...')
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
            self.Display_Graph(path, 0, 0)

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
                self.Display_Graph(neg_cycle, profit, 1)

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

    def Display_Graph(self, path, profit, flag):

        path_edges = []
        graph_view = nx.MultiDiGraph()

        for s, e, v in self.arches:
            graph_view.add_edge(s, e, weight=round(10**(-v), 4))

        pos = nx.circular_layout(graph_view)

        if flag == 0:

            plt.title('NO Arbitrage Opportunity, NO Negative Cycle', fontsize=20)

        else:

            # Colouring the negative cycle
            for i in range(len(path)-1):
                path_edges.append((path[i+1], path[i]))

            plt.text(-1.3, -1.3, "Found Negative Cycle: \n\n" + '  ' + " --> ".join([self.tickers[i] for i in path[::-1]])
                     + "\n\nProfit: " + str(profit),
                     bbox=dict(boxstyle="square", facecolor="white"), size=12.5)
            plt.title('ARBITRAGE OPPORTUNITY', fontsize=20)

        edge_labels = dict([((u, v,), d['weight'])
                            for u, v, d in graph_view.edges(data=True)])
        edge_colors = [
            'black' if not edge in path_edges else 'red' for edge in graph_view.edges()]
        node_colors = ['green' for path in graph_view.nodes()]

        labels = {}
        for i in range(len(self.tickers)):
            labels[i] = self.tickers[i]

        nx.draw_networkx_edge_labels(
            graph_view, pos, label_pos=0.28, edge_labels=edge_labels)
        nx.draw(graph_view, pos, node_size=1500, node_color=node_colors, edge_color=edge_colors, with_labels=False,
                connectionstyle='arc3, rad = 0.1')
        nx.draw_networkx_labels(graph_view, pos, labels,
                                font_size=16, font_color='black')

        plt.show()

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