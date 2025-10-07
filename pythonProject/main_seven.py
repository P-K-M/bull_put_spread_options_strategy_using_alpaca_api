import math

from itertools import combinations

from ibapi.contract import Contract

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
        self.macd_size = 100
        self.bar_size = '1 min'
        self.historical_bar_size = self.bar_size

        if int(self.bar_size.split()[0]) * self.macd_size >= 6000:
            self.historical_data_duration = "{} S".format(int(self.bar_size.split()[0]) * self.macd_size)
        else:
            self.historical_data_duration = "{} S".format(6000)

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
            self.CHF_JPY_FX(), self.EUR_CHF_FX(), self.AUD_CHF_FX(), self.GBP_CHF_FX(), self.CAD_CHF_FX()
        ]

        self.bellman_pairs = [
            self.USD_EUR_FX(), self.USD_AUD_FX(), self.USD_GBP_FX(), self.USD_CAD_FX(), self.USD_JPY_FX(),
            self.USD_CHF_FX(), self.EUR_USD_FX(), self.EUR_AUD_FX(), self.EUR_GBP_FX(), self.EUR_CAD_FX(),
            self.EUR_JPY_FX(), self.EUR_CHF_FX(), self.AUD_USD_FX(), self.AUD_EUR_FX(), self.AUD_GBP_FX(),
            self.AUD_CAD_FX(), self.AUD_JPY_FX(), self.AUD_CHF_FX(), self.GBP_USD_FX(), self.GBP_EUR_FX(),
            self.GBP_AUD_FX(), self.GBP_CAD_FX(), self.GBP_JPY_FX(), self.GBP_CHF_FX(), self.CAD_USD_FX(),
            self.CAD_EUR_FX(), self.CAD_AUD_FX(), self.CAD_GBP_FX(), self.CAD_JPY_FX(), self.CAD_CHF_FX(),
            self.JPY_USD_FX(), self.JPY_EUR_FX(), self.JPY_AUD_FX(), self.JPY_GBP_FX(), self.JPY_CAD_FX(),
            self.JPY_CHF_FX(), self.CHF_USD_FX(), self.CHF_EUR_FX(), self.CHF_AUD_FX(), self.CHF_GBP_FX(),
            self.CHF_CAD_FX(), self.CHF_JPY_FX()
        ]

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

    def get_historical_data(self, currency_pairs):
        print('Getting historical data...')

        count = 0

        for contract in self.currency_pairs:
            bars = self.get_historical_market_data(contract, bar_size=self.historical_bar_size,
                                                   duration=self.historical_data_duration,
                                                   data_type="MIDPOINT")

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

    def find_stop(self, current_value, historical_data, now):

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
            print("its the weekend no trading today")
            return None

        seconds_to_wait = (self.pnl_time - self.current_dt).total_seconds()
        Timer(seconds_to_wait, self.get_pnl).start()

        if not self.isConnected():
            self.reconnect()

        time.sleep(2)

        print("Printing dataframe:")
        print(self.dataframe)  # This line prints the dataframe

        for i, contract in enumerate(self.currency_pairs):
            req_id = self.get_unique_id() + i

            self.market_data[req_id] = PriceInformation(contract)

            self.reqMarketDataType(1)
            self.reqMktData(req_id, contract, "", False, False, [])

        self.Bellman_Ford()

    # Get Midpoint Data
    def get_midpoint_price(self):

        self.get_historical_data(self.currency_pairs)
        time.sleep(1)
        self.fetchAccountUpdates()
        portfolio_value = self.portfolio_value
        print("Current Portfolio Value:", portfolio_value)

        for i, contract in enumerate(self.currency_pairs):

            req_id = self.get_unique_id() + i

            # contract = self.fxPair(pair)

            print('The next unique id is: ', req_id)

            print(f"Contract: {contract}")
            current_datetime = datetime.datetime.today().astimezone(self.nyc)
            current_datetime = current_datetime.replace(second=0, microsecond=0)
            ts = current_datetime.strftime("%Y%m%d %H:%M:%S")

            if req_id in self.market_data:
                market_data = self.market_data[req_id]
                if market_data.Bid is not None and market_data.Ask is not None:
                    midpoint_price = (market_data.Bid + market_data.Ask) / 2
                    if contract in ['USD.JPY', 'EUR.JPY', 'AUD.JPY', 'GBP.JPY', 'CAD.JPY', 'CHF.JPY']:
                        rounded_price = round(midpoint_price, 3)
                    else:
                        rounded_price = round(midpoint_price, 5)
                    print("Midpoint price:", rounded_price)

                    self.prices[contract] = {
                        'req_id': req_id,
                        'ts': ts,
                        'price': midpoint_price
                    }

                    time.sleep(1)

                    self.price = self.prices[contract]['price']
                    self.ts = self.prices[contract]['ts']

                    stop_price = self.find_stop(
                        self.price, self.historical_data[contract], self.ts
                    )

                    self.stop_prices[contract] = stop_price

                    portfolio_value = self.portfolio_value
                    print("Portfolio Value:", portfolio_value)

                    if portfolio_value is not None:
                        self.lots_to_buy = portfolio_value * self.risk // (self.price - stop_price)
                        if self.lots_to_buy == 0:
                            self.lots_to_buy = 1
                        for position in self.positions:
                            if position.contract == contract:
                                self.lots_to_buy -= position.pos
                        if self.lots_to_buy * self.price > portfolio_value:
                            print(f"Not enough funds to buy for {contract}")
                        else:
                            print(f"Lots to buy for {contract}: {self.lots_to_buy}")

    # Building a matrix data structure and appending edges to graph
    def build_graph(self):
        for i in range(self.vertices):
            for j in range(self.vertices):
                if i == j:
                    # The diagonal is always equal to 1
                    data = 1
                else:
                    # Constructing currency pair
                    contract = self.currency_pairs[i]
                    print(contract)

                    # Retrieve exchange rate from self.prices if available
                    if contract in self.prices[contract]:
                        data = self.prices[contract]['price']
                    else:
                        # Handle the case where exchange rate data is not available
                        print(f"No data available for {contract}. Using default value.")
                        data = 1  # Set a default value or handle it according to your requirements

                # Keeping track of exchange rates
                self.currency_matrix[i][j] = data
                self.define_arches(i, j, round(-math.log(data, 10), 5))

    def Bellman_Ford(self):

        while self.current_dt < self.stop_time:

            self.get_midpoint_price()

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

                    # # Place trades based on the negative cycle
                    # for i in range(len(neg_cycle) - 1):
                    #     from_currency = self.tickers[neg_cycle[i]]
                    #     to_currency = self.tickers[neg_cycle[i + 1]]
                    #     contract = f"{from_currency}/{to_currency}"
                    #     action = "BUY" if self.currency_matrix[neg_cycle[i]][
                    #                           neg_cycle[i + 1]] > 1 else "SELL"
                    #     price = self.prices[contract]['price']
                    #     # Place the trade
                    #     self.mkt_trade(contract, action, price)

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

app = App("127.0.0.1", 7497, 1)
try:
    app.main()
except KeyboardInterrupt:
    app.disconnect()