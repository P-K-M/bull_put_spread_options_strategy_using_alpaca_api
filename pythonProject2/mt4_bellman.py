import math
import numpy as np
import yfinance as yf
import pandas as pd
import time
import csv
from pathlib import Path

num = 6

class App:
    def __init__(self):
        labels = ['USD', 'EUR', 'AUD', 'GBP', 'CAD', 'CHF']
        self.vertices = num
        self.arches = []
        self.tickers = labels
        self.currency_matrix = np.zeros((num, num))
        self.mt4_location = "C:\\Users\\STEVE\\AppData\\Roaming\\MetaQuotes\\Terminal\\B313D8B5E3EEA7D6CF15F515B9125C20\\MQL4\\Files\\LastSignal.csv"
        self.path = str(Path(__file__).parent.absolute()).replace("\\", "\\\\") + "\\\\"

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
                    ticker = self.tickers[i] + self.tickers[j] + '=X'
                    print(ticker)
                    fx_data = yf.download(tickers=ticker, period='1d', interval='1m')

                    # Check if fx_data is not empty
                    if not fx_data.empty:
                        # Assuming the close price represents the exchange rate
                        data = fx_data['Close'].iloc[-1]
                        print(data)
                    else:
                        # Handle the case where fx_data is empty
                        print(f"No data available for {ticker}. Using default value.")
                        data = 1  # Set a default value or handle it according to your requirements

                # Keeping track of exchange rates
                self.currency_matrix[i][j] = data
                self.define_arches(i, j, round(-math.log(data, 10), 5))

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
        else:
            for neg_cycle in Neg_cycles:
                print("\nFound negative cycle:")
                print('  ' + " --> ".join([self.tickers[i] for i in neg_cycle[::-1]]))
                prec = neg_cycle[-1]
                for i in neg_cycle[-2::-1]:
                    profit *= self.currency_matrix[prec][i]
                    prec = i
                profit = round(profit, 4)
                print("  Profit: ", profit)
                
                # Place trades on MT4
                self.place_mt4_order(neg_cycle)

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

    def place_mt4_order(self, cycle):
        for i in range(len(cycle) - 1):
            s = cycle[i]
            e = cycle[i + 1]
            pair = self.tickers[s] + self.tickers[e] + '.a'
            signal = 'OP_BUY' if self.currency_matrix[s][e] < 1 else 'OP_SELL'
            sl = self.currency_matrix[s][e] * 0.99  # Example stop loss
            tp = self.currency_matrix[s][e] * 1.01  # Example take profit
            rounding = 3 if 'JPY' in pair else 5
            order = f"{pair},{signal},{sl:.{rounding}f},{tp:.{rounding}f},,,"
            with open(self.mt4_location, 'w', newline='') as file:
                writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='-')
                writer.writerow([order])
            print(f"Placed {signal} order for {pair} with SL: {sl:.{rounding}f}, TP: {tp:.{rounding}f}")
            time.sleep(5)  # Sleep to avoid overwhelming the system

def main():
    app = App()
    app.Bellman_Ford()

if __name__ == "__main__":
    main()
