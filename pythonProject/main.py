# import math
#
# class ArbitrageDetection:
#     def __init__(self, exchange_rates):
#         self.exchange_rates = exchange_rates
#         self.arches = []
#         self.tickers = []
#         self.currency_matrix = {}
#         self.vertices = 0
#
#     def define_arches(self, s, e, v):  # s=start, e=end, v=value
#         self.arches.append([s, e, v])
#         print(f"Defined edge: {s} -> {e} with value {v}")
#
#     def build_graph(self):
#         # Create a list of unique tickers (currencies)
#         currencies = set()
#         for from_currency, to_currency, rate in self.exchange_rates:
#             currencies.add(from_currency)
#             currencies.add(to_currency)
#         self.tickers = list(currencies)
#         self.vertices = len(self.tickers)
#
#         # Create a mapping from currency to index
#         currency_index = {currency: idx for idx, currency in enumerate(self.tickers)}
#
#         # Define the arches using the provided exchange rates
#         for from_currency, to_currency, rate in self.exchange_rates:
#             from_idx = currency_index[from_currency]
#             to_idx = currency_index[to_currency]
#             weight = -math.log(rate)
#             self.define_arches(from_idx, to_idx, weight)
#
#             # Build the currency matrix for profit calculation
#             if from_currency not in self.currency_matrix:
#                 self.currency_matrix[from_currency] = {}
#             self.currency_matrix[from_currency][to_currency] = rate
#
#     def Bellman_Ford(self):
#         print('\nCollecting data, computing Bellman Ford algorithm, searching for arbitrage opportunity...')
#         self.build_graph()
#
#         dist = [float("Inf")] * self.vertices
#         path = [float("Inf")] * self.vertices
#         dist[0] = 0
#         path[0] = 0
#
#         for _ in range(self.vertices - 1):
#             for s, e, v in self.arches:
#                 if dist[s] != float("Inf") and dist[s] + v < dist[e]:
#                     dist[e] = dist[s] + v
#                     path[e] = s
#
#         Neg_cycles = self.Negative_Cycle(dist, path)
#
#         if not Neg_cycles:
#             print("\nNo arbitrage opportunity.")
#         else:
#             for neg_cycle in Neg_cycles:
#                 print("\nFound negative cycle:")
#                 print('  ' + " --> ".join([self.tickers[i] for i in neg_cycle[::-1]]))
#                 profit = 1
#                 prec = neg_cycle[-1]
#                 for i in neg_cycle[-2::-1]:
#                     profit *= self.currency_matrix[self.tickers[prec]][self.tickers[i]]
#                     prec = i
#                 profit = round(profit, 4)
#                 print("  Profit: ", profit)
#
#     def Negative_Cycle(self, dist, path):
#         Neg_cycles = []
#         for s, e, v in self.arches:
#             if dist[s] + v < dist[e] and dist[s] != float("Inf"):
#                 neg_cycle = [e, s]
#                 aux = s
#
#                 while path[aux] not in neg_cycle:
#                     neg_cycle.append(path[aux])
#                     aux = path[aux]
#                 neg_cycle.append(path[aux])
#
#                 if neg_cycle[0] == neg_cycle[-1] and len(neg_cycle) > 3:
#                     Neg_cycles.append(neg_cycle)
#
#         return Neg_cycles if Neg_cycles else False
#
# # Given exchange rates as (from_currency, to_currency, rate)
# exchange_rates = [
#     ("USD", "EUR", 0.919399977), ("USD", "AUD", 1.493430018), ("USD", "GBP", 0.787699997), ("USD", "CAD", 1.361739993), ("USD", "CHF", 0.909089983),
#     ("EUR", "USD", 1.087665915), ("EUR", "AUD", 1.623800039), ("EUR", "GBP", 0.85650003), ("EUR", "CAD", 1.480489969), ("EUR", "CHF", 0.988449991),
#     ("AUD", "USD", 0.669599533), ("AUD", "EUR", 0.615660012), ("AUD", "GBP", 0.527320027), ("AUD", "CAD", 0.911689997), ("AUD", "CHF", 0.608669996),
#     ("GBP", "USD", 1.269518852), ("GBP", "EUR", 1.167389989), ("GBP", "AUD", 1.895799994), ("GBP", "CAD", 1.728489995), ("GBP", "CHF", 1.15395999),
#     ("CAD", "USD", 0.734386921), ("CAD", "EUR", 0.675199986), ("CAD", "AUD", 1.096500039), ("CAD", "GBP", 0.578100026), ("CAD", "CHF", 0.667620003),
#     ("CHF", "USD", 1.100001097), ("CHF", "EUR", 1.011100054), ("CHF", "AUD", 1.64230001), ("CHF", "GBP", 0.865999997), ("CHF", "CAD", 1.496899962)
# ]
#
# # Create an instance of ArbitrageDetection
# arbitrage_detector = ArbitrageDetection(exchange_rates)
# arbitrage_detector.Bellman_Ford()

# import math
#
# # List of currencies
# currencies = ['USD', 'EUR', 'AUD', 'GBP', 'CAD', 'CHF']
#
# # Exchange rates
# exchange_rates = {
#     ('USD', 'EUR'): 0.919399977,
#     ('USD', 'AUD'): 1.493430018,
#     ('USD', 'GBP'): 0.787699997,
#     ('USD', 'CAD'): 1.361739993,
#     ('USD', 'CHF'): 0.909089983,
#     ('EUR', 'USD'): 1.087665915,
#     ('EUR', 'AUD'): 1.623800039,
#     ('EUR', 'GBP'): 0.85650003,
#     ('EUR', 'CAD'): 1.480489969,
#     ('EUR', 'CHF'): 0.988449991,
#     ('AUD', 'USD'): 0.669599533,
#     ('AUD', 'EUR'): 0.615660012,
#     ('AUD', 'GBP'): 0.527320027,
#     ('AUD', 'CAD'): 0.911689997,
#     ('AUD', 'CHF'): 0.608669996,
#     ('GBP', 'USD'): 1.269518852,
#     ('GBP', 'EUR'): 1.167389989,
#     ('GBP', 'AUD'): 1.895799994,
#     ('GBP', 'CAD'): 1.728489995,
#     ('GBP', 'CHF'): 1.15395999,
#     ('CAD', 'USD'): 0.734386921,
#     ('CAD', 'EUR'): 0.675199986,
#     ('CAD', 'AUD'): 1.096500039,
#     ('CAD', 'GBP'): 0.578100026,
#     ('CAD', 'CHF'): 0.667620003,
#     ('CHF', 'USD'): 1.100001097,
#     ('CHF', 'EUR'): 1.011100054,
#     ('CHF', 'AUD'): 1.64230001,
#     ('CHF', 'GBP'): 0.865999997,
#     ('CHF', 'CAD'): 1.496899962
# }
#
# # Create graph with negative log weights
# edges = []
# for (src, dest), rate in exchange_rates.items():
#     weight = -math.log(rate)
#     edges.append((src, dest, weight))
#
# # Print edges with weights
# print("Edges with weights (negative log of exchange rates):")
# for src, dest, weight in edges:
#     print(f"{src} -> {dest}: {weight:.6f}")
#
# # Bellman-Ford algorithm to find negative cycle
# def bellman_ford(currencies, edges, start):
#     distance = {currency: float('inf') for currency in currencies}
#     predecessor = {currency: None for currency in currencies}
#     distance[start] = 0
#
#     for _ in range(len(currencies) - 1):
#         for src, dest, weight in edges:
#             if distance[src] + weight < distance[dest]:
#                 distance[dest] = distance[src] + weight
#                 predecessor[dest] = src
#
#     # Check for negative cycle
#     for src, dest, weight in edges:
#         if distance[src] + weight < distance[dest]:
#             cycle = []
#             visited = set()
#             while dest not in visited:
#                 visited.add(dest)
#                 dest = predecessor[dest]
#             cycle_start = dest
#             cycle.append(cycle_start)
#             next_currency = predecessor[cycle_start]
#             while next_currency != cycle_start:
#                 cycle.append(next_currency)
#                 next_currency = predecessor[next_currency]
#             cycle.append(cycle_start)
#             cycle.reverse()
#             return cycle
#
#     return None
#
# start_currency = 'USD'
# negative_cycle = bellman_ford(currencies, edges, start_currency)
#
# if negative_cycle:
#     print("\nNegative cycle detected:")
#     for i in range(len(negative_cycle)):
#         print(negative_cycle[i], end=" -> " if i != len(negative_cycle) - 1 else "\n")
#     profit = math.exp(-sum(-math.log(exchange_rates[(negative_cycle[i], negative_cycle[i + 1])])
#                           for i in range(len(negative_cycle) - 1)))
#     print(f"Profit: {profit}")
# else:
#     print("No negative cycle detected.")

import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class App:
    def __init__(self):
        labels = ['USD', 'EUR', 'AUD', 'GBP', 'CAD', 'CHF']
        self.vertices = len(labels)
        self.arches = []
        self.tickers = labels
        self.currency_matrix = np.zeros((self.vertices, self.vertices))

    def define_arches(self, s, e, v, pair):
        self.arches.append([s, e, v])
        print(f"Defined edge: {s} ({self.tickers[s]}) -> {e} ({self.tickers[e]}) [{pair}] with value {v}")

    def build_graph(self):
        exchange_rates = {
            'CHFCAD': 1.49354,
            'CHFGBP': 0.85859,
            'CHFAUD': 1.64927,
            'CHFEUR': 1.00818,
            'CHFUSD': 1.09391,
            'CADCHF': 0.66956,
            'CADGBP': 0.57487,
            'CADAUD': 1.10426,
            'CADEUR': 0.67503,
            'CADUSD': 0.73244,
            'GBPCHF': 1.16472,
            'GBPCAD': 1.73955,
            'GBPAUD': 1.92092,
            'GBPEUR': 1.17424,
            'GBPUSD': 1.2741,
            'AUDCHF': 0.60633,
            'AUDCAD': 0.90559,
            'AUDGBP': 0.52059,
            'AUDEUR': 0.6113,
            'AUDUSD': 0.66328,
            'EURCHF': 0.99189,
            'EURCAD': 1.48142,
            'EURGBP': 0.85162,
            'EURAUD': 1.63587,
            'EURUSD': 1.08506,
            'USDCHF': 0.91416,
            'USDCAD': 1.36531,
            'USDGBP': 0.78487,
            'USDAUD': 1.50768,
            'USDEUR': 0.92163
        }

        tickers_pairs = [
            ("CHF", "CAD"), ("CHF", "GBP"), ("CHF", "AUD"), ("CHF", "EUR"), ("CHF", "USD"),
            ("CAD", "CHF"), ("CAD", "GBP"), ("CAD", "AUD"), ("CAD", "EUR"), ("CAD", "USD"),
            ("GBP", "CHF"), ("GBP", "CAD"), ("GBP", "AUD"), ("GBP", "EUR"), ("GBP", "USD"),
            ("AUD", "CHF"), ("AUD", "CAD"), ("AUD", "GBP"), ("AUD", "EUR"), ("AUD", "USD"),
            ("EUR", "CHF"), ("EUR", "CAD"), ("EUR", "GBP"), ("EUR", "AUD"), ("EUR", "USD"),
            ("USD", "CHF"), ("USD", "CAD"), ("USD", "GBP"), ("USD", "AUD"), ("USD", "EUR")
        ]

        tickers_indices = {ticker: idx for idx, ticker in enumerate(self.tickers)}

        for pair in tickers_pairs:
            start, end = pair
            s_idx = tickers_indices[start]
            e_idx = tickers_indices[end]
            ticker = start + end

            if ticker in exchange_rates:
                rate = exchange_rates[ticker]
                self.currency_matrix[s_idx][e_idx] = rate
                self.define_arches(s_idx, e_idx, round(-math.log(rate, 10), 5), ticker)
            else:
                self.currency_matrix[s_idx][e_idx] = 1
                self.define_arches(s_idx, e_idx, 0, ticker)

    def Bellman_Ford(self):
        print('\nCollecting data, computing Bellman Ford algorithm, searching for arbitrage opportunity...')
        self.build_graph()
        dist = [float("Inf")] * self.vertices
        path = [float("Inf")] * self.vertices
        dist[0] = 0
        path[0] = 0
        profit = 1

        for _ in range(self.vertices - 1):
            for s, e, v in self.arches:
                if dist[s] != float("Inf") and dist[s] + v < dist[e]:
                    dist[e] = dist[s] + v
                    path[e] = s

        Neg_cycles = self.Negative_Cycle(dist, path)

        if not Neg_cycles:
            print("\nNo arbitrage opportunity.")
            self.Display_Graph(path, 0, 0)
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
                self.Display_Graph(neg_cycle, profit, 1)

    def Negative_Cycle(self, dist, path):
        Neg_cycles = []
        flag = False
        for s, e, v in self.arches:
            if dist[s] + v < dist[e] and dist[s] != float("Inf"):
                neg_cycle = [e, s]
                aux = s
                while path[aux] not in neg_cycle:
                    neg_cycle.append(path[aux])
                    aux = path[aux]
                neg_cycle.append(path[aux])
                if neg_cycle[0] == neg_cycle[-1] and len(neg_cycle) > 3:
                    Neg_cycles.append(neg_cycle)
                    flag = True

        if flag:
            return Neg_cycles
        else:
            return False

    def Display_Graph(self, path, profit, flag):
        path_edges = []
        graph_view = nx.MultiDiGraph()
        for s, e, v in self.arches:
            graph_view.add_edge(s, e, weight=round(10 ** (-v), 4))

        pos = nx.circular_layout(graph_view)

        if flag == 0:
            plt.title('NO Arbitrage Opportunity, NO Negative Cycle', fontsize=20)
        else:
            for i in range(len(path) - 1):
                path_edges.append((path[i + 1], path[i]))
            plt.text(-1.3, -1.3,
                     "Found Negative Cycle: \n\n" + '  ' + " --> ".join([self.tickers[i] for i in path[::-1]])
                     + "\n\nProfit: " + str(profit),
                     bbox=dict(boxstyle="square", facecolor="white"), size=12.5)
            plt.title('ARBITRAGE OPPORTUNITY', fontsize=20)

        edge_labels = dict([((u, v,), d['weight']) for u, v, d in graph_view.edges(data=True)])
        edge_colors = ['black' if not edge in path_edges else 'red' for edge in graph_view.edges()]
        node_colors = ['green' for path in graph_view.nodes()]
        labels = {i: self.tickers[i] for i in range(len(self.tickers))}

        nx.draw_networkx_edge_labels(graph_view, pos, label_pos=0.28, edge_labels=edge_labels)
        nx.draw(graph_view, pos, node_size=1500, node_color=node_colors, edge_color=edge_colors, with_labels=False,
                connectionstyle='arc3, rad = 0.1')
        nx.draw_networkx_labels(graph_view, pos, labels, font_size=16, font_color='black')
        plt.show()


def main():
    app = App()
    app.Bellman_Ford()


if __name__ == "__main__":
    main()