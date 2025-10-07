# bellman_ford.py
import sys
import json
import logging

# Set up logging
logging.basicConfig(filename='bellman_ford.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

class App:
    def __init__(self, vertices, arches, tickers, currency_matrix):
        self.vertices = vertices
        self.arches = arches
        self.tickers = tickers
        self.currency_matrix = currency_matrix

    def Bellman_Ford(self):
        logging.info('Collecting data, computing Bellman Ford algorithm, searching for arbitrage opportunity...')

        # Initialize distances and paths
        dist = [float("Inf")] * self.vertices
        path = [float("Inf")] * self.vertices
        dist[0] = 0
        path[0] = 0
        profit = 1

        logging.debug("Initial distances: %s", dist)
        logging.debug("Initial paths: %s", path)

        # Relax all edges
        for _ in range(self.vertices - 1):
            for s, e, v in self.arches:
                if dist[s] != float("Inf") and dist[s] + v < dist[e]:
                    dist[e] = dist[s] + v
                    path[e] = s
                    logging.debug("Updated distance for %d: %f, path: %d", e, dist[e], path[e])

        # Detect negative cycles
        Neg_cycles = self.Negative_Cycle(dist, path)

        if not Neg_cycles:
            logging.info("No arbitrage opportunity.")
            return []

        else:
            results = []
            for neg_cycle in Neg_cycles:
                cycle_str = " --> ".join([self.tickers[i] for i in neg_cycle[::-1]])
                logging.info("Found negative cycle:\n  %s", cycle_str)
                prec = neg_cycle[-1]
                for i in neg_cycle[-2::-1]:
                    profit *= self.currency_matrix[prec][i]
                    prec = i
                profit = round(profit, 4)
                logging.info("  Profit: %f", profit)
                results.append({"cycle": cycle_str, "profit": profit})
            return results

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
            logging.info("Negative cycles detected: %s", Neg_cycles)
            return Neg_cycles
        else:
            logging.info("No negative cycles detected.")
            return False

if __name__ == "__main__":
    logging.info("Reading input data...")
    data = json.load(sys.stdin)
    logging.debug("Input data: %s", data)
    app = App(data['vertices'], data['arches'], data['tickers'], data['currency_matrix'])
    results = app.Bellman_Ford()
    logging.info("Output results: %s", results)
    json.dump(results, sys.stdout)
