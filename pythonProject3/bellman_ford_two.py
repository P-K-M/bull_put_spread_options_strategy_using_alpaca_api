import math
import numpy as np
import logging
import zmq
import mysql.connector

# Configure logging
logging.basicConfig(filename='mql4_python_connection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

num = 6

class App:
    def __init__(self):
        labels = ['USD', 'EUR', 'AUD', 'GBP', 'CAD', 'CHF']
        self.vertices = num
        self.arches = []
        self.tickers = labels
        self.currency_matrix = np.zeros((num, num))

        # Set up ZeroMQ to receive data from MT4
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://localhost:5555")  # Listening on port 5555
            logging.info("ZeroMQ connection established on port 5555.")
        except Exception as e:
            logging.error(f"ZeroMQ connection failed: {str(e)}")
            raise

        # Establish connection to the MySQL database
        self.db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Panama@7321",
            database="forex_data"
        )

    # Get midpoint price from MySQL database
    def get_midpoint(self, symbol):
        cursor = self.db_connection.cursor()
        query = "SELECT midpoint FROM currency_prices WHERE symbol = %s ORDER BY timestamp DESC LIMIT 1"
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        cursor.close()
        if result:
            return result[0]
        else:
            print(f"No data available for {symbol}. Using default value.")
            return 1  # Default value if no data is found

    def define_arches(self, s, e, v):
        self.arches.append([s, e, v])
        logging.info(f"Defined edge: {s} -> {e} with value {v}")
        print(f"Defined edge: {s} -> {e} with value {v}")

    def build_graph(self):
        currency_data = {}
        for i in range(self.vertices):
            for j in range(self.vertices):
                if i == j:
                    data = 1  # The diagonal is always 1
                else:
                    symbol = f"{self.tickers[i]}{self.tickers[j]}.a"

                    print(symbol)

                    if symbol not in currency_data:
                        currency_data[symbol] = self.get_midpoint(symbol)
                    data = currency_data[symbol] if currency_data[symbol] else 1  # Default to 1 if no data

                # Set exchange rate and define arches consistently
                self.currency_matrix[i][j] = data
                self.define_arches(i, j, round(-math.log(data, 10), 5))

    def Bellman_Ford(self):
        # 1° Creating graph
        logging.info("Starting Bellman-Ford algorithm...")
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
            logging.info("No arbitrage opportunity detected.")
            print("\nNo arbitrage opportunity.")
        else:
            for neg_cycle in Neg_cycles:
                # Construct a message to send to MQL4
                cycle_str = ' --> '.join([self.tickers[i] for i in neg_cycle[::-1]])
                print("\nFound negative cycle:")
                print(f"  {cycle_str}")

                # Calculate profit for the cycle
                prec = neg_cycle[-1]
                for i in neg_cycle[-2::-1]:
                    profit *= self.currency_matrix[prec][i]
                    prec = i
                profit = round(profit, 4)

                # Log and print the profit
                logging.info(f"Profit: {profit}")
                print("  Profit: ", profit)

                # Send the message to MQL4
                message = f"Cycle: {cycle_str}; Profit: {profit}"
                response = self.send_to_mql4(message)
                logging.info(f"Sent to MQL4: {message}; Response: {response}")

    def send_to_mql4(self, message):
        try:
            self.socket.send_string(message)
            response = self.socket.recv_string()  # Wait for reply from MQL4
            return response
        except Exception as e:
            logging.error(f"Failed to send message to MQL4: {str(e)}")
            return None

    def Negative_Cycle(self, dist, path):
        Neg_cycles = []
        unique_cycles = set()  # Track unique cycles to avoid duplicates
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

                # Selecting valid cycle with a restriction on length
                cycle_set = frozenset(neg_cycle)
                if neg_cycle[0] == neg_cycle[-1] and 3 < len(neg_cycle) <= 4 and cycle_set not in unique_cycles:
                    Neg_cycles.append(neg_cycle)
                    unique_cycles.add(cycle_set)
                    flag = True

        if (flag):
            return Neg_cycles
        else:
            return False

def main():
    app = App()
    app.Bellman_Ford()


if __name__ == "__main__":
    main()
