import math
import numpy as np
import logging
import zmq
import mysql.connector
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    filename='mql4_python_connection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

num = 6
run_duration = timedelta(minutes=30)  # Set duration for 1 hour

class App:
    def __init__(self):
        labels = ['USD', 'EUR', 'AUD', 'GBP', 'CAD', 'CHF']
        self.vertices = num
        self.arches = []
        self.tickers = labels
        self.currency_matrix = np.zeros((num, num))
        self.last_timestamp = None  # Track the last timestamp used in a query

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

        # Initialize or reset last_timestamp before each Bellman-Ford run

    def reset_timestamp(self):
        # Reset to a very old date or None so it fetches the latest data each time
        self.last_timestamp = None

    def get_midpoint(self, symbol):
        # Refresh the database connection each time to ensure fresh data retrieval
        self.db_connection.reconnect(attempts=3, delay=1)

        cursor = self.db_connection.cursor()

        # If we have a last_timestamp, try to get data newer than it
        if self.last_timestamp:
            query = """
                SELECT midpoint, timestamp FROM currency_prices 
                WHERE symbol = %s AND timestamp > %s 
                ORDER BY timestamp DESC LIMIT 1
            """
            cursor.execute(query, (symbol, self.last_timestamp))
            result = cursor.fetchone()

            # If no new data is found, get the latest available data
            if not result:
                query = """
                    SELECT midpoint, timestamp FROM currency_prices 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC LIMIT 1
                """
                cursor.execute(query, (symbol,))
                result = cursor.fetchone()
        else:
            # Initial case: no last_timestamp set, so get the latest data
            query = """
                SELECT midpoint, timestamp FROM currency_prices 
                WHERE symbol = %s 
                ORDER BY timestamp DESC LIMIT 1
            """
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()

        cursor.close()

        if result:
            self.last_timestamp = result[1]  # Update last timestamp to the latest found
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
                self.define_arches(i, j, round(-math.log(data), 5))

    def Bellman_Ford(self):
        # 1° Creating graph
        logging.info("Starting Bellman-Ford algorithm...")
        print('\nCollecting data, computing Bellman-Ford algorithm, searching for arbitrage opportunity...')
        self.build_graph()

        dist = [float("Inf")] * self.vertices
        path = [float("Inf")] * self.vertices
        dist[0] = 0
        path[0] = 0

        # Relax edges |V| - 1 times
        for _ in range(self.vertices - 1):
            for s, e, v in self.arches:
                if dist[s] != float("Inf") and dist[s] + v < dist[e]:
                    dist[e] = dist[s] + v
                    path[e] = s

        # Detect negative cycles
        Neg_cycles = self.Negative_Cycle(dist, path)

        # Results, if there is a negative cycle --> computing possible profit
        if not Neg_cycles:
            logging.info("No arbitrage opportunity detected.")
            print("\nNo arbitrage opportunity.")
            # Notify MQL4 to continue streaming data
            self.send_to_mql4("No arbitrage opportunity detected. Continue streaming data.")
        else:
            profitable_cycles = []
            for neg_cycle in Neg_cycles:
                profit = 1
                prec = neg_cycle[-1]
                for i in neg_cycle[-2::-1]:
                    profit *= self.currency_matrix[prec][i]
                    prec = i
                profit = round(profit, 4)

                # Print all negative cycles with their profit
                cycle_str = ' --> '.join([self.tickers[i] for i in neg_cycle[::-1]])
                print(f"\nCycle detected: {cycle_str}\nProfit: {profit}")
                logging.info(f"Cycle detected: {cycle_str}; Profit: {profit}")

                # Collect profitable cycles
                if profit >= 1.0001:
                    profitable_cycles.append((cycle_str, profit))

            # Send the most profitable cycle to MQL4, if any
            if profitable_cycles:
                # Find the cycle with the highest profit
                most_profitable_cycle = max(profitable_cycles, key=lambda x: x[1])
                cycle_str, profit = most_profitable_cycle
                message = f"Cycle: {cycle_str}; Profit: {profit}"
                response = self.send_to_mql4(message)
                logging.info(f"Sent to MQL4: {message}; Response: {response}")

            else:
                logging.info("No profitable arbitrage opportunity detected.")
                print("\nNo profitable arbitrage opportunity detected.")
                self.send_to_mql4("No profitable arbitrage opportunity detected. Continue streaming data.")

    def run_for_duration(self, duration):
        start_time = datetime.now()
        end_time = start_time + duration
        while datetime.now() < end_time:
            self.Bellman_Ford()
            # Optional pause between checks to avoid excessive polling
            time.sleep(2)
        # Notify MQL4 to stop streaming and any other operations
        stop_message = "Trading duration elapsed. Stop all operations."
        response = self.send_to_mql4(stop_message)
        logging.info(f"Sent to MQL4: {stop_message}; Response: {response}")
        print("Trading duration elapsed. Stopping operations.")

    def send_to_mql4(self, message):
        try:
            self.socket.send_string(message)
            response = self.socket.recv_string()
            while "All trades executed" not in response:
                time.sleep(1)
                response = self.socket.recv_string()
            logging.info("Cycle executed by MQL4; searching for next cycle.")
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
    app.run_for_duration(run_duration)

if __name__ == "__main__":
    main()
