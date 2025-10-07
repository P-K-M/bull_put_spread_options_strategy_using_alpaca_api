from api_interface import Main, PriceInformation

from datetime import datetime

from pytz import timezone

from threading import Timer
from ta.trend import macd
import pandas as pd
import numpy as np
import datetime, time

from datetime import timedelta


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

        self.start_time = self.today.replace(hour=1, minute=0, second=0, microsecond=0)
        self.start_time = self.start_time.astimezone(self.nyc)

        self.stop_time = self.today.replace(hour=16, minute=30, second=0, microsecond=0)
        self.stop_time = self.today.replace(hour=16, minute=25, second=0, microsecond=0)
        # self.stop_time = self.stop_time.astimezone(self.nyc)

        self.pnl_time = self.stop_time.astimezone(self.nyc)

        self.lots_to_buy = 0

        self.default_stop = 0.99950

        self.risk = 0.000002

        # self.currency_pairs = [
        #     self.EUR_USD_FX(),
        #     self.AUD_USD_FX(),
        #     self.GBP_USD_FX(),
        #     self.NZD_USD_FX(),
        #     self.USD_CAD_FX(),
        #     self.USD_JPY_FX(),
        #     self.USD_CNH_FX(),
        #     self.USD_CHF_FX()
        # ]
        self.currency_pairs = [
            self.EUR_USD_FX(),
            self.GBP_USD_FX()
        ]
        # self.currency_pairs = [
        #
        #     self.EUR_USD_FX(),
        #     self.AUD_USD_FX(),
        #     self.GBP_USD_FX(),
        #     self.NZD_USD_FX()
        # ]

        self.dataframe = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close'])

        self.target_prices = {}

        self.stop_prices = {}

        self.historical_data = {}

        self.market_data_dict = {}

        self.latest_cost_basis = {}

        self.stop_prices_after_buy = {}

        self.target_prices_after_buy = {}

        self.stop_target_prices = {}

        self.open_orders = []

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

        for i, contract in enumerate(self.currency_pairs):
            req_id = self.get_unique_id() + i

            self.market_data[req_id] = PriceInformation(contract)

            self.reqMarketDataType(1)
            self.reqMktData(req_id, contract, "", False, False, [])

        self.loop()

    def loop(self):

        for contract in self.currency_pairs:

            if contract in self.stop_target_prices:
                del self.stop_target_prices[contract]

            # Cancel any existing open orders on watched symbols
            self.open_orders = self.reqOpenOrders()
            if self.open_orders:
                print("Existing Orders:")
                [print(order) for order in self.open_orders]
            else:
                print("No existing orders.")
            if self.open_orders is not None:
                for order, contract in self.open_orders:
                    if contract.symbol in [contract.symbol for contract in self.currency_pairs]:
                        # Cancel the order
                        print("Canceling order:", order.orderId)
                        self.cancel_order(order.orderId)

                if len(self.open_orders) == 0:
                    print("No orders to cancel.")
                else:
                    print("No existing orders to cancel.")

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

                        # Print the stop price, target price, and price of each contract
                        print("Contract:", contract)
                        print("Stop Price:", self.stop_prices[contract])
                        print("Target Price:", self.target_prices[contract])
                        print("Current Price:", price)
                        print()

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

                        # Now we check to see if it might be time to buy
                        if self.current_dt < (self.stop_time - timedelta(minutes=15)):

                            self.get_positions()
                            existing_positions = self.positions

                            # See if we've already bought in first
                            contract_identifier = (contract.symbol, contract.currency)
                            if contract_identifier in [(pos.contract.symbol, pos.contract.currency) for pos in
                                                       existing_positions]:
                                position = next((pos for pos in existing_positions if
                                                 (pos.contract.symbol,
                                                  pos.contract.currency) == contract_identifier),
                                                None)
                                if position and position.pos > 0:
                                    print(
                                        "Skipping contract {} with currency {} because it has an existing position "
                                        "greater than 0.".format(
                                            contract.symbol, contract.currency))
                                else:

                                    print("Calculating MACD for contract:", contract.symbol, contract.currency)

                                    # Check for buy signals
                                    print("Checking for buy signals")

                                    # check for a positive, increasing MACD
                                    print("checking for positive MACD")
                                    Histogram = macd(
                                        self.historical_data[contract]['Close'].dropna(),
                                        window_fast=12,
                                        window_slow=26
                                    )
                                    print("Histogram:", Histogram)

                                    if len(Histogram) > 0:
                                        if Histogram.iloc[-1] < 0 or not (
                                                Histogram.iloc[-3] < Histogram.iloc[-2] < Histogram.iloc[-1]):
                                            print("MACD not positive or increasing, continuing")
                                        else:
                                            Histogram = macd(
                                                self.historical_data[contract]['Close'].dropna(),
                                                window_fast=40,
                                                window_slow=60
                                            )
                                            print("Histogram:", Histogram)
                                            if len(Histogram) > 0:
                                                if Histogram.iloc[-1] < 0 or (
                                                        len(Histogram) > 1 and np.diff(Histogram)[-1] < 0):
                                                    print("MACD not positive or increasing, continuing")
                                                else:
                                                    # buy

                                                    print("Buying contract:", contract.symbol)

                                                    self.trade(contract, "BUY", price)

                                                    # Store stop and target prices for this contract after buying
                                                    print("Storing stop and target prices after buying for contract:",
                                                          contract.symbol, contract.currency)
                                                    print("Original Stop Price:", self.stop_prices[contract])
                                                    print("Original Target Price:", self.target_prices[contract])

                                                    self.stop_target_prices[contract] = {
                                                        'stop_price': stop_price,
                                                        'target_price': target_price
                                                    }

                                                    print("Stop and Target Prices:", self.stop_target_prices)

                            else:
                                print(
                                    "Contract {} with currency {} is not in existing positions. "
                                    "Calculating MACD.".format(
                                        contract.symbol, contract.currency))
                                # Calculate MACD and check for buy signals
                                print("Calculating MACD for contract:", contract.symbol, contract.currency)

                                # Check for buy signals
                                print("Checking for buy signals")

                                # check for a positive, increasing MACD
                                print("checking for positive MACD")
                                Histogram = macd(
                                    self.historical_data[contract]['Close'].dropna(),
                                    window_fast=12,
                                    window_slow=26
                                )
                                print("Histogram:", Histogram)

                                if len(Histogram) > 0:
                                    if Histogram.iloc[-1] < 0 or not (
                                            Histogram.iloc[-3] < Histogram.iloc[-2] < Histogram.iloc[-1]):
                                        print("MACD not positive or increasing, continuing")
                                    else:
                                        Histogram = macd(
                                            self.historical_data[contract]['Close'].dropna(),
                                            window_fast=40,
                                            window_slow=60
                                        )
                                        print("Histogram:", Histogram)

                                        if len(Histogram) > 0:
                                            if Histogram.iloc[-1] < 0 or (
                                                    len(Histogram) > 1 and np.diff(Histogram)[-1] < 0):
                                                print("MACD not positive or increasing, continuing")
                                            else:
                                                # buy

                                                print("Buying contract:", contract.symbol)

                                                self.trade(contract, "BUY", price)

                                                # Store stop and target prices for this contract after buying
                                                print("Storing stop and target prices after buying for contract:",
                                                      contract.symbol, contract.currency)
                                                print("Original Stop Price:", self.stop_prices[contract])
                                                print("Original Target Price:", self.target_prices[contract])

                                                self.stop_target_prices[contract] = {
                                                    'stop_price': stop_price,
                                                    'target_price': target_price
                                                }

                                                print("Stop and Target Prices:", self.stop_target_prices)

                        # Check for sell signals
                        if self.current_dt < (self.stop_time - timedelta(minutes=5)):

                            contract_identifier = (contract.symbol, contract.currency)
                            if contract_identifier in [(pos.contract.symbol, pos.contract.currency) for pos in
                                                       existing_positions]:
                                position = next((pos for pos in existing_positions if
                                                 (pos.contract.symbol, pos.contract.currency) == contract_identifier),
                                                None)
                                if position and position.pos <= 0:
                                    print(
                                        "Skipping contract {} with currency {} because it has no existing position "
                                        "greater than 0.".format(
                                            contract.symbol, contract.currency))
                                else:
                                    print("Calculating MACD for contract:", contract.symbol, contract.currency)

                                    # Sell for a loss if it's fallen below our stop price
                                    # Sell for a loss if it's below our cost basis and MACD < 0
                                    # Sell for a profit if it's above our target price

                                    # check for a negative, decreasing MACD
                                    print("checking for negative MACD")

                                    Histogram = macd(
                                        self.historical_data[contract]['Close'].dropna(),
                                        window_fast=13,
                                        window_slow=21
                                    )
                                    print("Histogram:", Histogram)

                                    print("Before accessing dictionary - Contract ID:", id(contract))

                                    if price <= self.stop_target_prices[contract]['stop_price']:
                                        print("price is less than or equal to the stop price")
                                        print("Stop Price:", self.stop_target_prices[contract]['stop_price'])
                                        print("Original Stop Price:", self.stop_prices[contract])
                                        print("Current Price:", price)
                                    else:
                                        print("price is not less than or equal to the stop price")
                                        print("Stop Price:", self.stop_target_prices[contract]['stop_price'])
                                        print("Original Stop Price:", self.stop_prices[contract])
                                        print("Current Price:", price)

                                    if price >= self.stop_target_prices[contract]['target_price']:
                                        print("price is greater than or equal to the target price")
                                        print("Target Price:", self.stop_target_prices[contract]['target_price'])
                                        print("Original Target Price:", self.target_prices[contract])
                                        print("Current Price:", price)
                                    else:
                                        print("price is not greater than or equal to the target price")
                                        print("Target Price:", self.stop_target_prices[contract]['target_price'])
                                        print("Original Target Price:", self.target_prices[contract])
                                        print("Current Price:", price)

                                    if price >= self.stop_target_prices[contract]['target_price']:

                                        print("Selling due to target price being reached for contract:",
                                              contract.symbol, contract.currency)

                                        # sell using the determined price
                                        self.trade(contract, "SELL", price)

                                        if position and position.pos > 0:
                                            # Contract has a position greater than 0, don't remove it
                                            print("Contract has a position greater than 0")
                                        else:
                                            # Contract has no existing position greater than 0, remove it
                                            # from stop_target_prices
                                            if contract in self.stop_target_prices:
                                                del self.stop_target_prices[contract]

                                    elif price <= self.stop_target_prices[contract]['stop_price']:
                                        print("Selling due to stop price being reached for contract:",
                                              contract.symbol, contract.currency)

                                        self.trade(contract, "SELL", price)

                                        if position and position.pos > 0:
                                            # Contract has a position greater than 0, don't remove it
                                            print("Contract has a position greater than 0")
                                        else:
                                            # Contract has no existing position greater than 0, remove it from
                                            # stop_target_prices
                                            if contract in self.stop_target_prices:
                                                del self.stop_target_prices[contract]

                                    elif len(Histogram) > 1 and Histogram.iloc[-1] < 0 and Histogram.iloc[-1] < \
                                            Histogram.iloc[-2]:
                                        print("Selling due to negative and decreasing MACD for contract:",
                                              contract.symbol, contract.currency)

                                        self.trade(contract, "SELL", price)

                                        if position and position.pos > 0:
                                            # Contract has a position greater than 0, don't remove it
                                            print("Contract has a position greater than 0")
                                        else:
                                            # Contract has no existing position greater than 0, remove it from
                                            # stop_target_prices
                                            if contract in self.stop_target_prices:
                                                del self.stop_target_prices[contract]

                                    else:
                                        print(
                                            f"No selling signal for {contract} at price {price}")
                            else:
                                print(
                                    "Skipping contract {} with currency {} because it's not in existing positions."
                                    .format(contract.symbol, contract.currency))

                        else:

                            if self.current_dt >= (self.stop_time - timedelta(minutes=5)):

                                # Check for Liquidation signals

                                contract_identifier = (contract.symbol, contract.currency)
                                if contract_identifier in [(pos.contract.symbol, pos.contract.currency) for pos in
                                                           existing_positions]:
                                    position = next((pos for pos in existing_positions if
                                                     (pos.contract.symbol,
                                                      pos.contract.currency) == contract_identifier),
                                                    None)
                                    if position:
                                        if position.pos > 0:
                                            # Sell the position
                                            quantity = position.pos
                                            self.mkt_trade(contract, "SELL",
                                                           price)  # Assuming SELL is the correct action
                                            print(f"Sold {quantity} of {contract.symbol} at market price.")
                                        elif position.pos < 0:
                                            # Buy to cover the short position
                                            quantity = -position.pos
                                            self.mkt_trade(contract, "BUY", price)
                                            print(f"Bought {quantity} of {contract.symbol} at market price.")
                                        else:
                                            print(
                                                "Skipping contract {} with currency {} because it has no "
                                                "existing position.".format(
                                                    contract.symbol, contract.currency))
                                    else:
                                        print("No existing position found for contract {} with currency {}.".format(
                                            contract.symbol, contract.currency))

    def trade(self, contract, action, price):

        if contract in ['EUR/USD', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'USD/CNH', 'NZD/USD', 'USD/CHF']:
            rounded_price = round(price, 5)
        elif contract == 'USD/JPY':
            rounded_price = round(price, 3)
        else:
            rounded_price = round(price, 5)

        print(
            "=========================================================== trading {} with action: {} price: {}"
            .format(contract, action, rounded_price))

        self.quantity = self.lots_to_buy

        # self.quantity = 50

        order = self.limit_order(action, self.quantity,
                                 self.min_price_increment(rounded_price, contract))
        order.conditionsCancelOrder = True
        order.conditions.append(self.TimeCondition(True, False, delta=datetime.timedelta(seconds=30)))
        self.placeOrder(self.get_order_id(), contract, order)

    def mkt_trade(self, contract, action, price):

        if contract in ['EUR/USD', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'USD/CNH']:
            rounded_price = round(price, 5)
        elif contract == 'USD/JPY':
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
        if contract in ['EUR/USD', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'USD/CNH']:
            return round(round(price / 0.00005) * 0.00005, 5)
        elif contract == 'USD/JPY':
            return round(round(price / 0.005) * 0.005, 3)
        else:
            return price  # or raise an exception indicating an unsupported currency pair


app = App("127.0.0.1", 7497, 1)
try:
    app.main()
except KeyboardInterrupt:
    app.disconnect()