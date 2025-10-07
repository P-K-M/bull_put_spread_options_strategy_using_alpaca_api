import threading

from alpaca.data.live import CryptoDataStream
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from alpaca.data.historical import CryptoHistoricalDataClient

from alpaca.trading.requests import (
    GetOrdersRequest,
    GetAssetsRequest
)

from alpaca.trading.enums import (
    AssetClass,
    AssetStatus
)

from pydantic import ValidationError
from pytz import timezone
from ta.trend import macd
import pandas as pd
import numpy as np
import datetime

import logging

# Alpaca API credentials
api_key_id = 'PKON7XLHTRMPTZDJ6JB3'
api_secret = '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL'
base_url = 'https://paper-api.alpaca.markets/v2'

trading_client = TradingClient(api_key_id, api_secret, paper=True)

# Parameters to be optimised at your discretion
min_crypto_price = 0.0

max_crypto_price = 50.0

# Stop limit to default to
default_stop = .95

# How much of our portfolio to allocate to any one position
risk = 0.001


# Get Historical Data
def get_1000m_history_data(symbols):
    print('Getting historical Data...')
    # no keys required for crypto data
    client = CryptoHistoricalDataClient()

    minute_history = {}
    c = 0
    for symbol in symbols:
        request_params = CryptoBarsRequest(symbol_or_symbols=symbol,
                                           timeframe=TimeFrame.Minute, limit=1000
                                           )
        data = client.get_crypto_bars(request_params)

        minute_history[symbol] = data.df

        c += 1
        print('{}/{}'.format(c, len(symbols)))
    print('Success.')
    return minute_history


# Get Tickers
def get_tickers():
    print('Getting Current Ticker Data...')

    # get list of crypto pairs
    req = GetAssetsRequest(
        asset_class=AssetClass.CRYPTO,
        status=AssetStatus.ACTIVE
    )
    assets = trading_client.get_all_assets(req)

    symbols = [asset.symbol for asset in assets if asset.tradable]
    # no keys required for crypto data
    client = CryptoHistoricalDataClient()

    # Set the start time
    start_date = pd.to_datetime("2024-12-01").tz_localize('America/New_York')
    end_date = pd.to_datetime("2025-01-20").tz_localize('America/New_York')

    # Get data for the specified symbols
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_date, end=end_date
    )

    data = client.get_crypto_bars(request_params)

    data = data.df

    # Optimize the section of code below based on your preference

    # # Filter cryptocurrencies based on the defined price range
    # filtered_data = data[(data['close'] >= min_crypto_price) & (data['close'] <= max_crypto_price)]
    #
    # # Calculate 20-day moving average of closing prices
    # filtered_data['20_day_ma'] = filtered_data['close'].rolling(window=20).mean()

    # Calculate 20-day moving average of closing prices
    data['20_day_ma'] = data['close'].rolling(window=20).mean()

    # Narrow down the list based on open prices being higher than the 20-day moving average
    momentum_cryptos = data[data['open'] > data['20_day_ma']]

    print(momentum_cryptos)

    # Reset the index to move the 'symbol' from the index to a column
    momentum_cryptos.reset_index(inplace=True)

    # Extract the list of symbols
    tickers = momentum_cryptos['symbol'].unique().tolist()

    return tickers


# Find Stop
def find_stop(current_value, minute_history, now):
    # Reset the entire index, converting it to columns
    minute_history = minute_history.reset_index()

    # Set the 'timestamp' column as the new index
    if 'timestamp' in minute_history.columns:
        minute_history = minute_history.set_index('timestamp')
    else:
        print("Error: 'timestamp' column not found.")
        return current_value * default_stop  # Fallback if timestamp isn't found

    # Proceed with resampling
    series = minute_history['low'][-100:].dropna().resample('5min').min()

    series = series[now.floor('1D'):]

    diff = np.diff(series.values)
    low_index = np.where((diff[:-1] <= 0) & (diff[1:] > 0))[0] + 1

    if len(low_index) > 0:
        return series.iloc[low_index[-1]] - 0.01

    return current_value * default_stop


# Run
def run(tickers, market_open_dt, market_close_dt):
    # Establish streaming connection
    logging.basicConfig(level=logging.INFO)

    trading_stream = TradingStream('PKON7XLHTRMPTZDJ6JB3', '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL', paper=True)

    wss_client = CryptoDataStream('PKON7XLHTRMPTZDJ6JB3', '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')

    # get list of tickers to consider
    tickers = get_tickers()

    # Update initial state with information from tickers
    prev_closes = {}
    prev_lows = {}

    # Iterate through tickers
    for ticker in tickers:
        # no keys required for crypto data
        client = CryptoHistoricalDataClient()

        # Set the start and end time
        start_date = pd.to_datetime("2025-01-19").tz_localize('America/New_York')
        end_date = pd.to_datetime("2025-01-20").tz_localize('America/New_York')

        # Get data for the specified symbols
        request_params = CryptoBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Day, start=start_date, end=end_date
        )

        data = client.get_crypto_bars(request_params)

        data = data.df

        # Get the previous close price and low for the current ticker
        prev_close = data['close']
        prev_low = data['low']

        # Add the low and previous close price to the dictionaries
        prev_lows[ticker] = prev_low
        prev_closes[ticker] = prev_close

    symbols = tickers
    print('Tracking {} symbols.'.format(len(symbols)))
    minute_history = get_1000m_history_data(symbols)

    portfolio_value = float(trading_client.get_account().portfolio_value)

    # Print the portfolio value
    print("Portfolio Value:", portfolio_value)

    open_orders = {}

    # Cancel any existing open orders on watched symbols
    request_params = GetOrdersRequest(status=QueryOrderStatus.OPEN)

    orders = trading_client.get_orders(filter=request_params)

    # Loop through the existing orders and cancel those with symbols in your watchlist
    for order in orders:
        if order.symbol in symbols:
            trading_client.cancel_orders()

    positions = {}
    stop_prices = {}
    latest_cost_basis = {}

    # Keep track of what we're buying/selling
    target_prices = {}
    partial_fills = {}

    # Use trade updates to keep track of our portfolio
    def run_trading_stream():

        print("Streaming trade updates")

        async def handle_trade_updates(data):
            try:
                print("Received TradeUpdate:", data)
                symbol = data.order.symbol  # Use dot notation instead of square brackets
                last_order = open_orders.get(symbol)

                if last_order is not None:
                    event = data.event

                    if event == 'new':
                        # Handle the new order event
                        qty = float(data.order.qty) if data.order.qty is not None else 0
                        price = float(data.order.limit_price) if data.order.limit_price is not None else 0
                        position_qty = float(data.position_qty) if data.position_qty is not None else 0

                        print(f"New order event for {symbol}: qty={qty}, price={price}, position_qty={position_qty}")

                        # Update your positions, open_orders, etc.
                        positions[symbol] = position_qty
                        open_orders[symbol] = data.order

                    elif event == 'partial_fill':
                        # Handle the partial fill event
                        qty = float(data.position_qty)
                        if data.order.side == 'sell':
                            qty = qty
                        positions[symbol] = (
                                positions.get(symbol, 0) - partial_fills.get(symbol, 0)
                        )
                        partial_fills[symbol] = qty
                        positions[symbol] += qty
                        open_orders[symbol] = data.order

                    elif event == 'fill':
                        # Handle the fill event
                        qty = float(data.position_qty)
                        if data.order.side == 'sell':
                            qty = qty
                        positions[symbol] = (
                                positions.get(symbol, 0) - partial_fills.get(symbol, 0)
                        )
                        partial_fills[symbol] = 0
                        positions[symbol] += qty
                        open_orders[symbol] = None

                    elif event == 'canceled' or event == 'rejected':
                        # Handle the canceled or rejected event
                        partial_fills[symbol] = 0
                        open_orders[symbol] = None

            except ValidationError as e:
                # Handle the validation error
                print(f"Validation error in trade update: {e}")

        @trading_stream.subscribe_trade_updates
        async def print_trade_update(*symbols):
            print('trade update', *symbols)

        trading_stream.subscribe_trade_updates(handle_trade_updates)

        trading_stream.run()

    def run_wss_client():

        async def handle_second_bar(data):
            print(data)
            symbol = data.symbol
            if symbol not in minute_history:
                minute_history.update(get_1000m_history_data([symbol]))

            # Extract the minute-level timestamp
            ts = data.timestamp.replace(second=0, microsecond=0)

            try:
                current = minute_history[data.symbol].loc[ts]
            except KeyError:
                current = None
            new_data = []
            if current is None:
                new_data = [
                    data.open,
                    data.high,
                    data.low,
                    data.close,
                    data.volume,
                    data.trade_count,
                    data.vwap
                ]
            else:
                new_data = [
                    current.open,
                    max(data.high, current['high']),
                    min(data.low, current['low']),
                    data.close,
                    current.volume + data.volume,
                    data.trade_count,
                    data.vwap
                ]
            minute_history[symbol].loc[ts] = new_data

            # Next, check for existing orders for the stock
            existing_order = open_orders.get(symbol)
            if existing_order is not None:
                # Make sure the order's not too old
                submission_ts = existing_order.submitted_at.astimezone(
                    timezone('America/New_York')
                )
                order_lifetime = ts - submission_ts
                if order_lifetime.seconds // 60 > 3:
                    # Cancel it so we can try again for a fill
                    trading_client.cancel_order_by_id(existing_order.id)
                    print(f"Canceled order for {symbol}: {existing_order.id}")

            port_value = float(trading_client.get_account().portfolio_value)

            # Print the portfolio value
            print("Portfolio Value:", port_value)

            # Now we check to see if it might be time to buy or sell
            SinceMarketOpen = ts - market_open_dt
            until_market_close = market_close_dt - ts

            if (
                    SinceMarketOpen.seconds // 60 > 15
            ):
                print("Checking for buy signals")
                position = positions.get(symbol, 0)
                if position > 0:
                    print(f"Skipping MACD check for {symbol} because the position is greater than 0.")

                else:
                    print(f"Checking positive MACD check for {symbol}")
                    # Get the change since yesterday's market close
                    daily_pct_change = (
                            (data.close - prev_closes[symbol]) / prev_closes[symbol]
                    )
                    # Print out the result
                    print(f"The daily percent change for {symbol} is: {daily_pct_change}")

                    if (daily_pct_change > 0).any():
                        # Print a message indicating the conditions are met
                        print("Conditions met! The following criteria have been satisfied:")
                        print(f"- Daily percent change > 1%: {daily_pct_change}")

                        # Check for a positive, increasing MACD
                        print("Checking for positive MACD (12-26 period)")
                        Histogram = macd(
                            minute_history[symbol]['close'].dropna(),
                            window_fast=12,
                            window_slow=26
                        )
                        print("Histogram (12-26):", Histogram)
                        if len(Histogram) > 0:
                            if Histogram.iloc[-1] < 0 or not (
                                    Histogram.iloc[-3] < Histogram.iloc[-2] < Histogram.iloc[-1]
                            ):
                                print("MACD (12-26) not positive or increasing.")
                            else:
                                # If MACD (12-26) is positive and increasing, proceed to check the MACD for 40-60 period.
                                print("MACD (12-26) is positive and increasing. Checking MACD (40-60 period).")

                                Histogram = macd(
                                    minute_history[symbol]['close'].dropna(),
                                    window_fast=40,
                                    window_slow=60
                                )
                                print("Histogram (40-60):", Histogram)
                                if len(Histogram) > 0:
                                    if Histogram.iloc[-1] < 0 or (
                                            len(Histogram) > 1 and np.diff(Histogram)[-1] < 0
                                    ):
                                        print("MACD (40-60) not positive or increasing.")
                                    else:
                                        print("Both MACD (12-26) and MACD (40-60) are positive and increasing.")

                                        # Stock has passed all checks; figure out how much to buy
                                        stop_price = find_stop(
                                            data.close, minute_history[symbol], ts
                                        )
                                        stop_prices[symbol] = stop_price

                                        # Use the symbol mapping for consistent formatting
                                        target_price = data.close + ((data.close - stop_price) * 2)

                                        target_prices[symbol] = target_price

                                        shares_to_buy = portfolio_value * risk // (
                                                data.close - stop_price
                                        )
                                        if shares_to_buy == 0:
                                            shares_to_buy = 1
                                        shares_to_buy -= positions.get(symbol, 0)

                                        # Check if shares_to_buy is positive and not rounded down to zero
                                        if shares_to_buy <= 0:
                                            return

                                        limit_price = data.close

                                        # Check if limit_price is within the acceptable range for Alpaca API
                                        if limit_price < 0 or (limit_price >= 1.0 and limit_price < 1.0):
                                            return

                                        print('Submitting buy for {} shares of {} at {}'.format(
                                            shares_to_buy, symbol, limit_price
                                        ))
                                        try:
                                            limit_order_data = LimitOrderRequest(
                                                symbol=symbol,
                                                limit_price=limit_price,
                                                qty=str(shares_to_buy),
                                                side=OrderSide.BUY,
                                                time_in_force=TimeInForce.GTC
                                            )

                                            o = trading_client.submit_order(
                                                order_data=limit_order_data
                                            )
                                            open_orders[symbol] = o
                                            print(f"open_orders['{symbol}'] = {o}")

                                            latest_cost_basis[symbol] = data.close
                                            print(f"latest_cost_basis['{symbol}'] = {data.close}")

                                        except Exception as e:
                                            print(e)

            if (
                    SinceMarketOpen.seconds // 60 >= 24 and
                    until_market_close.seconds // 60 > 15
            ):
                print("Checking for sell signals")
                # Check for Sell signals
                # We can't liquidate if there's no position

                position = positions.get(symbol, 0)
                if position == 0:
                    print(f"Skipping MACD check for {symbol} because the position is equal to 0.")

                else:
                    print(f"Checking negative MACD check for {symbol} with position quantity: {position}")
                    # Sell for a loss if it's fallen below our stop price
                    # Sell for a loss if it's below our cost basis and MACD < 0
                    # Sell for a profit if it's above our target price
                    Histogram = macd(
                        minute_history[symbol]['close'].dropna(),
                        window_fast=13,
                        window_slow=21
                    )
                    print("Histogram:", Histogram)

                    # Check if it's time to sell the symbol
                    if data.close <= stop_prices[symbol]:
                        print("price is less than or equal to the stop price")
                        print("Original Stop Price:", stop_prices[symbol])
                        print("Current Price:", data.close)
                    else:
                        print("price is not less than or equal to the stop price")
                        print("Original Stop Price:", stop_prices[symbol])
                        print("Current Price:", data.close)

                    if data.close >= target_prices[symbol] and Histogram.iloc[-1] <= 0:
                        print("price is greater than or equal to the target price")
                        print("Original Target Price:", target_prices[symbol])
                        print("Current Price:", data.close)
                    else:
                        print("price is not greater than or equal to the target price")
                        print("Original Target Price:", target_prices[symbol])
                        print("Current Price:", data.close)

                    if data.close <= latest_cost_basis[symbol] and Histogram.iloc[-1] <= 0:
                        print("price is less than or equal to the latest_cost_basis")
                        print("Original latest_cost_basis:", latest_cost_basis[symbol])
                        print("Current Price:", data.close)
                    else:
                        print("price is not less than or equal to the latest_cost_basis")
                        print("Original latest_cost_basis:", latest_cost_basis[symbol])
                        print("Current Price:", data.close)

                    if len(Histogram) > 0:
                        if (
                                data.close <= stop_prices[symbol] or
                                (data.close >= target_prices[symbol] and Histogram.iloc[-1] <= 0) or
                                (data.close <= latest_cost_basis[symbol] and Histogram.iloc[-1] <= 0)
                        ):

                            # With this code to round the limit price based on the specified constraints
                            if data.close >= 1.0:
                                data.close = round(data.close, 2)
                                print(f"Rounded Close Value: {data.close}")
                            elif data.close <= 1.0:
                                data.close = round(data.close, 4)
                                print(f"Rounded Close Value: {data.close}")

                            print('Submitting sell for {} shares of {} at {}'.format(
                                position, symbol, data.close
                            ))
                            try:

                                # Use the minimum notional value as the notional in the sell order
                                limit_order_data = LimitOrderRequest(
                                    symbol=symbol,
                                    limit_price=str(data.close),
                                    qty=str(position),
                                    side=OrderSide.SELL,
                                    time_in_force=TimeInForce.GTC
                                )

                                o = trading_client.submit_order(
                                    order_data=limit_order_data
                                )
                                open_orders[symbol] = o
                                print(f"open_orders['{symbol}'] = {o}")

                                latest_cost_basis[symbol] = data.close
                                print(f"latest_cost_basis['{symbol}'] = {data.close}")

                            except Exception as e:
                                print(e)
                        else:
                            print("No conditions met")
            elif (
                    until_market_close.seconds // 60 <= 15
            ):
                print("Checking for liquidation position")
                # Liquidate remaining positions on watched symbols at market
                try:
                    trading_client.get_all_positions()
                except Exception as e:
                    # Exception here indicates that we have no position
                    return
                print('Trading over, liquidating remaining position in {}'.format(
                    symbol)
                )
                # Map to convert position.symbol to match the format in symbols
                symbol_mapping = {
                    "AAVEUSD": "AAVE/USD",
                    "AVAXUSD": "AVAX/USD",
                    "BATUSD": "BAT/USD",
                    "CRVUSD": "CRV/USD",
                    "DOTUSD": "DOT/USD",
                    "GRTUSD": "GRT/USD",
                    "LINKUSD": "LINK/USD",
                    "SHIBUSD": "SHIB/USD",
                    "LTCUSD": "LTC/USD",
                    "SUSHIUSD": "SUSHI/USD",
                    "UNIUSD": "UNI/USD",
                    "USDCUSD": "USDC/USD",
                    "XTZUSD": "XTZ/USD",
                    "DOGEUSD": "DOGE/USD"
                }

                # Track any positions bought during previous executions
                existing_positions = trading_client.get_all_positions()
                for position in existing_positions:
                    print(f"Symbol: {position.symbol}")
                    print(f"Quantity: {position.qty}")
                    print(f"Market Value: {position.unrealized_intraday_pl}")
                    print("------")

                    if position.symbol in symbol_mapping and float(position.unrealized_intraday_pl) > 0:
                        formatted_symbol = symbol_mapping[position.symbol]
                        # symbol = formatted_symbol
                        positions[formatted_symbol] = float(position.unrealized_intraday_pl)

                        # Recalculate cost basis and stop price
                        symbol = formatted_symbol

                        # Close the position only if market value is greater than 0
                        trading_client.close_position(position.symbol)

                        # Print the liquidated symbol
                        print(f"Liquidated Symbol: {symbol}")

                        trading_stream.stop()

                        print('Finished liquidating all positions.')

        wss_client.subscribe_bars(handle_second_bar, *symbols)

        # start our websocket streaming
        wss_client.run()

    # Create threads for trading_stream and wss_client
    trading_stream_thread = threading.Thread(target=run_trading_stream)
    wss_client_thread = threading.Thread(target=run_wss_client)

    # Start both threads
    trading_stream_thread.start()
    wss_client_thread.start()

    # Wait for both threads to finish
    trading_stream_thread.join()
    wss_client_thread.join()


if __name__ == "__main__":
    # Get when the market opens or opened today
    nyc = timezone('America/New_York')
    today = datetime.datetime.today().astimezone(nyc)
    today_str = datetime.datetime.today().astimezone(nyc).strftime('%Y-%m-%d')

    market_open = today.replace(hour=9, minute=00, second=0, microsecond=0)

    market_open = market_open.astimezone(nyc)
    market_close = today.replace(hour=21, minute=30, second=0, microsecond=0)
    market_close = market_close.astimezone(nyc)

    # Wait until just before we might want to trade
    current_dt = datetime.datetime.today().astimezone(nyc)
    since_market_open = current_dt - market_open

    run(get_tickers(), market_open, market_close)