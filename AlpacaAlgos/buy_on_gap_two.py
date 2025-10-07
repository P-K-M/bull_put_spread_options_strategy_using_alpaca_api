import threading
import time

from alpaca.data.historical.stock import StockHistoricalDataClient

from alpaca.data.live.stock import StockDataStream

from alpaca.data.timeframe import *
from alpaca.trading.client import *
from alpaca.trading.stream import *
from alpaca.trading.requests import *
from alpaca.data.requests import (
    StockBarsRequest
)
from alpaca.trading.requests import (
    GetAssetsRequest,
    LimitOrderRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import (
    AssetExchange,
    OrderSide,
    TimeInForce,
    QueryOrderStatus
)

from pydantic import ValidationError
from pytz import timezone
import pandas as pd
import numpy as np
import datetime

import logging

# Alpaca API credentials
api_key_id = 'PKON7XLHTRMPTZDJ6JB3'
api_secret = '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL'
base_url = 'https://paper-api.alpaca.markets/v2'

trading_client = TradingClient(api_key_id, api_secret, paper=True)

# We only consider stocks with per-share prices inside this range
min_share_price = 2.0
max_share_price = 30.0
# Minimum previous-day dollar volume for a stock we might consider
min_last_dv = 500000

# Stop limit to default to
default_stop = .95

# How much of our portfolio to allocate to any one position
risk = 0.001


# Get Historical Data
def get_1000m_history_data(symbols):
    print('Getting historical Data...')

    # setup stock historical data client
    stock_historical_data_client = StockHistoricalDataClient('PKON7XLHTRMPTZDJ6JB3',
                                                             '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')

    minute_history = {}
    c = 0
    for symbol in symbols:
        request_params = StockBarsRequest(symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute, limit=1000
        )
        data = stock_historical_data_client.get_stock_bars(request_params)

        minute_history[symbol] = data.df

        c += 1
        print('{}/{}'.format(c, len(symbols)))
    print('Success.')
    return minute_history

# Get Tickers
def get_tickers():
    print('Getting Current Ticker Data...')

    # get list of assets which are us_equity (default), active, and in NASDAQ
    # ref. https://docs.alpaca.markets/reference/get-v2-assets-1
    req = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,  # default asset_class is us_equity
        # status=AssetStatus.ACTIVE,
        exchange=AssetExchange.ARCA
    )
    assets = trading_client.get_all_assets(req)

    symbols = [asset.symbol for asset in assets if asset.tradable]

    # setup stock historical data client
    stock_historical_data_client = StockHistoricalDataClient('PKON7XLHTRMPTZDJ6JB3',
                                                             '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')

    # Set the start and end time
    start_date = pd.to_datetime("2024-07-01").tz_localize('America/New_York')
    end_date = pd.to_datetime("2024-08-19").tz_localize('America/New_York')

    # Get data for the specified symbols
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_date, end=end_date
    )

    data = stock_historical_data_client.get_stock_bars(request_params)

    data = data.df

    # Calculate 20-day moving average of closing prices
    data['20_day_ma'] = data['close'].rolling(window=20).mean()

    # Narrow down the list based on open prices being higher than the 20-day moving average
    momentum_stocks = data[data['open'] > data['20_day_ma']]

    # Reset the index to move the 'symbol' from the index to a column
    momentum_stocks.reset_index(inplace=True)

    # Extract the list of symbols
    tickers = momentum_stocks['symbol'].unique().tolist()

    return tickers

# Find Stop
def find_stop(current_value, minute_history, now):
    # Step 1: Reset the entire index, converting it to columns
    minute_history = minute_history.reset_index()

    # Step 2: Set the 'timestamp' column as the new index
    if 'timestamp' in minute_history.columns:
        minute_history = minute_history.set_index('timestamp')
    else:
        print("Error: 'timestamp' column not found.")
        return current_value * default_stop  # Fallback if timestamp isn't found

    # Confirm the new index type
    print("Modified Index:", minute_history.index)

    # Step 3: Proceed with resampling
    series = minute_history['low'][-100:].dropna().resample('5min').min()
    print("Series after resampling:", series)

    series = series[now.floor('1D'):]
    print("Series after filtering:", series)

    diff = np.diff(series.values)
    low_index = np.where((diff[:-1] <= 0) & (diff[1:] > 0))[0] + 1

    if len(low_index) > 0:
        return series.iloc[low_index[-1]] - 0.01

    return current_value * default_stop

# Run
def run(tickers, market_open_dt, market_close_dt):

    # Establish streaming connection
    logging.basicConfig(level=logging.INFO)

    trading_stream = TradingStream('PKON7XLHTRMPTZDJ6JB3',
                                   '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL',
                                   paper=True)

    wss_client = StockDataStream('PKON7XLHTRMPTZDJ6JB3', '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')

    # get list of tickers to consider
    tickers = get_tickers()

    # get list of stocks that gapped down
    gap_down_stocks = []

    for ticker in tickers:
        # setup stock historical data client
        stock_historical_data_client = StockHistoricalDataClient('PKON7XLHTRMPTZDJ6JB3',
                                                                 '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')

        # Get data for the specified symbols
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Day, start=datetime.datetime(2024, 4, 1),
            end=datetime.datetime(2024, 8, 15), limit = 100
        )

        data = stock_historical_data_client.get_stock_bars(request_params)
        data = data.df

        # Calculate daily close-to-close returns for the last 90 days and compute standard deviation
        data['daily_returns'] = data['close'].pct_change()
        rolling_std_90 = data['daily_returns'].rolling(window=90).std()

        prev_day_low = data['low'].iloc[-2]
        gap_down_stocks.append([ticker, prev_day_low, rolling_std_90])

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
    stop_target_prices = {}

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

                        # Update your logic based on the new order event
                        # For example, you can update positions, open_orders, etc.

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

                    # Add other cases for 'fill', 'canceled', 'rejected', etc.

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
                SinceMarketOpen.seconds // 60 > 15 and
                SinceMarketOpen.seconds // 60 < 60
            ):
                print("Checking for buy signals")
                position = positions.get(symbol, 0)
                if position > 0:
                    print(f"Skipping check for {symbol} because the position is greater than 0.")
                else:
                    print(f"Checking for {symbol}")

                    todays_open = data.open
                    for stock in gap_down_stocks:
                        ticker, prev_day_low, rolling_std_90 = stock[:3]
                        if ticker == symbol:
                            return_from_low_to_open = (todays_open - prev_day_low) / prev_day_low
                            if return_from_low_to_open < -rolling_std_90:
                                stock.append(return_from_low_to_open)

                    # After open prices are set, filter the stocks
                    filtered_stocks = [s for s in gap_down_stocks if len(s) == 4]
                    filtered_stocks = sorted(filtered_stocks, key=lambda x: x[3])
                    stocks_to_buy = [ticker for ticker, _, _, _ in filtered_stocks[:10]]

                    # Place buy orders for selected stocks
                    for symbol in stocks_to_buy:
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
                    until_market_close.seconds // 60 <= 5
            ):
                # Liquidate all positions
                print("Checking for liquidation position")
                try:
                    existing_positions = trading_client.get_all_positions()
                    if not existing_positions:
                        print("No positions to liquidate.")
                        return
                except Exception as e:
                    # Exception here indicates that we have no position
                    print(f"Error fetching positions: {e}")
                    return

                for position in existing_positions:
                    print(f"Symbol: {position.symbol}")
                    print(f"Quantity: {position.qty}")
                    print(f"Market Value: {position.unrealized_intraday_pl}")
                    print("------")

                # Close all positions and cancel open orders
                try:
                    closed_positions = trading_client.close_all_positions(cancel_orders=True)
                    for closed_position in closed_positions:
                        print(f"Liquidated Symbol: {closed_position.symbol}")
                except Exception as e:
                    print(f"Error liquidating positions: {e}")

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

    # Set up the calendar request
    calendar_request = GetCalendarRequest(start=today_str, end=today_str)

    # Fetch the calendar
    calendar = trading_client.get_calendar(filters=calendar_request)[0]

    market_open = today.replace(
        hour=calendar.open.hour,
        minute=calendar.open.minute,
        second=0
    ).astimezone(nyc)

    market_close = today.replace(
        hour=calendar.close.hour,
        minute=calendar.close.minute,
        second=0
    ).astimezone(nyc)

    # Wait until just before we might want to trade
    current_dt = datetime.datetime.today().astimezone(nyc)
    since_market_open = current_dt - market_open
    while since_market_open.seconds // 60 <= 14:
        time.sleep(1)
        current_dt = datetime.datetime.today().astimezone(nyc)
        since_market_open = current_dt - market_open

    run(get_tickers(), market_open, market_close)