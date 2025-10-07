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
    OrderSide,
    TimeInForce,
    QueryOrderStatus
)

from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pydantic import ValidationError
from pytz import timezone
import pandas as pd
import numpy as np
import datetime
import logging
import threading
import time

# Alpaca API credentials
api_key_id = 'PKON7XLHTRMPTZDJ6JB3'
api_secret = '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL'
base_url = 'https://paper-api.alpaca.markets/v2'

trading_client = TradingClient(api_key_id, api_secret, paper=True)

# Get Historical Data
def get_1000m_history_data(symbols):
    print('Getting historical Data...')
    # no keys required for crypto data
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


def get_pairs():
    print('Getting Current Ticker Data...')

    # get list of crypto pairs
    req = GetAssetsRequest(
        asset_class=AssetClass.CRYPTO,
        status=AssetStatus.ACTIVE
    )
    assets = trading_client.get_all_assets(req)

    symbols = [asset.symbol for asset in assets if asset.tradable]

    # Filter for the pairs of interest
    pairs_to_test = [(symbols[i], symbols[j]) for i in range(len(symbols)) for j in range(i + 1, len(symbols))]

    # setup stock historical data client
    stock_historical_data_client = StockHistoricalDataClient('PKON7XLHTRMPTZDJ6JB3',
                                                             '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')

    # Set the start time
    start_date = pd.to_datetime("2024-01-01").tz_localize('America/New_York')
    end_date = pd.to_datetime("2024-12-31").tz_localize('America/New_York')

    cointegration_results = []

    for symbol1, symbol2 in pairs_to_test:
        try:
            # Get data for the specified symbols
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_date, end=end_date
            )
            data = stock_historical_data_client.get_stock_bars(request_params)

            if data.df.empty:
                print(f"   No data for {symbol1} or {symbol2}. Skipping.")
                continue

            # Reset index to make 'symbol' and 'timestamp' regular columns
            df = data.df.reset_index()

            # Debug: Check if 'symbol' and 'close' columns exist after resetting index
            if 'symbol' not in df.columns or 'close' not in df.columns:
                print(f"Required columns missing in data for {symbol1} and {symbol2}.")
                continue

            # Format the 'timestamp' column to match the required format (YYYYMMDD)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d')

            # Pivot data to organize close prices by symbol
            df = df.pivot(index='timestamp', columns='symbol', values='close')

            # Ensure enough data is available
            if df.isnull().any().any() or df.shape[0] < 50:
                print(f"Insufficient data for {symbol1} and {symbol2}.")
                continue

            # Perform Johansen Test
            print(f"   Performing Johansen test on {symbol1} and {symbol2}...")
            result = coint_johansen(df[[symbol1, symbol2]].values, det_order=0, k_ar_diff=1)

            # Print the Johansen test results
            print(f"Johansen Test Results for {symbol1} and {symbol2}:")
            print("LR1 Statistics:", result.lr1)
            print("Critical Values (LR1):", result.cvt)
            print("LR2 Statistics:", result.lr2)
            print("Critical Values (LR2):", result.cvm)
            print("Eigenvalues:", result.eig)
            print("Eigenvectors:", result.evec)

            # Use the largest eigenvalue as the strength metric
            cointegration_strength = result.eig[0]
            print(f"   Cointegration Strength (Largest Eigenvalue): {cointegration_strength}")

            # Check if the test indicates cointegration
            if np.any(result.lr1 > result.cvt[:, 1]):  # 5% significance level
                cointegration_results.append((symbol1, symbol2, cointegration_strength))
            else:
                print(f"   No significant cointegration found for this pair.")

        except Exception as e:
            print(f"Error processing {symbol1} and {symbol2}: {e}")
            continue

    # Filter for unique symbols based on the strongest cointegration strength
    unique_results = []
    seen_symbols = set()

    # Sort results by strength (highest first)
    cointegration_results.sort(key=lambda x: x[2], reverse=True)

    for symbol1, symbol2, strength in cointegration_results:
        if symbol1 not in seen_symbols and symbol2 not in seen_symbols:
            unique_results.append((symbol1, symbol2, strength))
            seen_symbols.update([symbol1, symbol2])

    # Extract the top 10 strongest pairs or fewer if not enough pairs are available
    if len(unique_results) < 5:
        print(f"\nOnly {len(unique_results)} cointegrated pairs found. Returning all available pairs.")
        tradable_pairs = unique_results
    else:
        tradable_pairs = unique_results[:5]

    # Display the pairs
    print(f"\nTop Cointegrated Pairs:")
    for rank, pair in enumerate(tradable_pairs, start=1):
        print(f"{rank}. Pair: {pair[0]} & {pair[1]}, Strength: {pair[2]}")

    # Return only the symbol pairs without the strength
    return [(pair[0], pair[1]) for pair in tradable_pairs]


def run(pairs, market_open_dt, market_close_dt):
    # Establish streaming connection
    logging.basicConfig(level=logging.INFO)

    trading_stream = TradingStream('PKON7XLHTRMPTZDJ6JB3', '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL', paper=True)

    wss_client = StockDataStream('PKON7XLHTRMPTZDJ6JB3', '62NRuErPALxdxeeV3onDPbaYzeoHzdTRjhcgijOL')

    # How much of our portfolio to allocate to any one position
    risk = 0.02

    # Get tradable pairs if not provided
    if pairs is None:
        pairs = [(pair[0], pair[1]) for pair in get_pairs()]

    symbols = [symbol for pair in pairs for symbol in pair if isinstance(symbol, str)]

    print(f"Tracking {len(symbols)} symbols.")
    print('Tracking {} symbols.'.format(len(symbols)))
    minute_history = get_1000m_history_data(symbols)

    print(f"Tracking {len(pairs)} pairs.")

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

    # Keep track of what we're buying/selling
    positions = {}

    partial_fills = {}

    pair_positions = {pair: {"long": 0, "short": 0} for pair in pairs}
    kalman_states = {
        pair: {
            "beta": np.zeros(2),
            "R": np.zeros((2, 2)),
            "P": np.zeros((2, 2)),
            "Vw": 0.0001 / (1 - 0.0001) * np.eye(2),
            "Ve": 0.001,
        }
        for pair in pairs
    }

    # Tunable thresholds
    long_entry_threshold = -0.3  # Reduced sensitivity
    short_entry_threshold = 0.3
    long_exit_threshold = 0.1  # Slightly above zero for robustness
    short_exit_threshold = -0.1

    # Kalman filter update function
    def kalman_filter_update(state, x, y):
        """Update Kalman filter state."""
        beta, R, P, Vw, Ve = state["beta"], state["R"], state["P"], state["Vw"], state["Ve"]
        R = P + Vw
        yhat = np.dot(x, beta)
        Q = np.dot(x.T, np.dot(R, x)) + Ve
        e = y - yhat
        K = np.dot(R, x) / Q
        beta += K * e
        P = R - np.outer(K, x).dot(R)
        return beta, R, P, e, Q

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
            # Ensure symbol exists in minute_history
            if symbol not in minute_history:
                minute_history.update(get_1000m_history_data([symbol]))

            # Extract the minute-level timestamp
            ts = data.timestamp.replace(second=0, microsecond=0)

            # Update minute-level historical data
            try:
                current = minute_history[symbol].loc[ts]
            except KeyError:
                current = None

            if current is None:
                new_data = [
                    data.open,
                    data.high,
                    data.low,
                    data.close,
                    data.volume,
                    data.trade_count,
                    data.vwap,
                ]
            else:
                new_data = [
                    current.open,
                    max(data.high, current.high),
                    min(data.low, current.low),
                    data.close,
                    current.volume + data.volume,
                    current.trade_count + data.trade_count,
                    data.vwap,  # May need volume-weighted adjustment
                ]
                minute_history[symbol].loc[ts] = new_data

                # Cancel outdated orders
                existing_order = open_orders.get(symbol)
                if existing_order is not None:
                    submission_ts = existing_order.submitted_at.astimezone(
                        timezone('America/New_York')
                    )
                    order_lifetime = ts - submission_ts
                    if order_lifetime.seconds // 60 > 3:
                        trading_client.cancel_order_by_id(existing_order.id)
                        logging.info(f"Canceled order for {symbol}: {existing_order.id}")

            # Helper function to ensure data is aligned and normalized
            def align_and_normalize_data(data, symbol):
                if "timestamp" not in data.columns:
                    data = data.rename(columns={"index": "timestamp"})
                data = data.set_index("timestamp")
                data.index = pd.to_datetime(data.index)
                logging.debug(f"Data for {symbol} normalized with {len(data)} rows.")
                return data

            # Process trading signals only for symbols in pairs
            for symbol1, symbol2 in pairs:
                # Ensure both symbols have updated historical data
                for sym in [symbol1, symbol2]:
                    if sym not in minute_history or minute_history[sym].empty:
                        try:
                            fetched_data = get_1000m_history_data([sym])
                            if fetched_data and sym in fetched_data:
                                minute_history[sym] = fetched_data[sym]
                                logging.info(f"Fetched and updated data for {sym}")
                        except Exception as e:
                            logging.error(f"Failed to fetch or update data for {sym}: {e}")
                            continue

                # Extract and normalize data for both symbols
                try:
                    data1 = align_and_normalize_data(minute_history[symbol1].reset_index(), symbol1)
                    data2 = align_and_normalize_data(minute_history[symbol2].reset_index(), symbol2)
                except KeyError as e:
                    logging.error(f"Missing historical data for {symbol1} or {symbol2}: {e}")
                    continue

                # Merge and align data using an outer join
                try:
                    aligned_data = (
                        pd.concat(
                            [data1, data2],
                            axis=1,
                            join="outer",
                            keys=[symbol1, symbol2]
                        )
                        .sort_index(level=0)  # Sort by timestamp
                        .ffill()  # Forward-fill missing data
                    )
                    logging.info(f"Aligned data points for {symbol1}/{symbol2}: {len(aligned_data)}")
                except Exception as e:
                    logging.error(f"Error aligning data for {symbol1}/{symbol2}: {e}")
                    continue

                # Skip pairs with insufficient aligned data
                if aligned_data.empty or len(aligned_data) < 50:
                    logging.info(f"Skipping {symbol1}/{symbol2}: insufficient aligned data points.")
                    continue

                # Extract close prices for both symbols
                try:
                    x = aligned_data[(symbol1, "close")]
                    y = aligned_data[(symbol2, "close")]
                except KeyError as e:
                    logging.error(f"Missing close price data for {symbol1}/{symbol2}: {e}")
                    continue

                if x.empty or y.empty:
                    logging.info(f"Skipping {symbol1}/{symbol2}: close price data missing.")
                    continue

                # Kalman filter update
                state = kalman_states.get((symbol1, symbol2))
                if not state or len(x) < 1 or len(y) < 1:
                    logging.warning(f"Skipping {symbol1}/{symbol2}: insufficient Kalman state or data.")
                    continue

                x_augmented = np.vstack([x.values, np.ones_like(x.values)]).T
                beta, R, P, e, Q = kalman_filter_update(state, x_augmented[-1], y.iloc[-1])
                state["beta"], state["R"], state["P"] = beta, R, P
                logging.info(
                    f"Updated Kalman filter for {symbol1}/{symbol2}: beta={beta}, residual={e}, variance={Q}"
                )

                # Scaled residual
                e_scaled = e / np.sqrt(Q + 1e-8)

                # Generate signals
                long_entry = e_scaled < long_entry_threshold
                short_entry = e_scaled > short_entry_threshold
                long_exit = e_scaled > long_exit_threshold
                short_exit = e_scaled < short_exit_threshold

                logging.info(
                    f"Signals for {symbol1}/{symbol2}: long_entry={long_entry}, short_entry={short_entry}, "
                    f"long_exit={long_exit}, short_exit={short_exit}"
                )

                # Determine position adjustments
                current_position = pair_positions[(symbol1, symbol2)]
                target_position = {"long": 0, "short": 0}

                if long_entry:
                    target_position["long"] = 1
                    target_position["short"] = -1
                elif short_entry:
                    target_position["long"] = -1
                    target_position["short"] = 1
                elif long_exit and current_position["long"] > 0:
                    target_position["long"] = 0
                elif short_exit and current_position["short"] < 0:
                    target_position["short"] = 0
                else:
                    logging.info(f"No changes to target positions for {symbol1}/{symbol2}.")

                logging.info(f"Target position for {symbol1}/{symbol2}: {target_position}")

                # Calculate time since market open and until market close
                since_market_open = ts - market_open_dt
                until_market_close = market_close_dt - ts

                # Ensure signals are checked only within the designated time frame
                if since_market_open.seconds // 60 > 15 and until_market_close.seconds // 60 > 15:
                    # Extract and validate limit prices for both symbols in the pair
                    limit_price_symbol1 = aligned_data[(symbol1, "close")].iloc[-1]
                    limit_price_symbol2 = aligned_data[(symbol2, "close")].iloc[-1]

                    # Round the limit prices according to the specified constraints
                    if limit_price_symbol1 >= 1.0:
                        limit_price_symbol1 = round(limit_price_symbol1, 2)
                    else:
                        limit_price_symbol1 = round(limit_price_symbol1, 4)

                    if limit_price_symbol2 >= 1.0:
                        limit_price_symbol2 = round(limit_price_symbol2, 2)
                    else:
                        limit_price_symbol2 = round(limit_price_symbol2, 4)

                    if limit_price_symbol1 <= 0 or limit_price_symbol2 <= 0:
                        print(
                            f"Skipping {symbol1}/{symbol2}: invalid limit prices "
                            f"({limit_price_symbol1}, {limit_price_symbol2})."
                        )
                        continue

                    print(f"Rounded limit prices: {symbol1}={limit_price_symbol1}, {symbol2}={limit_price_symbol2}")

                    try:
                        # Check for entry signals
                        print("Checking for entry signals")
                        position_symbol1 = positions.get(symbol1, 0)
                        position_symbol2 = positions.get(symbol2, 0)

                        if position_symbol1 > 0:
                            print(f"Skipping entry check for {symbol1} because the position is greater than 0.")
                        else:
                            print(f"Checking entry for {symbol1}")

                        if position_symbol2 > 0:
                            print(f"Skipping entry check for {symbol2} because the position is greater than 0.")
                        else:
                            print(f"Checking entry for {symbol2}")

                        # Check for exit signals
                        position_symbol1 = positions.get(symbol1, 0)
                        position_symbol2 = positions.get(symbol2, 0)

                        if position_symbol1 == 0:
                            print(f"Skipping exit check for {symbol1} because the position is equal to 0.")
                        else:
                            print(f"Checking exit for {symbol1} with position quantity: {position_symbol1}")

                        if position_symbol2 == 0:
                            print(f"Skipping exit check for {symbol2} because the position is equal to 0.")
                        else:
                            print(f"Checking exit for {symbol2} with position quantity: {position_symbol2}")

                        # Calculate position sizes for both symbols
                        signal_strength = max(0.01, abs(e / np.sqrt(Q)))  # Confidence in the signal
                        max_exposure = portfolio_value * risk

                        # Calculate position sizes using individual limit prices
                        position_size_symbol1 = max(0.001, min(max_exposure,
                                                               max_exposure * signal_strength) / limit_price_symbol1)
                        position_size_symbol2 = max(0.001, min(max_exposure,
                                                               max_exposure * signal_strength) / limit_price_symbol2)

                        # Round position sizes to four decimal points
                        position_size_symbol1 = round(position_size_symbol1, 4)
                        position_size_symbol2 = round(position_size_symbol2, 4)

                        print(
                            f"Order details for {symbol1}/{symbol2}: "
                            f"position_size_symbol1={position_size_symbol1}, position_size_symbol2={position_size_symbol2}, "
                            f"limit_price_symbol1={limit_price_symbol1}, limit_price_symbol2={limit_price_symbol2}"
                        )

                        # Handle long entry for symbol1
                        if target_position["long"] > current_position["long"]:
                            qty_to_buy = position_size_symbol1
                            print(f"Entering long position for {symbol1}: qty={qty_to_buy}")
                            trading_client.submit_order(
                                LimitOrderRequest(
                                    symbol=symbol1,
                                    limit_price=limit_price_symbol1,
                                    qty=f"{qty_to_buy:.4f}",  # Rounded to four decimal places
                                    side=OrderSide.BUY,
                                    time_in_force=TimeInForce.GTC
                                )
                            )

                        # Handle long exit for symbol1
                        elif target_position["long"] < current_position["long"]:
                            qty_to_sell = position_size_symbol1
                            print(f"Exiting long position for {symbol1}: qty={qty_to_sell}")
                            trading_client.submit_order(
                                LimitOrderRequest(
                                    symbol=symbol1,
                                    limit_price=limit_price_symbol1,
                                    qty=f"{qty_to_sell:.4f}",  # Rounded to four decimal places
                                    side=OrderSide.SELL,
                                    time_in_force=TimeInForce.GTC
                                )
                            )

                        # Handle short entry for symbol2
                        if target_position["short"] > current_position["short"]:
                            qty_to_sell = position_size_symbol2
                            print(f"Entering short position for {symbol2}: qty={qty_to_sell}")
                            trading_client.submit_order(
                                LimitOrderRequest(
                                    symbol=symbol2,
                                    limit_price=limit_price_symbol2,
                                    qty=f"{qty_to_sell:.4f}",  # Rounded to four decimal places
                                    side=OrderSide.SELL,
                                    time_in_force=TimeInForce.GTC
                                )
                            )

                        # Handle short exit for symbol2
                        elif target_position["short"] < current_position["short"]:
                            qty_to_buy = position_size_symbol2
                            print(f"Exiting short position for {symbol2}: qty={qty_to_buy}")
                            trading_client.submit_order(
                                LimitOrderRequest(
                                    symbol=symbol2,
                                    limit_price=limit_price_symbol2,
                                    qty=f"{qty_to_buy:.4f}",  # Rounded to four decimal places
                                    side=OrderSide.BUY,
                                    time_in_force=TimeInForce.GTC
                                )
                            )

                        # Update pair positions
                        pair_positions[(symbol1, symbol2)] = target_position
                        print(f"Updated positions for {symbol1}/{symbol2}: {pair_positions[(symbol1, symbol2)]}")

                    except Exception as e:
                        print(e)


                elif (
                        until_market_close.seconds // 60 <= 15
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

    # Call run function with precomputed pairs
    pairs = get_pairs()
    run(pairs, market_open, market_close)
