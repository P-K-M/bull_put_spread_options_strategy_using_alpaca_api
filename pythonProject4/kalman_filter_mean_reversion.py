# Imports and basic configuration
# Purpose: group and document all imports and top-level configuration used across
# the Bull Put Spread strategy.

# -------------------------
# Standard library imports
# -------------------------
import threading
import logging
import traceback
import os, json
import time
from zoneinfo import ZoneInfo
from datetime import date, datetime, timedelta, timezone

# -------------------------
# Numerical & statistics
# -------------------------
# pandas / numpy are used for data manipulation and numeric arrays. scipy provides
# probability functions and solvers used by pricing/IV calculations.
import pandas as pd
import numpy as np

from statsmodels.tsa.vector_ar.vecm import coint_johansen

# -------------------------
# Typing and developer helpers
# -------------------------
# Used to provide type hints which improves readability and helps code review.
from typing import TypedDict, Optional, Dict, Any, List, Tuple

# -------------------------
# Alpaca (data + trading)
# -------------------------
# Alpaca packages provide both historical & live data access and trading client
# objects. They are central for live/paper trading and option chain retrieval.
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.requests import (
    GetAssetsRequest,
    LimitOrderRequest,
    GetCalendarRequest
)
from alpaca.trading.enums import (
    AssetStatus,
    OrderSide,
    OrderClass,
    TimeInForce,
    ContractType,
    AssetClass,
    PositionSide
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("iron_condor_strategy")
logging.basicConfig(level=logging.INFO)

API_KEY = "PKX0U7WKPVDN9DI6VTWS"
SECRET_KEY = "gJaqPzEK2sfADA4uhkdf9ldA3YIgof9D3ocYVpnE"
PEPER_URL = "https://paper-api.alpaca.markets/v2"
PAPER = True

# Instantiate trading client for calendar use
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

class KalmanFilterStrategy:
    """
    Pairs trading strategy using Kalman Filter regression
    with Alpaca data and trading API integration.
    """
    def __init__(self,
        symbols: List[str],
        notional_per_pair: float = 10000.0,
        delta: float = 0.0001,
        Ve: float = 1e-3,
        threshold_z: float = 1.0,
        min_history_bars: int = 60,
        env: str = "paper",):

        """Initialize strategy with Alpaca API clients"""
        # === Alpaca API Clients ===
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
        self.trading_stream = TradingStream(API_KEY, SECRET_KEY, paper=PAPER)
        self.wss_client = StockDataStream(API_KEY, SECRET_KEY)
        self.stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        # === Trading Configuration ===
        self.timezone = ZoneInfo('America/New_York')
        self.today = datetime.now(self.timezone).date()
        self.logger = logger
        # self.logger.setLevel(logging.DEBUG)
        self.logger.debug("KalmanFilterStrategy")

        # === Buying Power ===
        self.buy_power_limit = 0.30  # Use up to 30% of account buying power per spread
        self.buying_power = float(self.trading_client.get_account().buying_power)
        self.buying_power_limit = self.buying_power * self.buy_power_limit

        self.minute_history = {}
        self.pairs = []
        self.open_orders = {}

        self.notional_per_pair = notional_per_pair
        self.delta = delta
        self.Ve = Ve
        self.threshold_z = threshold_z
        self.min_history_bars = min_history_bars

        # === Streaming state ===
        self.price_history: Dict[str, pd.Series] = {}  # will hold rolling close prices
        self.symbol_x: Optional[str] = None
        self.symbol_y: Optional[str] = None

        # -----------------------------------------------------------------------------
        # === Utility functions (historical data, get tradable pairs) ===
        # -----------------------------------------------------------------------------

    def get_1000m_history_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Retrieve the last 1000 minutes of historical data for a list of symbols.

                Args:
                    symbols (List[str]): List of ticker symbols to fetch minute-level data for.

                Returns:
                    Dict[str, pd.DataFrame]: Dictionary mapping symbols to their OHLCV history.
                """
        self.logger.info("Fetching 1000-minute historical data...")
        minute_history = {}

        for idx, symbol in enumerate(symbols, start=1):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=1000
                )
                bars = self.stock_data_client.get_stock_bars(request)
                minute_history[symbol] = bars.df
                self.logger.info(f"[{idx}/{len(symbols)}] Retrieved data for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")

        self.logger.info("Historical data retrieval complete.")
        return minute_history

    def get_pairs(self):

        """
        Discover cointegrated crypto pairs using Johansen test.
        Returns list of (sym1, sym2).
        """

        self.logger.info("[SCREEN] Starting Kalman Filter asset selection...")

        # === setup stock historical data client ===
        stock_historical_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        # === Load ETF symbols from CSV file ===
        etf_symbols = pd.read_csv('etf.csv')['Symbols'].tolist()

        # === Step 1: Get tradable assets (filter only predefined watchlist) ===
        assets = self.trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )

        symbols = [a.symbol for a in assets if a.tradable and a.symbol in etf_symbols]

        # Filter for the pairs of interest
        pairs_to_test = [(symbols[i], symbols[j]) for i in range(len(symbols)) for j in range(i + 1, len(symbols))]

        cointegration_results = []

        for symbol1, symbol2 in pairs_to_test:
            try:
                # Get data for the specified symbols
                # Historical data
                start_dt = (datetime.now(self.timezone) - timedelta(days=180)).date()
                end_dt = datetime.now(self.timezone).date()

                data = self.stock_data_client.get_stock_bars(
                    StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_dt, end=end_dt)
                )

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

    # -----------------------
    # Kalman Regression (pairs)
    # -----------------------
    def kalman_regression_pair(self, x: np.ndarray, y: np.ndarray, burn_in: int = 30):
        """
        Compute time-varying beta (slope) and residuals for y = beta0*x + beta1 (intercept).
        Uses Kalman Filter regression.

        Args:
            x (np.ndarray): Series of independent variable prices
            y (np.ndarray): Series of dependent variable prices
            burn_in (int): Number of initial points to ignore for noisy estimates

        Returns:
            dict: {"beta": beta, "yhat": yhat, "e": residuals, "Q": variance}
        """
        n = len(x)
        X = np.vstack([x, np.ones(n)]).T
        yhat = np.full(n, np.nan)
        e = np.full(n, np.nan)
        Q = np.full(n, np.nan)
        beta = np.full((2, n), np.nan)

        R = np.zeros((2, 2))
        P = np.zeros((2, 2))
        Vw = self.delta / (1 - self.delta) * np.eye(2)

        # Seed with OLS for better early estimates
        if n >= burn_in:
            bx, bi = np.polyfit(x[:burn_in], y[:burn_in], 1)
            beta[:, 0] = [bx, bi]
        else:
            beta[:, 0] = 0.0

        for t in range(n):
            if t > 0:
                beta[:, t] = beta[:, t - 1].copy()
                R = P + Vw
            Q[t] = max(X[t].dot(R).dot(X[t].T) + self.Ve, 1e-12)  # guard against zero
            yhat[t] = X[t].dot(beta[:, t])
            e[t] = y[t] - yhat[t]
            K = R.dot(X[t].T) / Q[t]
            beta[:, t] = beta[:, t] + K * e[t]
            P = R - np.outer(K, X[t]).dot(R)

        return {"beta": beta, "yhat": yhat, "e": e, "Q": Q}

    # -----------------------
    # Signal generator (no look-ahead)
    # -----------------------
    def generate_pair_signal(self, series_x: pd.Series, series_y: pd.Series, max_hold: int = 60):
        """
        Generate trading signal from Kalman residuals.
        +1: LONG x, SHORT y
        -1: SHORT x, LONG y
        0: Exit / flat
        """
        if len(series_x) < self.min_history_bars:
            return 0, None

        res = self.kalman_regression_pair(series_x.values, series_y.values)
        e = pd.Series(res["e"], index=series_x.index).shift(1)
        Q = pd.Series(res["Q"], index=series_x.index).shift(1)
        beta = pd.DataFrame(res["beta"].T, index=series_x.index, columns=["slope", "intercept"]).shift(1)

        last_idx = series_x.index[-1]
        resid, var = e.iloc[-1], Q.iloc[-1]
        if pd.isna(resid) or pd.isna(var):
            return 0, None

        threshold = np.sqrt(max(var, 1e-12)) * self.threshold_z

        # Entry & exit logic
        signal = 0
        if resid < -threshold:
            signal = 1
        elif resid > threshold:
            signal = -1
        elif abs(resid) < 0.1 * threshold:  # small buffer band to avoid churn
            signal = 0

        return signal, {"beta": beta.iloc[-1], "resid": resid, "threshold": threshold}

    # ===============================================================
    # Leg Sizing
    # ===============================================================
    def compute_leg_qtys(self,
                         px_x: float,
                         px_y: float,
                         hedge_slope: float,
                         side: int,
                         per_trade_cap_frac: float = 0.03,
                         max_trade_value: float = 5000.0,
                         tick_size: float = 0.01) -> tuple[int, int, float]:
        """
        Compute hedge-adjusted integer share quantities for a pair trade with
        portfolio risk caps, buying power enforcement, hedge slope, and a tick-aware limit price.

        Args:
            px_x, px_y (float): Current prices of symbols X and Y.
            hedge_slope (float): Hedge ratio from Kalman regression.
            side (int): +1 for long spread, -1 for short spread.
            per_trade_cap_frac (float): Fraction of portfolio allowed per trade (default=3%).
            max_trade_value (float): Absolute cap on trade dollar size.
            tick_size (float): Minimum price increment, default 0.01.

        Returns:
            (int, int, float): Final hedge-balanced quantities (qx, qy) and tick-rounded limit price.
        """
        try:
            if px_x <= 0 or px_y <= 0 or hedge_slope <= 0:
                return 0, 0, 0.0

            # --------- Account values (pulled internally) ---------
            try:
                account = self.trading_client.get_account()
                portfolio_value = float(account.portfolio_value)
                buying_power = float(account.buying_power)
            except Exception as e:
                self.logger.error("[ACCOUNT ERROR] Could not fetch account info: %s", e)
                return 0, 0, 0.0

            # --------- Risk Caps ---------
            per_trade_cap = portfolio_value * per_trade_cap_frac
            target_dollar = min(per_trade_cap, max_trade_value)
            per_leg_dollar = target_dollar / 2.0

            def safe_floor(dividend, divisor):
                return max(0, int(dividend // divisor)) if divisor > 0 else 0

            # Max shares allowed by caps
            qty_x_cap = safe_floor(per_leg_dollar, px_x)
            qty_y_cap = safe_floor(per_leg_dollar, px_y)

            # Max shares allowed by buying power (combined cost of x + slope*y)
            max_qty_bp = safe_floor(buying_power, px_x + abs(hedge_slope) * px_y)
            qty_x_cap = min(qty_x_cap, max_qty_bp)
            qty_y_cap = min(qty_y_cap, max_qty_bp)

            if qty_x_cap <= 0 or qty_y_cap <= 0:
                self.logger.info("[SKIP] Not enough BP or cap for any shares.")
                return 0, 0, 0.0

            # --------- Hedge Adjustment ---------
            qx = min(qty_x_cap, int(qty_y_cap / hedge_slope))
            qy = int(round(qx * hedge_slope))

            if qx <= 0 or qy <= 0:
                self.logger.info("[SKIP] Hedge-adjusted quantities invalid.")
                return 0, 0, 0.0

            # --------- Synthetic Spread Value ---------
            spread_val = side * (px_x - hedge_slope * px_y)

            # Tick-aware rounding for limit price
            ticks = int(spread_val // tick_size)
            limit_price = round(ticks * tick_size, 2)

            # --------- Logging ---------
            if qx == qty_x_cap:
                self.logger.info("[SIZING] Limited by X-leg cap ($%.2f)", per_leg_dollar)
            elif qy == qty_y_cap:
                self.logger.info("[SIZING] Limited by Y-leg cap ($%.2f)", per_leg_dollar)
            elif qx == max_qty_bp or qy == max_qty_bp:
                self.logger.info("[SIZING] Limited by Buying Power.")

            self.logger.info(
                "[SIZING RESULT] qx=%d, qy=%d, hedge_slope=%.3f, limit_price=%.2f",
                qx, qy, hedge_slope, limit_price
            )

            return (qx, qy, limit_price)

        except Exception as e:
            self.logger.error(f"[SIZING ERROR] compute_leg_qtys failed: {e}", exc_info=True)
            return 0, 0, 0.0

    # -----------------------
    # Execution Helpers
    # -----------------------

    def get_price_for_order(self, symbol: str, side: OrderSide) -> float | None:
        """Resolve execution price for BUY/SELL using quotes or minute_history fallback."""
        try:
            if side == OrderSide.BUY:
                px = getattr(self, "quote_dict", {}).get(symbol, {}).get("ask")
            else:
                px = getattr(self, "quote_dict", {}).get(symbol, {}).get("bid")

            if px:
                return float(px)

            # fallback: last close from minute history
            if symbol in self.minute_history:
                return float(self.minute_history[symbol]["close"].iloc[-1])

            return None
        except Exception:
            return None

    def place_limit_order_with_timeout(self,
                                       symbol: str,
                                       side: OrderSide,
                                       qty: int,
                                       limit_price: float,
                                       timeout_seconds: int = 180,
                                       time_in_force: TimeInForce = TimeInForce.DAY):
        """
        Place a single LIMIT order and cancel if unfilled after timeout_seconds.
        Returns the order object or None.
        """
        if qty <= 0 or limit_price <= 0:
            self.logger.warning(f"[ORDER SKIP] Invalid params for {symbol}: qty={qty}, price={limit_price}")
            return None

        try:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=time_in_force,
                limit_price=float(limit_price)
            )
            order = self.trading_client.submit_order(req)
            order_id = getattr(order, "id", None)
            self.logger.info(f"[LIMIT ORDER] {side.name} {qty} {symbol} @ {limit_price:.4f} (id={order_id})")

            if order_id:
                self.open_orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "limit_price": limit_price,
                    "submitted_at": datetime.now(timezone.utc)
                }

                def _cancel_if_open(oid):
                    try:
                        existing = self.trading_client.get_order_by_id(oid)
                        status = getattr(existing, "status", "").lower()
                        if status not in ("filled", "canceled", "rejected", "expired"):
                            self.trading_client.cancel_order_by_id(oid)
                            self.logger.info(f"[AUTO-CANCEL] Order {oid} canceled after {timeout_seconds}s")
                        # cleanup
                        self.open_orders.pop(oid, None)
                    except Exception as e:
                        self.logger.error(f"[AUTO-CANCEL ERROR] {e}", exc_info=True)

                t = threading.Timer(timeout_seconds, _cancel_if_open, args=(order_id,))
                t.daemon = True
                t.start()

            return order

        except Exception as e:
            self.logger.error(f"[ORDER ERROR] Failed to place {side.name} {qty} {symbol}: {e}", exc_info=True)
            return None

    def submit_pair_open_limit(self,
                               symbol_long: str,
                               symbol_short: str,
                               qty_long: int,
                               qty_short: int,
                               timeout_seconds: int = 180):
        """
        Open a pair: Buy long leg, Sell short leg via limit orders.
        Returns (order_long, order_short).
        """
        try:
            # Initialize placeholders
            price_long = None
            price_short = None

            # Get reference prices
            price_long = self.get_price_for_order(symbol_long, OrderSide.BUY)
            price_short = self.get_price_for_order(symbol_short, OrderSide.SELL)

            if not price_long or not price_short:
                self.logger.warning(f"[PAIR OPEN] Missing price for {symbol_long}/{symbol_short}")
                return None, None

            tol = getattr(self, "limit_price_tolerance", 0.001)

            buy_limit = round(price_long * (1 + tol), 4)
            sell_limit = round(price_short * (1 - tol), 4)

            o_long = self.place_limit_order_with_timeout(
                symbol_long, OrderSide.BUY, qty_long, buy_limit, timeout_seconds
            )
            o_short = self.place_limit_order_with_timeout(
                symbol_short, OrderSide.SELL, qty_short, sell_limit, timeout_seconds
            )

            return o_long, o_short

        except Exception as e:
            self.logger.error(f"[PAIR OPEN ERROR] {e}", exc_info=True)
            return None, None

    def submit_pair_close(self,
                          symbol_long: str,
                          symbol_short: str,
                          qty: int,
                          timeout_seconds: int = 180):
        """
        Close a pair: Sell long leg, Buy short leg via limit orders.
        Returns (order_long, order_short).
        """
        try:
            px_long = self.get_price_for_order(symbol_long, OrderSide.SELL)
            px_short = self.get_price_for_order(symbol_short, OrderSide.BUY)

            if not px_long or not px_short:
                self.logger.warning(f"[PAIR CLOSE] Missing price for {symbol_long}/{symbol_short}")
                return None, None

            tol = getattr(self, "limit_price_tolerance", 0.001)
            sell_limit = round(px_long * (1 - tol), 4)
            buy_limit = round(px_short * (1 + tol), 4)

            o_long = self.place_limit_order_with_timeout(symbol_long, OrderSide.SELL, qty, sell_limit, timeout_seconds)
            o_short = self.place_limit_order_with_timeout(symbol_short, OrderSide.BUY, qty, buy_limit, timeout_seconds)

            return o_long, o_short

        except Exception as e:
            self.logger.error(f"[PAIR CLOSE ERROR] {e}", exc_info=True)
            return None, None

    # ==============================================================
    # SECTION 4: State Management & Signal Evaluation
    # ==============================================================

    def load_pair_state(self, path: str = "pair_state.json") -> list[dict]:
        """
        Load saved pair state from disk (used only at startup).
        Falls back to [] if file is missing or corrupt.
        """
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    state = json.load(f)
                if isinstance(state, list):
                    self.logger.info(f"[STATE LOADED] {len(state)} pairs from {path}")
                    return self.cleanup_pairs(state)
                else:
                    self.logger.warning(f"[LOAD ERROR] {path} did not contain a list")
        except Exception as e:
            self.logger.warning(f"[LOAD ERROR] Could not read {path}: {e}")

        return []

    def save_pair_state(self, state: list[dict], path: str = "pair_state.json") -> None:
        """
        Persist active pair state to disk safely.
        Writes to a temp file then renames for atomicity.
        """
        try:
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f, default=str, indent=2)
            os.replace(tmp_path, path)
            self.logger.debug(f"[STATE SAVED] {len(state)} pairs to {path}")
        except Exception as e:
            self.logger.error(f"[SAVE ERROR] Could not save pair state: {e}", exc_info=True)

    def cleanup_pairs(self, pairs: list[dict]) -> list[dict]:
        """
        Clean up invalid or duplicate pair states.
        Ensures qty is int > 0 and both legs are present.
        """
        cleaned = []
        seen = set()
        for p in pairs:
            try:
                long_sym = p.get("symbol_long")
                short_sym = p.get("symbol_short")
                qty = int(float(p.get("qty", 0)))
                if long_sym and short_sym and qty > 0:
                    key = f"{long_sym}-{short_sym}"
                    if key not in seen:
                        seen.add(key)
                        cleaned.append({
                            "symbol_long": long_sym,
                            "symbol_short": short_sym,
                            "qty": qty,
                            "entered_at": p.get("entered_at", str(datetime.now(timezone.utc))),
                            "avg_entry_long": float(p.get("avg_entry_long", 0)),
                            "avg_entry_short": float(p.get("avg_entry_short", 0)),
                        })
                else:
                    self.logger.warning(f"[CLEANUP] Dropping invalid entry: {p}")
            except Exception as e:
                self.logger.warning(f"[CLEANUP ERROR] Could not process {p}: {e}")
        return cleaned

    def sync_with_alpaca(self) -> list[dict]:
        """
        Sync live equity/crypto positions with Alpaca.
        Reconcile with saved pair state.
        """
        try:
            positions = self.trading_client.get_all_positions()
        except Exception as e:
            self.logger.error(f"[SYNC ERROR] Failed to fetch positions: {e}", exc_info=True)
            return []

        # Flatten Alpaca response
        pos_map: dict[str, dict] = {}
        for pos in positions:
            try:
                pos_data = pos.model_dump() if hasattr(pos, "model_dump") else pos.__dict__
                sym = pos_data.get("symbol")
                if not sym:
                    continue
                pos_map[sym] = {
                    "symbol": sym,
                    "qty": abs(int(float(pos_data.get("qty", 0)))),
                    "side": pos_data.get("side"),
                    "avg_entry": float(pos_data.get("avg_entry_price", 0))
                }
            except Exception as e:
                self.logger.warning(f"[SYNC WARN] Skipping position parse: {e}")

        # Load old state
        old_state = self.load_pair_state()
        old_map = {f"{s['symbol_long']}-{s['symbol_short']}": s for s in old_state}

        # Match into pairs
        live_pairs = []
        for key, pair in old_map.items():
            long_sym, short_sym = pair["symbol_long"], pair["symbol_short"]
            if long_sym in pos_map and short_sym in pos_map:
                updated_pair = {
                    "symbol_long": long_sym,
                    "symbol_short": short_sym,
                    "qty": min(pos_map[long_sym]["qty"], pos_map[short_sym]["qty"]),
                    "entered_at": pair.get("entered_at", str(datetime.now(timezone.utc))),
                    "avg_entry_long": pos_map[long_sym]["avg_entry"],
                    "avg_entry_short": pos_map[short_sym]["avg_entry"],
                }
                live_pairs.append(updated_pair)
            else:
                self.logger.info(f"[SYNC DROP] Missing one leg in Alpaca for {key}")

        live_pairs = self.cleanup_pairs(live_pairs)
        self.logger.info(f"[ALPACA SYNC] {len(live_pairs)} live pairs reconciled")

        for p in live_pairs:
            self.logger.debug(
                f"[PAIR] LONG {p['symbol_long']} / SHORT {p['symbol_short']} | "
                f"Qty={p['qty']} | AvgLong={p['avg_entry_long']} | AvgShort={p['avg_entry_short']}"
            )

        # Persist
        self.save_pair_state(live_pairs)

        # Track active symbols
        self.symbols_with_open_pairs = {f"{p['symbol_long']}-{p['symbol_short']}" for p in live_pairs}
        return live_pairs

    def evaluate_pair_signals(self) -> tuple[int, dict | None]:
        """
        Evaluate the latest Kalman-based signal for the active pair.
        Returns (signal, details) where:
            +1 → LONG x, SHORT y
            -1 → SHORT x, LONG y
             0 → Hold / Exit
        """
        try:
            series_x = self.price_history.get(self.symbol_x, pd.Series(dtype=float)).dropna()
            series_y = self.price_history.get(self.symbol_y, pd.Series(dtype=float)).dropna()

            # Require sufficient history
            if len(series_x) < self.min_history_bars or len(series_y) < self.min_history_bars:
                return 0, None

            # Align timestamps
            aligned = pd.concat([series_x, series_y], axis=1, join="inner").dropna()
            series_x, series_y = aligned.iloc[:, 0], aligned.iloc[:, 1]

            signal, details = self.generate_pair_signal(series_x, series_y)

            if details:
                self.logger.info(
                    f"[KALMAN] {self.symbol_x}/{self.symbol_y} | "
                    f"Signal={signal} Resid={details['resid']:.4f} "
                    f"Thresh={details['threshold']:.4f} "
                    f"Slope={details['beta']['slope']:.4f} "
                    f"Int={details['beta']['intercept']:.4f}"
                )

            return signal, details

        except Exception as e:
            self.logger.error(f"[EVAL ERROR] {e}", exc_info=True)
            return 0, None

    def run(self, market_open, market_close, pairs: list | None = None):
        """
        Start Kalman Filter pairs trading algorithm with time-controlled entry logic and threading.
        - market_open, market_close: timezone-aware datetimes for today.
        - pairs: optional list of (sym_x, sym_y). If None, get_pairs() will be used to find top pairs.
        """
        logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG)

        # Alpaca streaming clients (use same style as your BPS code)
        self.trading_stream = TradingStream(API_KEY, SECRET_KEY, paper=True)
        self.wss_client = StockDataStream(API_KEY, SECRET_KEY)

        # Runtime state (mirror BPS)
        self.minute_history = {}
        self.positions = {}
        self.partial_fills = {}
        self.open_orders = {}
        self.lock = threading.Lock()
        self.last_screening_minute = {}  # keyed by pair tuple (sym_x, sym_y)
        self.entry_log = {}

        # Day-of-week check (same as BPS)
        today = datetime.now().date()
        if today.weekday() > 4:
            self.logger.info("Today is not a trading day (Mon–Fri). Exiting.")
            return

        # Discover pairs if not passed
        if pairs is None:
            pairs = self.get_pairs()
        if not pairs:
            self.logger.info("[RUN] No candidate pairs found. Exiting.")
            return

        self.pairs = pairs[:5]  # trade at most top-5 by default (same idea as your get_pairs)
        self.logger.info(f"[PAIRS] Selected pairs: {self.pairs}")

        # Flatten symbols for subscription and history bootstrap
        symbols = sorted({s for p in self.pairs for s in p})
        for p in self.pairs:
            # initialize per-pair last screening times
            self.last_screening_minute[(p[0], p[1])] = None

        # Bootstrap minute history for all symbols (1000 min)
        self.minute_history = self.get_1000m_history_data(symbols)

        # Load persisted active pairs and sync with Alpaca
        try:
            self.active_pairs = self.sync_with_alpaca() or self.load_pair_state() or []
        except Exception:
            self.active_pairs = self.load_pair_state() or []

        self.logger.info(f"[STATE] Active pairs loaded: {len(self.active_pairs)}")

        # -----------------------
        # Trading stream thread (order updates)
        # -----------------------
        def run_trading_stream():
            async def handle_trade_updates(data):
                try:
                    symbol = data.order.symbol
                    event = data.event
                    qty = float(data.order.qty or 0)
                    price = float(getattr(data.order, "limit_price", 0) or 0)

                    self.logger.debug(
                        f"[TRADE UPDATE] {event.upper()} received for {symbol} (qty={qty}, price={price})")

                    if event == "new":
                        # "position_qty" may or may not be present in this payload
                        position_qty = float(getattr(data, "position_qty", 0))
                        self.positions[symbol] = position_qty
                        self.open_orders[symbol] = data.order
                        self.logger.info(f"[NEW ORDER] {symbol}: qty={qty}, price={price}, position_qty={position_qty}")

                    elif event == "partial_fill":
                        prev = self.partial_fills.get(symbol, 0)
                        self.positions[symbol] = qty - prev
                        self.partial_fills[symbol] = qty
                        self.open_orders[symbol] = data.order
                        self.logger.info(
                            f"[PARTIAL FILL] {symbol}: filled={qty}, remaining={getattr(data.order, 'remaining_qty', 'N/A')}")

                    elif event == "fill":
                        # filled -> update positions and clear open order
                        self.positions[symbol] = qty
                        self.partial_fills[symbol] = 0
                        self.open_orders[symbol] = None
                        self.logger.info(f"[FILLED] {symbol}: total position now {qty}")

                    elif event in ["canceled", "rejected"]:
                        self.partial_fills[symbol] = 0
                        self.open_orders[symbol] = None
                        self.logger.warning(f"[{event.upper()}] Order for {symbol} was {event}")

                except Exception as e:
                    self.logger.error(f"[ERROR] Exception in trade update handler: {e}", exc_info=True)

            @self.trading_stream.subscribe_trade_updates
            async def print_trade_update(*symbols):
                print('trade update', *symbols)

            # attach and run
            self.trading_stream.subscribe_trade_updates(handle_trade_updates)
            self.trading_stream.run()

        # -----------------------
        # Market data thread (bars) — similar to BPS handle_second_bar
        # -----------------------
        def run_wss_client():
            async def handle_second_bar(data):
                try:
                    symbol = data.symbol
                    ts = data.timestamp

                    # --- normalize timestamp to datetime (safe) ---
                    if isinstance(ts, str):
                        # try common formats; add more if needed
                        try:
                            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            ts = pd.to_datetime(ts)
                    elif not isinstance(ts, datetime):
                        ts = pd.to_datetime(ts).to_pydatetime()

                    ts = ts.replace(second=0, microsecond=0)
                    self.logger.debug(f"[BAR] {symbol} @ {ts}")

                    # --- ensure we have minute history DataFrame for this symbol ---
                    if symbol not in self.minute_history or self.minute_history[symbol] is None:
                        try:
                            self.minute_history[symbol] = self.get_1000m_history_data([symbol]).get(symbol,
                                                                                                    pd.DataFrame())
                        except Exception as e:
                            self.logger.error(f"[HIST FETCH ERR] Failed to fetch history for {symbol}: {e}",
                                              exc_info=True)
                            return

                    df = self.minute_history[symbol]

                    # --- normalize df index to MultiIndex (symbol, timestamp) like BPS expects ---
                    try:
                        if not isinstance(df.index, pd.MultiIndex):
                            # if df has DatetimeIndex or an index of timestamps, convert
                            if isinstance(df.index, (pd.DatetimeIndex, np.ndarray, list)):
                                tuples = [(symbol, pd.to_datetime(idx)) for idx in df.index]
                                df.index = pd.MultiIndex.from_tuples(tuples, names=["symbol", "timestamp"])
                            else:
                                # fallback: if df empty or unknown structure, create fresh frame with standard cols
                                df = pd.DataFrame(
                                    columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])
                    except Exception:
                        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])

                    # --- build new minute row from incoming bar object ---
                    new_data = [
                        getattr(data, "open", None),
                        getattr(data, "high", None),
                        getattr(data, "low", None),
                        getattr(data, "close", None),
                        getattr(data, "volume", None),
                        getattr(data, "trade_count", None),
                        getattr(data, "vwap", None),
                    ]
                    cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
                    index_key = (symbol, ts)
                    new_row = pd.DataFrame([new_data],
                                           index=pd.MultiIndex.from_tuples([index_key], names=["symbol", "timestamp"]),
                                           columns=cols)

                    # append / update
                    if index_key not in df.index:
                        df = pd.concat([df, new_row])
                    else:
                        # update existing row
                        df.loc[index_key] = new_row.iloc[0]

                    # keep window reasonable (tail)
                    df = df[~df.index.duplicated(keep="last")].tail(2000)
                    self.minute_history[symbol] = df

                    # --- iterate over pairs and evaluate signals ---
                    for sym_x, sym_y in list(self.pairs):
                        try:
                            # ensure both histories exist
                            if sym_x not in self.minute_history or sym_y not in self.minute_history:
                                continue

                            # extract close series (handle if index is multiindex)
                            try:
                                sx = self.minute_history[sym_x]["close"].rename(sym_x)
                            except Exception:
                                # if minute_history[sym_x] has a MultiIndex where first level is symbol, select rows
                                try:
                                    df_x = self.minute_history[sym_x]
                                    if isinstance(df_x.index, pd.MultiIndex):
                                        sx = df_x.xs(sym_x, level=0)["close"].rename(sym_x)
                                    else:
                                        continue
                                except Exception:
                                    continue

                            try:
                                sy = self.minute_history[sym_y]["close"].rename(sym_y)
                            except Exception:
                                try:
                                    df_y = self.minute_history[sym_y]
                                    if isinstance(df_y.index, pd.MultiIndex):
                                        sy = df_y.xs(sym_y, level=0)["close"].rename(sym_y)
                                    else:
                                        continue
                                except Exception:
                                    continue

                            # align timestamps with outer join and forward-fill, then drop NA
                            merged = pd.concat([sx, sy], axis=1, join="outer").sort_index().ffill().dropna()
                            if merged.empty or len(merged) < self.min_history_bars:
                                continue

                            # throttle screening per-pair: ensure last_check is a datetime before subtracting
                            last_check = self.last_screening_minute.get((sym_x, sym_y))
                            if last_check is not None:
                                # ensure last_check is datetime
                                if isinstance(last_check, str):
                                    try:
                                        last_check = datetime.fromisoformat(last_check)
                                    except Exception:
                                        last_check = pd.to_datetime(last_check).to_pydatetime()
                                if (ts - last_check).total_seconds() < 60:
                                    continue

                            # record screening time as datetime
                            self.last_screening_minute[(sym_x, sym_y)] = ts

                            # generate signal (no-lookahead)
                            series_x = merged[sym_x]
                            series_y = merged[sym_y]
                            signal, meta = self.generate_pair_signal(series_x, series_y)

                            # check if this pair is active in either orientation
                            active = None
                            for ap in (self.active_pairs or []):
                                if (ap.get("symbol_long") == sym_x and ap.get("symbol_short") == sym_y) or \
                                        (ap.get("symbol_long") == sym_y and ap.get("symbol_short") == sym_x):
                                    active = ap
                                    break

                            # === ENTRY behavior ===
                            if market_open + timedelta(minutes=15) <= ts < market_close - timedelta(minutes=15):
                                if active is None and signal in (1, -1) and meta is not None:
                                    beta = meta.get("beta", None)
                                    resid = float(meta.get("resid", 0.0))
                                    threshold = float(meta.get("threshold", 0.0))
                                    px_x = float(series_x.iloc[-1])
                                    px_y = float(series_y.iloc[-1])
                                    hedge_slope = float(beta["slope"]) if beta is not None and "slope" in beta else 1.0

                                    # compute leg sizes using the new sizing function
                                    qty_x, qty_y, limit_price = self.compute_leg_qtys(px_x, px_y, hedge_slope, side=signal)
                                    # prefer symmetric minimal quantity to preserve hedge
                                    qty = int(max(0, min(qty_x, qty_y)))
                                    if qty <= 0:
                                        self.logger.info(
                                            f"[SKIP SIZE] qty=0 for {sym_x}/{sym_y} (px {px_x}/{px_y}, slope {hedge_slope})")
                                        continue

                                    try:
                                        # use safer limit-order flow with auto-cancel
                                        if signal == 1:
                                            # LONG X, SHORT Y -> pass long symbol first
                                            order_long, order_short = self.submit_pair_open_limit(
                                                symbol_long=sym_x,
                                                symbol_short=sym_y,
                                                qty_long=qty,
                                                qty_short=qty,
                                                price_long=None,
                                                price_short=None,
                                                timeout_seconds=getattr(self, "order_timeout_seconds", 180)
                                            )
                                            if order_long or order_short:
                                                entry = {
                                                    "symbol_long": sym_x,
                                                    "symbol_short": sym_y,
                                                    "qty": qty,
                                                    "side": 1,
                                                    "entered_at": ts.isoformat(),
                                                    "entry_px_x": px_x,
                                                    "entry_px_y": px_y,
                                                    "beta": hedge_slope,
                                                    "resid_entry": resid,
                                                    "threshold_entry": threshold,
                                                    "order_ids": (
                                                        getattr(order_long, "id", None),
                                                        getattr(order_short, "id", None)
                                                    )
                                                }
                                                self.active_pairs.append(entry)
                                                self.save_pair_state(self.active_pairs)
                                                self.logger.info(
                                                    f"[OPEN] {sym_x}/{sym_y} LONGX-SHORTY qty={qty} resid={resid:.6f}")
                                        else:
                                            # SHORT X, LONG Y -> pass long symbol first (sym_y)
                                            order_long, order_short = self.submit_pair_open_limit(
                                                symbol_long=sym_y,
                                                symbol_short=sym_x,
                                                qty_long=qty_y,
                                                qty_short=qty_x,
                                                price_long=None,
                                                price_short=None,
                                                timeout_seconds=getattr(self, "order_timeout_seconds", 180)
                                            )
                                            if order_long or order_short:
                                                entry = {
                                                    "symbol_long": sym_y,
                                                    "symbol_short": sym_x,
                                                    "qty": qty,
                                                    "side": -1,
                                                    "entered_at": ts.isoformat(),
                                                    "entry_px_x": px_x,
                                                    "entry_px_y": px_y,
                                                    "beta": hedge_slope,
                                                    "resid_entry": resid,
                                                    "threshold_entry": threshold,
                                                    "order_ids": (
                                                        getattr(order_long, "id", None),
                                                        getattr(order_short, "id", None)
                                                    )
                                                }
                                                self.active_pairs.append(entry)
                                                self.save_pair_state(self.active_pairs)
                                                self.logger.info(
                                                    f"[OPEN] {sym_y}/{sym_x} LONGY-SHORTX qty={qty} resid={resid:.6f}")

                                    except Exception as e:
                                        self.logger.error(f"[ENTRY ERROR] Failed to open {sym_x}/{sym_y}: {e}",
                                                          exc_info=True)

                            # === EXIT / FLIP behavior ===
                            if active is not None:
                                try:
                                    # recompute full (no-shift) Kalman regression to evaluate exit/flip
                                    full = self.kalman_regression_pair(series_x.values.astype(float),
                                                                       series_y.values.astype(float))
                                    e_latest = float(full["e"][-1])
                                    Q_latest = float(full["Q"][-1])
                                    threshold_latest = np.sqrt(max(Q_latest, 1e-12)) * self.threshold_z

                                    # deduce active orientation: if active.long == sym_x then side=1 else -1
                                    active_side = 1 if (active.get("symbol_long") == sym_x and active.get(
                                        "symbol_short") == sym_y) else -1

                                    # Exit when residual crosses zero (mean reverted)
                                    if (active_side == 1 and e_latest >= 0) or (active_side == -1 and e_latest <= 0):
                                        self.logger.info(
                                            f"[EXIT] {active['symbol_long']}/{active['symbol_short']} residual crossed zero (e={e_latest:.6f}) -> closing")
                                        q_close = int(active.get("qty", 0))
                                        if q_close > 0:
                                            self.submit_pair_close(active["symbol_long"], active["symbol_short"],
                                                                   q_close)
                                        # remove active
                                        self.active_pairs = [ap for ap in (self.active_pairs or []) if not (
                                                ap.get("symbol_long") == active.get("symbol_long") and ap.get(
                                            "symbol_short") == active.get("symbol_short")
                                        )]
                                        self.save_pair_state(self.active_pairs)

                                    # Flip: opposite strong signal (force close, allow new entry later)
                                    elif (active_side == 1 and e_latest > threshold_latest) or (
                                            active_side == -1 and e_latest < -threshold_latest):
                                        self.logger.info(
                                            f"[FLIP] {active['symbol_long']}/{active['symbol_short']} flipping (e={e_latest:.6f} th={threshold_latest:.6f})")
                                        q_close = int(active.get("qty", 0))
                                        if q_close > 0:
                                            self.submit_pair_close(active["symbol_long"], active["symbol_short"],
                                                                   q_close)
                                        # remove active pair state (we allow new opposite-side entry later)
                                        self.active_pairs = [ap for ap in (self.active_pairs or []) if not (
                                                ap.get("symbol_long") == active.get("symbol_long") and ap.get(
                                            "symbol_short") == active.get("symbol_short")
                                        )]
                                        self.save_pair_state(self.active_pairs)

                                except Exception as e:
                                    self.logger.error(f"[EXIT CHECK ERROR] {e}", exc_info=True)

                        except Exception as e:
                            # per-pair error (continue scanning other pairs)
                            self.logger.error(f"[PAIR LOOP ERROR] {sym_x}/{sym_y}: {e}", exc_info=True)

                    # --- portfolio-level Friday near-close exit (mirror BPS) ---
                    try:
                        if ts.weekday() == 4 and ts >= (market_close - timedelta(hours=1)):
                            if self.active_pairs:
                                self.logger.info("[PORTFOLIO EXIT] Friday near close — closing all active pairs")
                                for ap in list(self.active_pairs):
                                    q_close = int(ap.get("qty", 0))
                                    if q_close > 0:
                                        self.submit_pair_close(ap["symbol_long"], ap["symbol_short"], q_close)
                                    self.active_pairs.remove(ap)
                                self.save_pair_state(self.active_pairs)
                    except Exception:
                        pass

                except Exception as e:
                    self.logger.error(f"[TRACEBACK] Exception in handle_second_bar: {e}\n{traceback.format_exc()}")

            # subscribe to bars for all symbols
            try:
                self.wss_client.subscribe_bars(handle_second_bar, *symbols)
                self.wss_client.run()
            except Exception as e:
                self.logger.error(f"[WSS ERROR] {e}", exc_info=True)

        # start threads
        trading_thread = threading.Thread(target=run_trading_stream)
        wss_thread = threading.Thread(target=run_wss_client)

        trading_thread.start()
        wss_thread.start()

        trading_thread.join()
        wss_thread.join()

if __name__ == "__main__":

    # Instantiate your strategy class
    strategy = KalmanFilterStrategy()

    # Get today's date in New York timezone
    nyc = ZoneInfo('America/New_York')
    today = datetime.today().astimezone(nyc)
    today_str = today.strftime('%Y-%m-%d')

    # Fetch market calendar using Alpaca
    try:
        calendar_request = GetCalendarRequest(start=today_str, end=today_str)
        calendar_data = strategy.trading_client.get_calendar(calendar_request)

        if not calendar_data:
            logger.info("Market is closed today. Exiting.")
            exit()

        calendar = calendar_data[0]
    except Exception as e:
        logger.error(f"Error fetching market calendar: {e}")
        exit()

    # Get actual open and close datetime objects
    market_open = today.replace(
        hour=calendar.open.hour,
        minute=calendar.open.minute,
        second=0,
        microsecond=0
    ).astimezone(nyc)

    market_close = today.replace(
        hour=calendar.close.hour,
        minute=calendar.close.minute,
        second=0,
        microsecond=0
    ).astimezone(nyc)

    # Wait until 15 minutes after market open before starting
    while (datetime.now(nyc) - market_open).seconds // 60 <= 14:
        time.sleep(1)

    # Run the strategy
    strategy.run(market_open, market_close)

