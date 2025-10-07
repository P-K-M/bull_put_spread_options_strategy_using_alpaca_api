import pandas as pd
import numpy as np
from scipy.stats import norm
from ta.trend import MACD, ADXIndicator
import time
import os, json

from scipy.optimize import brentq
from datetime import date, datetime, timedelta, timezone

from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple
import re
from collections import defaultdict

import yfinance as yf

from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
    GetAssetsRequest,
    LimitOrderRequest,
    GetCalendarRequest,
    OptionLegRequest
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

import threading

# Configure logging
import logging

import traceback

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

TRADING_CONFIG = {
    'RISK_FREE_RATE': 0.0389,  # 4.5% annual
    'TRADING_YEAR_SIZE': 252,
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'ADX_PERIOD': 14,
    'SMA_SHORT': 20,
    'SMA_LONG': 50
}


class IronCondorStrategy:
    def __init__(self):
        # === Alpaca API Clients ===
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
        self.trading_stream = TradingStream(API_KEY, SECRET_KEY, paper=PAPER)
        self.wss_client = StockDataStream(API_KEY, SECRET_KEY)
        self.option_historical_data_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
        self.stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        # === Trading Configuration ===
        self.timezone = ZoneInfo('America/New_York')
        self.today = datetime.now(self.timezone).date()
        self.logger = logger
        self.logger.debug("IronCondorStrategy initialized.")

        self.underlying_assets = [
            "SPY", "QQQ", "IWM", "DIA",  # Major Index ETFs
            "XLK", "XLF", "XLV", "XLE"  # Sectors
        ]

        # === Strategy Parameters ===
        self.strike_range = 0.05  # ±5% OTM on both sides
        self.buy_power_limit = 0.20
        self.risk_free_rate = self.get_risk_free_rate()
        self.target_profit_percent = 0.25
        self.delta_stop_loss = 0.40  # for both short put & short call
        self.vega_stop_loss_percent = 0.20

        # === Buying Power ===
        self.buying_power = float(self.trading_client.get_account().buying_power)
        self.buying_power_limit = self.buying_power * self.buy_power_limit

        # === Strict Criteria (updated) ===
        self.criteria = {
            'short_put': {'min_delta': 0.12, 'max_delta': 0.40, 'min_iv': 0.08, 'max_iv': 3.0,
                          'min_oi': 10, 'max_bid_ask_spread': 0.75},
            'long_put': {'min_delta': 0.01, 'max_delta': 0.25, 'min_iv': 0.02, 'max_iv': 3.0,
                         'min_oi': 5, 'max_bid_ask_spread': 1.00},
            'short_call': {'min_delta': 0.12, 'max_delta': 0.40, 'min_iv': 0.08, 'max_iv': 3.0,
                           'min_oi': 10, 'max_bid_ask_spread': 0.75},
            'long_call': {'min_delta': 0.01, 'max_delta': 0.25, 'min_iv': 0.02, 'max_iv': 3.0,
                          'min_oi': 5, 'max_bid_ask_spread': 1.00},

            'min_days_to_exp': 4,  # allow 4 DTE in strict if necessary
            'max_days_to_exp': 9,  # widen strict a bit
            'max_spread_width': 2.0,
            'max_otm_distance': 20.0,
            'min_credit_to_risk': 0.10,  # strict CRR moved to 10%
        }

        # === Relaxed Criteria (updated) ===
        self.relaxed_criteria = {
            'short_put': {'min_delta': 0.08, 'max_delta': 0.45, 'min_iv': 0.03, 'max_iv': 4.0,
                          'min_oi': 5, 'max_bid_ask_spread': 1.50},
            'long_put': {'min_delta': 0.01, 'max_delta': 0.30, 'min_iv': 0.01, 'max_iv': 4.0,
                         'min_oi': 0, 'max_bid_ask_spread': 2.00},
            'short_call': {'min_delta': 0.08, 'max_delta': 0.45, 'min_iv': 0.03, 'max_iv': 4.0,
                           'min_oi': 5, 'max_bid_ask_spread': 1.50},
            'long_call': {'min_delta': 0.01, 'max_delta': 0.30, 'min_iv': 0.01, 'max_iv': 4.0,
                          'min_oi': 0, 'max_bid_ask_spread': 2.00},

            'min_days_to_exp': 4,
            'max_days_to_exp': 12,
            'max_spread_width': 2.0,
            'max_otm_distance': 25.0,
            'min_credit_to_risk': 0.05,  # relaxed CRR down to 5%
        }

        # === Internal State ===
        self.minute_history: Dict[str, pd.DataFrame] = {}
        self.active_spreads = []
        self.rejection_reasons = defaultdict(int)
        self.last_traded: Dict[str, Any] = {}  # track last traded condors per symbol
        self.rejection_table: List[Dict[str, Any]] = []  # store all rejection reasons

        # === Time-to-Expiration Logic ===
        self.min_dte = 6
        self.max_dte = 9
        self.holding_period_days = 5

        self.logger.debug(
            f"IronCondorStrategy initialized on {self.today} with buying power ${self.buying_power:,.2f}.")

    def get_risk_free_rate(self) -> float:
        """
        Fetches the current risk-free rate from the 10-year Treasury Note (symbol: ^TNX).
        If unavailable, returns the default configured rate.

        Returns:
            float: Annualized risk-free interest rate (e.g., 0.045 for 4.5%)
        """
        try:
            t_bill = yf.Ticker("^TNX")
            data = t_bill.history(period="1d")
            if not data.empty:
                rate = data['Close'].iloc[-1] / 100
                self.logger.debug(f"Fetched risk-free rate from ^TNX: {rate:.4f}")
                return rate
            else:
                raise ValueError("Empty data returned from ^TNX")
        except Exception as e:
            self.logger.warning(
                f"Failed to fetch risk-free rate from ^TNX. Using fallback: {TRADING_CONFIG['RISK_FREE_RATE']} | Error: {e}")
            return TRADING_CONFIG['RISK_FREE_RATE']

    def get_expiry_within_range(self, expiries, min_dte=5, max_dte=7, today=None):
        """
        Select an expiry within [min_dte, max_dte] trading days.
        Returns the nearest valid expiry, or None if none are found.
        """

        if today is None:
            today = datetime.now(tz=ZoneInfo("America/New_York")).date()

        valid_expiries = []
        for exp in expiries:
            exp_date = pd.to_datetime(exp).date()
            dte = np.busday_count(today, exp_date)  # trading days only

            if min_dte <= dte <= max_dte:
                valid_expiries.append((dte, exp_date))

        if not valid_expiries:
            logging.warning(f"[EXPIRY] No valid expiries in {min_dte}-{max_dte} DTE range.")
            return None

        # Pick the nearest expiry (lowest DTE)
        chosen = min(valid_expiries, key=lambda x: x[0])
        logging.info(f"[EXPIRY] Selected expiry {chosen[1]} with {chosen[0]} DTE.")
        return chosen[1]

    def get_1000m_history_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
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

    # ==============================================================================
    # Implied Volatility, Greeks, and Technical Indicators
    # ==============================================================================

    def calculate_implied_volatility(self, option_price, S, K, T, r, option_type):
        """
        Estimate the implied volatility using the Black-Scholes model via Brent's method.
        Handles options close to intrinsic value gracefully.
        """
        sigma_lower, sigma_upper = 1e-6, 5.0
        intrinsic_value = max(0, (S - K) if option_type == 'call' else (K - S))
        extrinsic_value = option_price - intrinsic_value

        if extrinsic_value < 1e-3:
            self.logger.debug("[IV] Option has almost no extrinsic value. Likely illiquid or deep ITM/OTM.")
            return 0.01

        def option_price_diff(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return price - option_price

        try:
            iv = brentq(option_price_diff, sigma_lower, sigma_upper)
            self.logger.debug(f"[IV] Calibrated implied volatility: {iv:.4f}")
            return iv
        except ValueError as e:
            self.logger.warning(f"[IV] Brent solver failed: {e}. Falling back to 0.20.")
            return 0.20

    def calculate_greeks(self, option_price, strike_price, expiration, underlying_price,
                         risk_free_rate, option_type, IV=None):
        """
        Calculate Black-Scholes Greeks (Delta, Gamma, Theta, Vega) with fallback handling.
        """
        expiration = pd.Timestamp(expiration, tz='UTC') if expiration.tzinfo is None else expiration.tz_convert(
            'UTC')
        now = pd.Timestamp.now(tz='UTC')

        time_to_expiry_secs = (expiration - now).total_seconds()

        if time_to_expiry_secs <= 0:
            self.logger.warning("Option already expired — Greeks defaulted based on intrinsic value.")
            delta = -1.0 if (underlying_price < strike_price and option_type == 'put') else \
                1.0 if (underlying_price > strike_price and option_type == 'call') else 0.0
            return delta, 0.0, 0.0, 0.0

        # Approximate time to expiry in trading years
        T = time_to_expiry_secs / (252 * 24 * 60 * 60)

        if IV is None or IV <= 0.0:
            self.logger.debug("[GREEKS] IV missing or zero. Recalculating or defaulting to 0.20.")
            IV = self.calculate_implied_volatility(option_price, underlying_price, strike_price, T, risk_free_rate,
                                                   option_type)

        if IV <= 0 or underlying_price <= 0 or strike_price <= 0:
            self.logger.warning(f"[GREEKS] Invalid input: IV={IV}, S={underlying_price}, K={strike_price}")
            return 0.0, 0.0, 0.0, 0.0

        d1 = (np.log(underlying_price / strike_price) + (risk_free_rate + 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))
        d2 = d1 - IV * np.sqrt(T)

        delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (underlying_price * IV * np.sqrt(T))
        vega = (underlying_price * norm.pdf(d1) * np.sqrt(T)) / 100  # per 1% IV
        theta = (
                        - (underlying_price * norm.pdf(d1) * IV) / (2 * np.sqrt(T)) -
                        risk_free_rate * strike_price * np.exp(-risk_free_rate * T) *
                        (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))
                ) / 252

        return delta, gamma, theta, vega

    def calculate_rsi(self, series, period=14):
        """
        Calculate Relative Strength Index (RSI) using exponential smoothing.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-6)  # Prevent division by zero
        return 100 - (100 / (1 + rs))

    def calculate_moving_averages(self, df, short=20, long=50):
        """
        Add simple moving averages to price dataframe.
        """
        df = df.copy()
        df[f'{short}_MA'] = df['close'].rolling(window=short).mean()
        df[f'{long}_MA'] = df['close'].rolling(window=long).mean()
        return df

    def calculate_macd_features(self, df, fast=12, slow=26, signal=9):
        """
        Calculate MACD indicators and append to DataFrame.
        """
        df = df.copy()
        macd = MACD(close=df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        return df

    def calculate_adx_features(self, df, period=14):
        """
        Add Average Directional Index (ADX) to evaluate trend strength.
        """
        df = df.copy()
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=period)
        df['ADX'] = adx.adx()
        return df

    def add_all_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and append all relevant technical indicators for trade evaluation.
        Includes: MA(20, 50), MACD, ADX, RSI, Volume SMA(20)
        """
        df = self.calculate_moving_averages(dataframe)
        df = self.calculate_macd_features(df)
        df = self.calculate_adx_features(df)
        df['RSI'] = self.calculate_rsi(df['close'])
        df['Volume_SMA20'] = df['volume'].rolling(window=20).mean()
        return df

    def get_underlying_assets(self) -> List[str]:
        """
        Select underlying assets for Iron Condor strategy using neutral/range-bound criteria:
        - Primary: Liquidity + Range-bound price action + IV Rank
        - Exclude: Earnings/events within N days
        - Always include: SPY, QQQ, IWM, DIA
        """
        self.logger.info("[SCREEN] Starting Iron Condor asset selection...")

        # === Step 1: Get tradable assets (filter only predefined watchlist) ===
        assets = self.trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        symbols = [a.symbol for a in assets if a.tradable and a.symbol in self.underlying_assets]

        # Historical data
        start_dt = (datetime.now(self.timezone) - timedelta(days=180)).date()
        end_dt = datetime.now(self.timezone).date()

        bars = self.stock_data_client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_dt, end=end_dt)
        ).df.copy()

        if isinstance(bars.index, pd.MultiIndex):
            bars.reset_index(inplace=True)

        # Add indicators
        bars = bars.groupby('symbol', group_keys=False).apply(lambda df: self.add_all_features(df.copy()))
        bars = bars.reset_index()

        selected: List[Tuple[str, float]] = []

        for symbol in symbols:
            df = bars[bars['symbol'] == symbol].copy().set_index('timestamp')
            if len(df) < 50:
                self.logger.debug(f"[FILTER] {symbol}: Skipped (insufficient history)")
                continue

            try:
                latest = df.iloc[-1]
                score = 0
                reasons = []

                # === Neutrality / Range-bound filters ===
                if 40 <= latest['RSI'] <= 60:
                    score += 1
                else:
                    reasons.append("rsi_out_of_range")

                if latest['ADX'] < 20:
                    score += 1
                else:
                    reasons.append("strong_trend")

                if abs((latest['close'] - latest['50_MA']) / latest['50_MA']) < 0.05:
                    score += 1
                else:
                    reasons.append("too_far_from_ma")

                if latest['volume'] > latest['Volume_SMA20']:
                    score += 1
                else:
                    reasons.append("low_volume")

                # === IV Rank / IV filter ===
                option_price = latest['close'] * 0.05
                iv = self.calculate_implied_volatility(
                    option_price, latest['close'], latest['close'], 30 / 252,
                    self.risk_free_rate, 'call'
                )

                iv_value = float(iv) if iv is not None else 0.0
                iv_display = f"{iv_value:.3f}"

                if iv_value > 0.20:
                    score += 1
                else:
                    reasons.append("iv_too_low")

                # === Final selection ===
                if score >= 3:
                    selected.append((symbol, iv_value))
                    self.logger.info(
                        f"[SELECTED] {symbol} | Score={score}/5 | IV={iv_display} | "
                        f"Close={latest['close']:.2f}, RSI={latest['RSI']:.1f}, ADX={latest['ADX']:.1f}"
                    )
                else:
                    self.logger.debug(f"[REJECTED] {symbol} | Score={score}/5 | Reasons: {', '.join(reasons)}")

            except Exception as e:
                self.logger.warning(f"[ERROR] Screening failed for {symbol}: {e}")
                continue

        selected_symbols = [s for s, _ in sorted(selected, key=lambda x: x[1], reverse=True)]

        # === Always include baseline ETFs ===
        baseline_etfs = ["SPY", "QQQ", "IWM", "DIA"]
        for etf in baseline_etfs:
            if etf not in selected_symbols and etf in self.underlying_assets:
                self.logger.info(f"[FORCED INCLUDE] {etf} (baseline index ETF)")
                selected_symbols.append(etf)

        return selected_symbols

    """
        Handles retrieving, validating, pricing, and analyzing put option chains for index ETFs.
        """

    cache = {}  # Avoid repeated API calls for same symbol

    # ------------------------------------------------------------------------------
    # Retrieve option chains from Alpaca's API
    # ------------------------------------------------------------------------------
    def get_options(
            self,
            symbol: str,
            min_strike: float,
            max_strike: float,
            min_exp: date,
            max_exp: date,
            option_type: ContractType = ContractType.PUT
    ):
        """
        Fetch options within strike and expiration bounds for a given symbol.
        """
        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status=AssetStatus.ACTIVE,
                type=option_type,
                strike_price_gte=str(round(min_strike, 2)),
                strike_price_lte=str(round(max_strike, 2)),
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
            )
            contracts = self.trading_client.get_option_contracts(req).option_contracts
            self.logger.debug(f"[CHAIN] Retrieved {len(contracts)} {option_type.value} contracts for {symbol}")
            return contracts

        except Exception as e:
            self.logger.error(f"[CHAIN ERROR] {symbol}: Failed to fetch option contracts - {e}")
            return []

    # ------------------------------------------------------------------------------
    # Helpers for validating and converting option objects
    # ------------------------------------------------------------------------------
    def validate_sufficient_OI(self, option_data: dict, leg_type: str, criteria: dict) -> bool:
        """
        Check if the option contract has sufficient open interest and a valid open interest date,
        using the correct threshold from the criteria for either the short or long leg.

        Args:
            option_data (dict): The option contract data.
            leg_type (str): Either 'short_put' or 'long_put'.
            criteria (dict): Either self.criteria or self.relaxed_criteria.

        Returns:
            bool: True if OI is sufficient and OI date is valid.
        """
        open_interest = int(option_data.get('open_interest') or 0)
        open_interest_date = option_data.get('open_interest_date')
        min_oi = criteria.get(leg_type, {}).get('min_oi', 0)

        return bool(
            open_interest_date and
            open_interest >= min_oi
        )

    def ensure_dict(self, option_data) -> dict:
        """
        Convert option data to dictionary if Pydantic-style model.
        """
        return option_data.model_dump() if hasattr(option_data, "model_dump") else option_data

    # ------------------------------------------------------------------------------
    # Pricing & Greeks Calculation
    # ------------------------------------------------------------------------------

    def calculate_option_metrics(self, option_data: dict, underlying_price: float, risk_free_rate: float) -> dict:
        """
        Returns a dict of quotes, greeks, IV, and derived convenience fields.
        More robust handling of missing quotes & business-day time to expiry.
        """
        symbol = option_data.get('symbol', 'unknown')
        expiration = pd.Timestamp(option_data.get('expiration_date'))
        now_ts = pd.Timestamp.now(tz=expiration.tz) if hasattr(expiration, 'tz') else pd.Timestamp.now()

        # --- Quotes (defensive) ---
        try:
            quote_req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.option_historical_data_client.get_option_latest_quote(quote_req)[symbol]
            bid = self.safe_float(quote.bid_price)
            ask = self.safe_float(quote.ask_price)
        except Exception as e:
            self.logger.debug(f"[QUOTE ERROR] {symbol}: {e}")
            bid, ask = 0.0, 0.0

        # Derived quote metrics
        bid_ask_spread = max(0.0, ask - bid)
        if bid <= 0 and ask <= 0:
            mid_price = 0.0
        elif bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2.0
        elif bid > 0:
            mid_price = bid
        else:
            mid_price = ask

        # Days to expiry (calendar + trading)
        try:
            cal_days = max((expiration.normalize() - now_ts.normalize()).days, 0)
            try:
                bus_days = np.busday_count(
                    np.datetime64(now_ts.date()),
                    np.datetime64(expiration.date())
                )
            except Exception:
                bus_days = max(cal_days, 0)
            days_to_expiry = max(int(cal_days), 0)
            trading_days_to_expiry = max(int(bus_days), 1)
            time_to_expiry = trading_days_to_expiry / 252.0
        except Exception:
            days_to_expiry = max((expiration - now_ts).days, 0)
            trading_days_to_expiry = max(days_to_expiry, 1)
            time_to_expiry = trading_days_to_expiry / 252.0

        # Normalize option type
        raw_type = str(option_data.get("type", "")).lower()
        if raw_type in ["c", "call"]:
            option_type = "call"
        elif raw_type in ["p", "put"]:
            option_type = "put"
        else:
            option_type = None

        # Implied vol
        iv = None
        if mid_price > 0:
            try:
                iv = self.calculate_implied_volatility(
                    option_price=mid_price,
                    S=float(underlying_price),
                    K=float(option_data.get('strike_price')),
                    T=float(time_to_expiry),
                    r=float(risk_free_rate or 0.0),
                    option_type=option_type
                )
                if not (iv and iv > 0):
                    raise ValueError("non-positive IV returned")
            except Exception as e:
                self.logger.debug(f"[IV WARN] {symbol}: {e} -> using fallback IV=0.2")
                iv = 0.2
        else:
            iv = 0.2

        # Greeks
        try:
            delta, gamma, theta, vega = self.calculate_greeks(
                option_price=mid_price,
                strike_price=float(option_data.get('strike_price')),
                expiration=expiration,
                underlying_price=float(underlying_price),
                risk_free_rate=float(risk_free_rate or 0.0),
                option_type=option_type,
                IV=iv
            )
        except Exception as e:
            self.logger.debug(f"[GREEKS WARN] {symbol}: {e} -> zeros")
            delta, gamma, theta, vega = 0.0, 0.0, 0.0, 0.0

        # Convenience numbers
        strike_price = self.safe_float(option_data.get('strike_price'))
        strike_distance = strike_price - float(underlying_price) if underlying_price is not None else None
        abs_strike_distance = abs(strike_distance) if strike_distance is not None else None
        pct_from_underlying = (abs_strike_distance / float(
            underlying_price)) if underlying_price and abs_strike_distance is not None else None
        pct_spread = (bid_ask_spread / mid_price) if mid_price > 0 else None
        liquid = (bid > 0 and ask > 0 and mid_price > 0 and bid_ask_spread >= 0)

        return {
            "expiration_date": expiration,
            "calendar_days": days_to_expiry,
            "trading_days": trading_days_to_expiry,
            "time_to_expiry": time_to_expiry,
            "bid_price": bid,
            "ask_price": ask,
            "mid_price": mid_price,
            "bid_ask_spread": bid_ask_spread,
            "pct_spread": pct_spread,
            "option_price": mid_price,
            "implied_volatility": iv,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "strike_distance": strike_distance,
            "abs_strike_distance": abs_strike_distance,
            "pct_from_underlying": pct_from_underlying,
            "liquid": liquid,
        }

    # ------------------------------------------------------------------------------
    # Unified Option Dictionary Constructor
    # ------------------------------------------------------------------------------

    def safe_float(self, val, default=0.0):
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def build_option_dict(self, option_data, underlying_price: float, risk_free_rate: float) -> dict:
        """
        Unified option dict for downstream filtering.
        Adds derived fields helpful when building iron condors.
        """
        option_data = self.ensure_dict(option_data)
        symbol = option_data.get("symbol", "unknown")

        metrics = self.calculate_option_metrics(option_data, underlying_price, risk_free_rate)

        try:
            open_interest = int(option_data.get('open_interest') or 0)
        except (TypeError, ValueError):
            self.logger.debug(f"[OI] Invalid open_interest for {symbol}. Defaulting to 0.")
            open_interest = 0

        # Inline normalization (consistent with calculate_option_metrics)
        opt_type_raw = str(option_data.get('type') or "").lower()
        opt_type_norm = "call" if "call" in opt_type_raw else "put" if "put" in opt_type_raw else "unknown"
        is_call = "call" in opt_type_norm
        is_put = "put" in opt_type_norm

        candidate = {
            'id': option_data.get('id'),
            'name': option_data.get('name'),
            'symbol': symbol,
            'strike_price': self.safe_float(option_data.get('strike_price')),
            'root_symbol': option_data.get('root_symbol'),
            'underlying_symbol': option_data.get('underlying_symbol'),
            'underlying_asset_id': option_data.get('underlying_asset_id'),
            'close_price': self.safe_float(option_data.get('close_price')),
            'close_price_date': option_data.get('close_price_date'),
            'expiration_date': metrics['expiration_date'],
            'remaining_days': metrics['trading_days'],
            'calendar_days': metrics['calendar_days'],
            'open_interest': open_interest,
            'open_interest_date': option_data.get('open_interest_date'),
            'size': int(option_data.get('size') or 1),
            'status': option_data.get('status'),
            'style': option_data.get('style'),
            'tradable': bool(option_data.get('tradable')),
            'type': opt_type_norm,
            'is_call': is_call,
            'is_put': is_put,
            # Use consistent bid/ask/mid from metrics
            'bid_price': metrics['bid_price'],
            'ask_price': metrics['ask_price'],
            'mid_price': metrics['mid_price'],
            'bid_ask_spread': metrics['bid_ask_spread'],
            'pct_spread': metrics['pct_spread'],
            'implied_volatility': metrics['implied_volatility'],
            'delta': metrics['delta'],
            'gamma': metrics['gamma'],
            'theta': metrics['theta'],
            'vega': metrics['vega'],
            'option_price': metrics['option_price'],
            'initial_option_price': metrics['option_price'],
            'strike_distance': metrics['strike_distance'],
            'abs_strike_distance': metrics['abs_strike_distance'],
            'pct_from_underlying': metrics['pct_from_underlying'],
            'liquid': metrics['liquid'],
        }
        return candidate

    def check_option_conditions(self, candidate: dict, label: str, criteria: dict) -> bool:
        """
        Leg-aware option validator with detailed rejection logging.
        - Global expiry rules checked at top-level, but also validated here for consistency
        - Label lookup is case-insensitive (accepts 'SHORT_PUT' or 'short_put')
        - Mid = candidate.mid if >0 else (bid+ask)/2
        - All rejections flow into self.rejection_reasons
        """
        # normalize label
        label_norm = (label or "").upper()
        leg_map = {
            "SHORT_PUT": "short_put",
            "LONG_PUT": "long_put",
            "SHORT_CALL": "short_call",
            "LONG_CALL": "long_call",
        }
        leg_key = leg_map.get(label_norm) or leg_map.get(label_norm.lower())
        if not leg_key:
            self.logger.error(f"[CRITERIA ERROR] Invalid label '{label}'")
            return False

        leg_criteria = criteria.get(leg_key, {})
        symbol = candidate.get("symbol", "unknown")

        # --- safe casts ---
        bid = float(candidate.get("bid_price") or 0.0)
        ask = float(candidate.get("ask_price") or 0.0)
        candidate_mid = float(candidate.get("mid_price") or 0.0)
        mid = candidate_mid if candidate_mid > 0 else ((bid + ask) / 2.0 if (bid > 0 or ask > 0) else 0.0)

        spread = float(candidate.get("bid_ask_spread") or max(0.0, ask - bid))
        pct_spread = candidate.get("pct_spread")
        delta = candidate.get("delta")
        delta = float(delta) if delta not in (None, "") else None
        abs_delta = abs(delta) if delta is not None else None
        iv = float(candidate.get("implied_volatility") or 0.0)
        oi = int(candidate.get("open_interest") or 0)
        dte = int(candidate.get("remaining_days") or 0)

        # --- Liquidity ---
        if bid <= 0 or ask <= 0 or mid <= 0:
            self.logger.info(f"[REJECT] {label} {symbol} illiquid (bid={bid:.2f}, ask={ask:.2f}, mid={mid:.2f})")
            self.rejection_reasons[f"{label}_illiquid"] += 1
            return False

        # --- Spread checks ---
        max_bid_ask_spread = float(leg_criteria.get("max_bid_ask_spread", 0.5))
        if spread > max_bid_ask_spread:
            self.logger.info(f"[REJECT] {label} {symbol} spread {spread:.2f} > max {max_bid_ask_spread:.2f}")
            self.rejection_reasons[f"{label}_spread"] += 1
            return False
        max_pct_spread = leg_criteria.get("max_pct_spread")
        if max_pct_spread is not None and pct_spread is not None and pct_spread > float(max_pct_spread):
            self.logger.info(
                f"[REJECT] {label} {symbol} pct_spread {pct_spread:.2f} > max_pct_spread {max_pct_spread:.2f}")
            self.rejection_reasons[f"{label}_pct_spread"] += 1
            return False

        # --- Mid price floor ---
        min_mid = float(leg_criteria.get("min_mid_price", 0.05))
        if mid < min_mid:
            self.logger.info(f"[REJECT] {label} {symbol} mid {mid:.4f} < min_mid {min_mid:.4f}")
            self.rejection_reasons[f"{label}_mid_too_small"] += 1
            return False

        # --- Delta checks ---
        if delta is None:
            self.logger.info(f"[REJECT] {label} {symbol} missing delta")
            self.rejection_reasons[f"{label}_delta_missing"] += 1
            return False

        use_abs_delta = bool(leg_criteria.get("use_abs_delta", True))
        if use_abs_delta:
            min_delta = float(leg_criteria.get("min_delta", 0.0))
            max_delta = float(leg_criteria.get("max_delta", 1.0))
            if abs_delta < min_delta:
                self.logger.info(f"[REJECT] {label} {symbol} abs(delta) {abs_delta:.3f} < min {min_delta}")
                self.rejection_reasons[f"{label}_delta_too_low"] += 1
                return False
            if abs_delta > max_delta:
                self.logger.info(f"[REJECT] {label} {symbol} abs(delta) {abs_delta:.3f} > max {max_delta}")
                self.rejection_reasons[f"{label}_delta_too_high"] += 1
                return False
        else:
            min_delta = float(leg_criteria.get("min_delta", -99))
            max_delta = float(leg_criteria.get("max_delta", 999))
            if delta < min_delta or delta > max_delta:
                self.logger.info(f"[REJECT] {label} {symbol} delta {delta:.3f} outside [{min_delta}, {max_delta}]")
                self.rejection_reasons[f"{label}_delta_signed_range"] += 1
                return False

        # --- IV checks ---
        if iv <= 0:
            self.logger.info(f"[REJECT] {label} {symbol} IV {iv:.3f} non-positive")
            self.rejection_reasons[f"{label}_iv_nonpos"] += 1
            return False
        min_iv = leg_criteria.get("min_iv")
        max_iv = leg_criteria.get("max_iv")
        if min_iv is not None and iv < float(min_iv):
            self.logger.info(f"[REJECT] {label} {symbol} IV {iv:.2f} < min_iv {min_iv}")
            self.rejection_reasons[f"{label}_iv_too_low"] += 1
            return False
        if max_iv is not None and iv > float(max_iv):
            self.logger.info(f"[REJECT] {label} {symbol} IV {iv:.2f} > max_iv {max_iv}")
            self.rejection_reasons[f"{label}_iv_too_high"] += 1
            return False

        # --- Open interest ---
        min_oi = int(leg_criteria.get("min_oi", 0))
        if oi < min_oi:
            self.logger.info(f"[REJECT] {label} {symbol} OI {oi} < min_oi {min_oi}")
            self.rejection_reasons[f"{label}_oi"] += 1
            return False

        # --- DTE (global expiry criteria) ---
        min_dte = int(criteria.get("min_days_to_exp", 0))
        max_dte = int(criteria.get("max_days_to_exp", 999))
        if not (min_dte <= dte <= max_dte):
            if dte < min_dte:
                self.logger.info(f"[REJECT] {label} {symbol} DTE {dte} < min {min_dte}")
                self.rejection_reasons[f"{label}_dte_too_short"] += 1
            else:
                self.logger.info(f"[REJECT] {label} {symbol} DTE {dte} > max {max_dte}")
                self.rejection_reasons[f"{label}_dte_too_long"] += 1
            return False

        return True

    def find_iron_condor(
            self,
            put_options: List[Any],
            call_options: List[Any],
            underlying_price: float,
            risk_free_rate: float
    ) -> Tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict]]:
        """
        Iron Condor Finder (consistent, single rejection tracker).
        Expects self.rejection_reasons to exist and be a defaultdict(int).
        """
        # Ensure single source of rejection counts
        if not hasattr(self, "rejection_reasons") or self.rejection_reasons is None:
            self.rejection_reasons = defaultdict(int)

        def build_candidates(options, leg_label):
            valid = []
            for opt in options:
                try:
                    candidate = self.build_option_dict(opt, underlying_price, risk_free_rate)
                    # pass original label (case-insensitive handling inside check_option_conditions)
                    if self.check_option_conditions(candidate, leg_label, self.criteria):
                        valid.append(candidate)
                    else:
                        self.rejection_reasons[f"{leg_label}_FILTERED"] += 1
                except Exception as e:
                    self.logger.warning(f"[ERROR] Build failed for {getattr(opt, 'symbol', 'unknown')}: {e}")
                    self.rejection_reasons[f"{leg_label}_BUILD_ERROR"] += 1
            self.logger.info(
                f"[CANDIDATES] {leg_label}: {len(valid)} valid, {self.rejection_reasons.get(f'{leg_label}_FILTERED', 0)} filtered")
            return valid

        # build candidates (use uppercase labels consistently)
        sp_candidates = build_candidates(put_options, "SHORT_PUT")
        lp_candidates = build_candidates(put_options, "LONG_PUT")
        sc_candidates = build_candidates(call_options, "SHORT_CALL")
        lc_candidates = build_candidates(call_options, "LONG_CALL")

        if not sc_candidates and not lc_candidates:
            self.logger.warning(
                f"[CALLS] No valid call candidates found (SHORT_CALL rejected: {self.rejection_reasons.get('SHORT_CALL_FILTERED', 0)}, LONG_CALL rejected: {self.rejection_reasons.get('LONG_CALL_FILTERED', 0)})"
            )

        # group by expiry
        sp_by_exp, lp_by_exp = defaultdict(list), defaultdict(list)
        sc_by_exp, lc_by_exp = defaultdict(list), defaultdict(list)

        for c in sp_candidates: sp_by_exp[c["expiration_date"]].append(c)
        for c in lp_candidates: lp_by_exp[c["expiration_date"]].append(c)
        for c in sc_candidates: sc_by_exp[c["expiration_date"]].append(c)
        for c in lc_candidates: lc_by_exp[c["expiration_date"]].append(c)

        self.logger.info(
            f"[EXP CHECK] SP={list(sp_by_exp.keys())}, LP={list(lp_by_exp.keys())}, SC={list(sc_by_exp.keys())}, LC={list(lc_by_exp.keys())}")

        best_ic, best_score = None, float("-inf")

        # iterate only expiries that have all legs
        common_exps = set(sp_by_exp.keys()) & set(lp_by_exp.keys()) & set(sc_by_exp.keys()) & set(lc_by_exp.keys())
        for exp in common_exps:
            # ordering for pairing: short puts highest strike, long puts lower strikes
            puts_short = sorted(sp_by_exp[exp], key=lambda x: float(x["strike_price"]), reverse=True)
            puts_long = sorted(lp_by_exp[exp], key=lambda x: float(x["strike_price"]))
            calls_short = sorted(sc_by_exp[exp], key=lambda x: float(x["strike_price"]))
            calls_long = sorted(lc_by_exp[exp], key=lambda x: float(x["strike_price"]), reverse=True)

            for sp in puts_short:
                for lp in puts_long:
                    # ensure proper order: short_put strike > long_put strike
                    if float(lp["strike_price"]) >= float(sp["strike_price"]):
                        self.rejection_reasons["PUT_ORDER_FAIL"] += 1
                        continue
                    put_width = float(sp["strike_price"]) - float(lp["strike_price"])
                    if put_width > float(self.criteria.get("max_spread_width", 10.0)):
                        self.rejection_reasons["PUT_WIDTH_FAIL"] += 1
                        continue

                    for sc in calls_short:
                        for lc in calls_long:
                            # ensure proper order: long_call strike > short_call strike
                            if float(lc["strike_price"]) <= float(sc["strike_price"]):
                                self.rejection_reasons["CALL_ORDER_FAIL"] += 1
                                continue
                            call_width = float(lc["strike_price"]) - float(sc["strike_price"])
                            if call_width > float(self.criteria.get("max_spread_width", 10.0)):
                                self.rejection_reasons["CALL_WIDTH_FAIL"] += 1
                                continue

                            # cross-wing ordering: short_put strike must be < short_call strike
                            if float(sp["strike_price"]) >= float(sc["strike_price"]):
                                self.rejection_reasons["CONDOR_OVERLAP_FAIL"] += 1
                                continue

                            # buying power: use consistent conservative pricing inside the check
                            if not self.check_buying_power(sp, lp, sc, lc):
                                self.rejection_reasons["BUYING_POWER_FAIL"] += 1
                                continue

                            # score candidate
                            try:
                                score = self.score_iron_condor(sp, lp, sc, lc)
                                if score > best_score:
                                    best_score, best_ic = score, (sp, lp, sc, lc)
                            except Exception as e:
                                self.logger.warning(f"[IC SCORING ERROR] {e}")
                                self.rejection_reasons["SCORING_ERROR"] += 1

        # finalize
        if best_ic:
            sp, lp, sc, lc = best_ic
            self.logger.info(
                f"[IC SELECTED] SP={sp['strike_price']} LP={lp['strike_price']} | SC={sc['strike_price']} LC={lc['strike_price']} | Score={best_score:.4f}"
            )
            return sp, lp, sc, lc

        self.logger.warning(
            f"[IC FAILED] No valid iron condor found. Rejections summary: {dict(self.rejection_reasons)}")
        return None, None, None, None

    def score_iron_condor(self, short_put: dict, long_put: dict,
                          short_call: dict, long_call: dict) -> float:
        """
        Score an iron condor using candidate mid prices consistently.
        """
        try:
            # use candidate mid_price (or compute robustly from bid/ask if missing)
            def candidate_mid(c):
                c_mid = float(c.get("mid_price") or 0.0)
                if c_mid > 0:
                    return c_mid
                b = float(c.get("bid_price") or 0.0)
                a = float(c.get("ask_price") or 0.0)
                return (b + a) / 2.0 if (b > 0 or a > 0) else 0.0

            sp_mid = candidate_mid(short_put)
            lp_mid = candidate_mid(long_put)
            sc_mid = candidate_mid(short_call)
            lc_mid = candidate_mid(long_call)

            net_credit = (sp_mid - lp_mid) + (sc_mid - lc_mid)

            # widths (positive)
            put_width = float(short_put["strike_price"]) - float(long_put["strike_price"])
            call_width = float(long_call["strike_price"]) - float(short_call["strike_price"])
            max_width = max(put_width, call_width)

            max_loss = max_width - net_credit
            if max_loss <= 0 or net_credit <= 0:
                return 0.0

            reward_risk = net_credit / max_loss

            # theta benefit (short - long)
            net_theta = (
                    float(short_put.get("theta", 0.0)) - float(long_put.get("theta", 0.0)) +
                    float(short_call.get("theta", 0.0)) - float(long_call.get("theta", 0.0))
            )

            # liquidity & OI
            oi_sum = (
                    float(short_put.get("open_interest", 0)) +
                    float(long_put.get("open_interest", 0)) +
                    float(short_call.get("open_interest", 0)) +
                    float(long_call.get("open_interest", 0))
            )

            score = reward_risk
            if net_theta > 0:
                score *= (1 + 0.05 * net_theta)
            if oi_sum < 200:
                score *= 0.7

            return round(score, 4)

        except Exception as e:
            self.logger.warning(f"[IC SCORE ERROR] {e}")
            return 0.0

    def check_buying_power(self,
                           short_put: Dict[str, Any],
                           long_put: Dict[str, Any],
                           short_call: Dict[str, Any],
                           long_call: Dict[str, Any]) -> bool:
        """
        Conservative buying power check:
          - Uses short bid and long ask (conservative credit)
          - Uses contract multiplier = 100 * number_of_contracts (candidate.size)
          - Allows equality (max_loss > allowed_limit -> reject)
        """
        try:
            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)
            portfolio_value = float(account.portfolio_value)

            buy_power_limit = buying_power * float(getattr(self, "buy_power_limit", 0.05))
            self.logger.debug(
                f"[ACCOUNT] Buying Power: ${buying_power:,.2f} | Portfolio Value: ${portfolio_value:,.2f} | Allowed Risk Limit: ${buy_power_limit:,.2f}")

            # contract multiplier: number of contracts * 100
            contracts = int(short_put.get("size", 1)) or 1
            contract_multiplier = contracts * 100.0

            sp_strike = float(short_put['strike_price'])
            lp_strike = float(long_put['strike_price'])
            sc_strike = float(short_call['strike_price'])
            lc_strike = float(long_call['strike_price'])

            # conservative pricing: short uses bid, long uses ask
            sp_bid = float(short_put.get('bid_price') or 0.0)
            lp_ask = float(long_put.get('ask_price') or 0.0)
            sc_bid = float(short_call.get('bid_price') or 0.0)
            lc_ask = float(long_call.get('ask_price') or 0.0)

            put_width = abs(sp_strike - lp_strike)
            call_width = abs(lc_strike - sc_strike)

            # net credit using conservative prices
            net_credit = (sp_bid - lp_ask) + (sc_bid - lc_ask)

            # max risk in dollars for the structure
            max_loss = (max(put_width, call_width) - net_credit) * contract_multiplier

            # debug log to make later triage easier
            self.logger.debug(
                f"[BUYING POWER CHECK] contracts={contracts}, contract_multiplier={contract_multiplier}, put_width={put_width}, call_width={call_width}, net_credit={net_credit:.2f}, max_loss=${max_loss:.2f}, allowed=${buy_power_limit:.2f}")

            # reject only when strictly greater than allowed (allow equality)
            if max_loss > buy_power_limit:
                self.rejection_reasons['buying_power_exceeded'] += 1
                return False
            return True

        except Exception as e:
            self.logger.error(f"[IC ERROR] check_buying_power failed: {e}")
            # conservative fail-safe: reject when we cannot determine buying power
            self.rejection_reasons['buying_power_check_error'] += 1
            return False

    def roll_rinse_iron_condor(
            self,
            short_put: Dict[str, Any],
            long_put: Dict[str, Any],
            short_call: Dict[str, Any],
            long_call: Dict[str, Any],
            underlying_price: float
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        """
        Evaluates exit/roll/hold conditions for an Iron Condor.

        Returns:
            action: "HOLD", "EXIT", "ROLL", or "ERROR"
            spread_dict: updated spread if applicable
            reason: explanation string for logging
        """

        # ---------------------------
        # Helper: latest mid price
        # ---------------------------
        def latest_mid(symbol: str) -> Optional[float]:
            try:
                quote_req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
                q = self.option_historical_data_client.get_option_latest_quote(quote_req)[symbol]
                bid = self.safe_float(getattr(q, "bid_price", None))
                ask = self.safe_float(getattr(q, "ask_price", None))
                if bid and ask and bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
            except Exception as e:
                self.logger.warning(f"[QUOTE] mid fetch failed for {symbol}: {e}")
            return None

        def resolve_entry_price(leg: Dict[str, Any]) -> Optional[float]:
            for k in ("avg_entry_price", "initial_option_price", "option_price", "mid_price"):
                v = leg.get(k)
                if v and float(v) > 0:
                    return float(v)
            return latest_mid(leg.get("symbol"))

        try:
            # --- 0) Validate ---
            if not underlying_price or underlying_price <= 0:
                return "ERROR", None, f"Invalid underlying price={underlying_price}"

            symbol = short_put.get("underlying_symbol") or short_call.get("underlying_symbol")
            if not symbol:
                return "ERROR", None, "Missing underlying symbol"

            # --- 1) Entry net credit ---
            sp_entry = resolve_entry_price(short_put)
            lp_entry = resolve_entry_price(long_put)
            sc_entry = resolve_entry_price(short_call)
            lc_entry = resolve_entry_price(long_call)

            if not all([sp_entry, lp_entry, sc_entry, lc_entry]):
                return "ERROR", None, "Missing entry prices for one or more legs"

            entry_credit = (sp_entry - lp_entry) + (sc_entry - lc_entry)
            if entry_credit <= 0:
                return "ERROR", None, f"Non-credit entry (entry_credit={entry_credit:.2f})"

            # Profit target (similar rules as bull put spread)
            base_target = float(getattr(self, "target_profit_percent", 0.35) or 0.35)
            if entry_credit <= 0.20:
                target_profit = 0.50
            elif entry_credit < 0.30:
                target_profit = 0.30
            else:
                target_profit = max(min(base_target, 0.50), 0.30)

            relaxed_target = round(target_profit * 0.7, 2)

            # --- 2) Current PnL ---
            try:
                positions = self.trading_client.get_all_positions()
                spread_pnl = 0.0
                for leg in (short_put, long_put, short_call, long_call):
                    pos = next((p for p in positions if p.symbol == leg["symbol"]), None)
                    if pos:
                        spread_pnl += float(pos.unrealized_pl)
                profit_pct = spread_pnl / (entry_credit * 100)
                pnl_source = "broker"
            except Exception as e:
                self.logger.warning(f"[PNL] Broker PnL fetch failed, using mids: {e}")
                sp_mid = latest_mid(short_put["symbol"]) or sp_entry
                lp_mid = latest_mid(long_put["symbol"]) or lp_entry
                sc_mid = latest_mid(short_call["symbol"]) or sc_entry
                lc_mid = latest_mid(long_call["symbol"]) or lc_entry
                close_debit = (sp_mid - lp_mid) + (sc_mid - lc_mid)
                realized_profit = entry_credit - close_debit
                profit_pct = realized_profit / entry_credit
                pnl_source = "mid"

            # --- 3) Holding time & exit windows ---
            weekday_today = datetime.now(self.timezone).weekday()  # 0=Mon … 4=Fri
            relaxed_active = (weekday_today >= 2)
            friday_only = (weekday_today == 4)

            # --- 4) Risk metrics ---
            put_delta = abs(float(short_put.get("delta", 0.0)))
            call_delta = abs(float(short_call.get("delta", 0.0)))

            # --- Exit conditions ---
            if profit_pct >= target_profit:
                return "EXIT", None, f"Target {int(target_profit * 100)}% hit (pnl={profit_pct:.1%}, src={pnl_source})"

            if relaxed_active and profit_pct >= relaxed_target:
                return "EXIT", None, f"Relaxed target {int(relaxed_target * 100)}% hit midweek (pnl={profit_pct:.1%}, src={pnl_source})"

            if friday_only:
                if put_delta >= 0.65 or call_delta >= 0.65:
                    return "EXIT", None, f"Delta breach (Fri only, putΔ={put_delta:.2f}, callΔ={call_delta:.2f})"

            # --- Roll logic (Fri only, under pressure) ---
            if friday_only and profit_pct < -0.40:
                threatened_side = "PUT" if put_delta > call_delta else "CALL"
                try:
                    current_expiry = datetime.strptime(short_put["expiration_date"], "%Y-%m-%d")
                    new_expiration = current_expiry + timedelta(days=7)
                except Exception:
                    new_expiration = None

                try:
                    if threatened_side == "PUT":
                        width = abs(float(short_put["strike_price"]) - float(long_put["strike_price"]))
                        new_short = float(short_put["strike_price"]) - 2
                        new_long = new_short - width

                        new_spread = {
                            "short_put": {
                                **short_put,
                                "strike_price": new_short,
                                "expiration_date": new_expiration.strftime("%Y-%m-%d"),
                            },
                            "long_put": {
                                **long_put,
                                "strike_price": new_long,
                                "expiration_date": new_expiration.strftime("%Y-%m-%d"),
                            },
                            "short_call": short_call,
                            "long_call": long_call,
                            "entry_time": datetime.now(self.timezone).isoformat()
                        }

                    else:  # CALL side threatened
                        width = abs(float(long_call["strike_price"]) - float(short_call["strike_price"]))
                        new_short = float(short_call["strike_price"]) + 2
                        new_long = new_short + width

                        new_spread = {
                            "short_call": {
                                **short_call,
                                "strike_price": new_short,
                                "expiration_date": new_expiration.strftime("%Y-%m-%d"),
                            },
                            "long_call": {
                                **long_call,
                                "strike_price": new_long,
                                "expiration_date": new_expiration.strftime("%Y-%m-%d"),
                            },
                            "short_put": short_put,
                            "long_put": long_put,
                            "entry_time": datetime.now(self.timezone).isoformat()
                        }

                    return "ROLL", new_spread, f"Rolled {threatened_side} side (Fri only, pnl={profit_pct:.1%})"

                except Exception as e:
                    return "ERROR", None, f"Failed to compute roll: {e}"

            # --- HOLD ---
            self.logger.info(
                f"[HOLD] {symbol} | pnl={profit_pct:.1%}, "
                f"targets=({int(target_profit * 100)}% strict / {int(relaxed_target * 100)}% relaxed), "
                f"putΔ={put_delta:.2f}, callΔ={call_delta:.2f}, src={pnl_source}"
            )
            return "HOLD", None, "No exit criteria met"

        except Exception as e:
            self.logger.error(f"[IRON CONDOR ROLL ERROR] {e}")
            return "ERROR", None, str(e)

    def _leg_close_side(self, side_str: str | PositionSide) -> OrderSide:
        if isinstance(side_str, PositionSide):
            if side_str == PositionSide.SHORT:
                return OrderSide.BUY  # to close a short, buy back
            elif side_str == PositionSide.LONG:
                return OrderSide.SELL  # to close a long, sell to close
            else:
                raise ValueError(f"Unknown PositionSide: {side_str}")

        s = (str(side_str) or "").lower().replace("_to_close", "").replace("_to_open", "")
        if s == "sell":
            return OrderSide.BUY
        if s == "buy":
            return OrderSide.SELL
        raise ValueError(f"Unknown leg side: {side_str}")

    def _collect_legs_for_action(self, spread: Dict, action: str) -> List[OptionLegRequest]:
        """
        Collect all option legs for a given action (open/close).
        - Respects actual qty for each leg (not hardcoded to 1).
        - Ensures all legs share the same expiration date (iron condor consistency).
        """
        legs = []
        expirations = set()

        for leg_key, leg in spread.items():
            if not isinstance(leg, dict):
                continue  # skip non-leg entries (like 'meta', 'underlying', etc.)

            try:
                symbol = leg["symbol"]
                qty = leg.get("qty", 1)
                expiration = leg.get("expiration_date")
                expirations.add(expiration)

                side = None
                if action == "open":
                    side = leg.get("side")
                elif action == "close":
                    side = self._leg_close_side(leg.get("side"))

                if side is None:
                    raise ValueError(f"Invalid side for leg {leg_key} during {action}")

                legs.append(
                    OptionLegRequest(symbol=symbol, side=side, ratio_qty=qty)
                )

            except KeyError as e:
                self.logger.error(f"Missing key {e} in leg {leg_key}: {leg}")
                raise

        # --- Consistency check: all legs should share the same expiration ---
        if len(expirations) > 1:
            raise ValueError(f"Spread legs have mismatched expirations: {expirations}")

        return legs

    def submit_mleg_open(self, spread: dict, total_qty: int, limit_price: float | None = None):
        """
        Place an entry MLEG order for a 2-leg spread or 4-leg iron condor.
        - total_qty: number of multi-leg spreads to trade (e.g., 5 means 5 x each leg)
        - limit_price: credit (+) or debit (absolute) target for the net price; if None => MARKET
        """
        legs = self._collect_legs_for_action(spread, action="open")

        if limit_price is None:
            req = MarketOrderRequest(
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=legs,
                qty=total_qty
            )
        else:
            req = LimitOrderRequest(
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=legs,
                qty=total_qty,
                limit_price=float(limit_price)
            )
        res = self.trading_client.submit_order(req)
        self.logger.info(
            f"[MLEG OPEN] Submitted MLEG order (qty={total_qty}, limit={limit_price}) for {spread.get('symbol')}")
        return res

    def submit_mleg_close(self, spread: dict, total_qty: int, limit_price: float | None = None):
        """
        Close the spread atomically as a multi-leg order.
        If limit_price is None, uses MARKET; else closes at specified net debit/credit.
        """
        legs = self._collect_legs_for_action(spread, action="close")

        if limit_price is None:
            req = MarketOrderRequest(
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=legs,
                qty=total_qty
            )
        else:
            req = LimitOrderRequest(
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=legs,
                qty=total_qty,
                limit_price=float(limit_price)
            )
        res = self.trading_client.submit_order(req)
        self.logger.info(
            f"[MLEG CLOSE] Submitted MLEG close (qty={total_qty}, limit={limit_price}) for {spread.get('symbol')}")
        return res

    def load_spread_state(self, path="iron_condor_spread_state.json"):
        """
        Load saved spread state from disk (used only at startup).
        Falls back to empty if file is missing, corrupt, or contains invalid entries.
        """
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    state = json.load(f)

                    # 🔑 Ensure only dicts are returned
                    valid = [s for s in state if isinstance(s, dict)]
                    if len(valid) != len(state):
                        self.logger.warning(f"[STATE LOAD] Dropped {len(state) - len(valid)} invalid spread entries")
                    self.logger.info(f"[STATE LOADED] {len(valid)} spreads from {path}")
                    return valid
        except Exception as e:
            self.logger.warning(f"[LOAD ERROR] Could not read {path}: {e}")

        return []

    def save_spread_state(self, state, path="iron_condor_spread_state.json"):
        try:
            with open(path, "w") as f:
                json.dump(state, f, default=str)
        except Exception as e:
            self.logger.error(f"[SAVE ERROR] Could not save spread state: {e}")

    def cleanup_expired_spreads(self, spreads):
        """
        Remove expired spreads, keep valid or unparsable ones.
        Ignore non-dict junk entries safely.
        """
        today = datetime.now().date()
        cleaned_spreads = []
        for s in spreads:
            if not isinstance(s, dict):
                self.logger.warning(f"[CLEANUP] Skipping invalid spread entry: {s}")
                continue

            expiry_str = s.get("expiry")
            if not expiry_str or str(expiry_str).lower() == "none":
                cleaned_spreads.append(s)
                continue

            try:
                expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            except ValueError:
                try:
                    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d %H:%M:%S").date()
                except ValueError:
                    cleaned_spreads.append(s)  # Keep if unparsable
                    continue

            if expiry_date >= today:
                cleaned_spreads.append(s)

        return cleaned_spreads

    def sync_with_alpaca(self):
        """
        Strict sync: only keep what Alpaca says exists live,
        preserve entered_at from previous state if available.
        """

        def parse_option_symbol(sym: str) -> dict:
            m = re.match(r"^([A-Z]+)(\d{6})([PC])(\d+)$", sym)
            if not m:
                return {"underlying": sym, "expiry": None, "type": None, "strike": None}

            underlying, date, opt_type, strike = m.groups()
            expiry = datetime.strptime(date, "%y%m%d").date()
            strike_price = int(strike) / 1000

            return {
                "underlying": underlying,
                "expiry": expiry,
                "type": "put" if opt_type == "P" else "call",
                "strike": strike_price
            }

        try:
            positions = self.trading_client.get_all_positions()
        except Exception as e:
            self.logger.error(f"[SYNC ERROR] Failed to fetch positions: {e}")
            return []

        grouped_spreads: dict[str, dict] = {}

        # --- Build spreads from Alpaca positions ---
        for pos in positions:
            pos_data = pos.model_dump() if hasattr(pos, "model_dump") else pos.__dict__
            sym = pos_data.get("symbol")
            if not sym:
                continue

            parsed = parse_option_symbol(sym)
            base_symbol = parsed["underlying"]

            if base_symbol not in grouped_spreads:
                grouped_spreads[base_symbol] = {
                    "symbol": base_symbol,
                    "short_put": {},
                    "long_put": {},
                    "short_call": {},
                    "long_call": {},
                    "entered_at": str(datetime.now()),
                    "expiry": None,
                    "qty": None,
                }

            if parsed["type"] == "put":
                if pos.side.lower() == "short":
                    grouped_spreads[base_symbol]["short_put"] = pos_data
                    grouped_spreads[base_symbol]["qty"] = abs(int(float(pos.qty)))
                    grouped_spreads[base_symbol]["expiry"] = parsed["expiry"]
                elif pos.side.lower() == "long":
                    grouped_spreads[base_symbol]["long_put"] = pos_data
            elif parsed["type"] == "call":
                if pos.side.lower() == "short":
                    grouped_spreads[base_symbol]["short_call"] = pos_data
                elif pos.side.lower() == "long":
                    grouped_spreads[base_symbol]["long_call"] = pos_data

        live_spreads = list(grouped_spreads.values())

        # --- Preserve entered_at from old state ---
        old_state = self.load_spread_state() or []
        old_map = {s["symbol"]: s for s in old_state}

        for spread in live_spreads:
            sym = spread["symbol"]
            if sym in old_map and "entered_at" in old_map[sym]:
                spread["entered_at"] = old_map[sym]["entered_at"]

        self.logger.info(f"[ALPACA SPREADS] Live from API: {len(live_spreads)} items")
        for spread in live_spreads:
            short_p, long_p = spread["short_put"].get("symbol"), spread["long_put"].get("symbol")
            short_c, long_c = spread["short_call"].get("symbol"), spread["long_call"].get("symbol")
            self.logger.info(f"[DEBUG] {spread['symbol']} | "
                             f"ShortP: {short_p}, LongP: {long_p}, "
                             f"ShortC: {short_c}, LongC: {long_c}")

        # 🔑 Persist spreads only (no cooldown)
        self.save_spread_state(live_spreads)

        self.symbols_with_open_options = {spread["symbol"] for spread in live_spreads}
        return live_spreads

    def evaluate_iron_condor_options(self, put_options: list, call_options: list, price: float) -> None:
        """
        Evaluate short/long put and call candidates for Iron Condor.
        Logs rejection reasons for each leg and prints option details.

        Args:
            put_options (list): List of put option contracts.
            call_options (list): List of call option contracts.
            price (float): Current underlying price.
        """
        criteria = self.criteria  # should contain short_put, long_put, short_call, long_call

        short_put_table, long_put_table = [], []
        short_call_table, long_call_table = [], []
        columns = ["Strike", "Delta", "IV", "OI", "DTE", "Spread", "Reason"]

        all_options = put_options + call_options
        now = pd.Timestamp.now(tz="UTC")

        for opt in all_options:
            opt_dict = self.build_option_dict(opt, price, self.risk_free_rate)

            delta = float(opt_dict.get("delta", 0))
            iv = float(opt_dict.get("implied_volatility", 0))
            bid = float(opt_dict.get("bid_price", 0))
            ask = float(opt_dict.get("ask_price", 0))
            oi = opt_dict.get("open_interest", 0)
            expiration = opt_dict.get("expiration_date")

            if expiration is None:
                continue
            if expiration.tzinfo is None:
                expiration = expiration.tz_localize("UTC")

            dte = (expiration - now).days
            spread = ask - bid
            strike = float(opt_dict["strike_price"])

            # --- Helper to validate legs ---
            def check_leg(leg_type: str) -> list:
                # leg_type is 'short_put' etc.
                leg_criteria = self.criteria.get(leg_type, {})
                reasons = []

                # DTE uses top-level min/max (consistent)
                min_dte = self.criteria.get("min_days_to_exp")
                max_dte = self.criteria.get("max_days_to_exp")
                if min_dte is not None and dte < min_dte:
                    reasons.append(f"DTE<{min_dte}")
                if max_dte is not None and dte > max_dte:
                    reasons.append(f"DTE>{max_dte}")

                if oi == 0:
                    reasons.append("OI=0")
                if spread > leg_criteria.get("max_bid_ask_spread", float("inf")):
                    reasons.append(f"Spread>{leg_criteria.get('max_bid_ask_spread')}")
                if iv < leg_criteria.get("min_iv", -1):
                    reasons.append(f"IV<{leg_criteria.get('min_iv')}")
                if delta < leg_criteria.get("min_delta", -99):
                    reasons.append(f"Delta<{leg_criteria.get('min_delta')}")
                if delta > leg_criteria.get("max_delta", 999):
                    reasons.append(f"Delta>{leg_criteria.get('max_delta')}")
                return reasons

            # --- Assign option to relevant rejection table ---
            for leg_type, table in [
                ("short_put", short_put_table),
                ("long_put", long_put_table),
                ("short_call", short_call_table),
                ("long_call", long_call_table),
            ]:
                reasons = check_leg(leg_type)
                if reasons:
                    table.append([
                        strike, f"{delta:.2f}", f"{iv:.2f}", oi, dte, f"{spread:.2f}",
                        ", ".join(reasons)
                    ])

        # --- Logging rejection tables ---
        for name, table in [
            ("SHORT PUTS", short_put_table),
            ("LONG PUTS", long_put_table),
            ("SHORT CALLS", short_call_table),
            ("LONG CALLS", long_call_table),
        ]:
            if table:
                df = pd.DataFrame(table, columns=columns)
                self.logger.info(f"\n[REJECTED {name}]\n{df.to_string(index=False)}")
            else:
                self.logger.info(f"[REJECTED {name}] None")

        # --- Log all option details ---
        for opt in all_options:
            opt_dict = self.build_option_dict(opt, price, self.risk_free_rate)

            delta = opt_dict.get("delta")
            iv = opt_dict.get("implied_volatility")
            bid = opt_dict.get("bid_price", 0)
            ask = opt_dict.get("ask_price", 0)
            oi = opt_dict.get("open_interest", "N/A")

            self.logger.info(
                f"[OPTION] {opt_dict['symbol']} | Strike: {opt_dict['strike_price']} | "
                f"Exp: {opt_dict['expiration_date']} | "
                f"Delta: {f'{float(delta):.2f}' if delta is not None else 'N/A'} | "
                f"IV: {f'{float(iv):.2%}' if iv is not None else 'N/A'} | "
                f"Bid: {float(bid):.2f} | Ask: {float(ask):.2f} | OI: {oi}"
            )

    def run(self, market_open, market_close):
        """Start Bull Put Spread algorithm with time-controlled entry logic and threading."""
        logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)

        self.trading_stream = TradingStream(API_KEY, SECRET_KEY, paper=True)
        self.wss_client = StockDataStream(API_KEY, SECRET_KEY)

        self.minute_history = {}
        self.positions = {}
        self.partial_fills = {}
        self.open_orders = {}
        self.lock = threading.Lock()
        self.last_screening_minute = {}
        self.entry_log = {}

        today = datetime.now().date()
        if today.weekday() > 4:
            self.logger.info("Today is not a trading day (Mon–Fri). Exiting.")
            return

        symbols = self.get_underlying_assets()
        self.logger.info(f"[SCREENED] Bullish symbols to consider today: {symbols}")
        for symbol in symbols:
            self.last_screening_minute[symbol] = None

        self.minute_history = self.get_1000m_history_data(symbols)

        def run_trading_stream():
            async def handle_trade_updates(data):
                try:
                    symbol = data.order.symbol
                    event = data.event
                    qty = float(data.order.qty or 0)
                    price = float(data.order.limit_price or 0)

                    self.logger.debug(f"[TRADE UPDATE] {event.upper()} received for {symbol} "
                                      f"(qty={qty}, price={price})")

                    if event == "new":
                        position_qty = float(getattr(data, "position_qty", 0))
                        self.positions[symbol] = position_qty
                        self.open_orders[symbol] = data.order
                        self.logger.info(f"[NEW ORDER] {symbol}: qty={qty}, price={price}, "
                                         f"position_qty={position_qty}")

                    elif event == "partial_fill":
                        prev = self.partial_fills.get(symbol, 0)
                        self.positions[symbol] = qty - prev
                        self.partial_fills[symbol] = qty
                        self.open_orders[symbol] = data.order
                        self.logger.info(f"[PARTIAL FILL] {symbol}: filled={qty}, "
                                         f"remaining={data.order.remaining_qty}")

                    elif event == "fill":
                        self.positions[symbol] = qty
                        self.partial_fills[symbol] = 0
                        self.open_orders[symbol] = None
                        self.logger.info(f"[FILLED] {symbol}: total position now {qty}")

                    elif event in ["canceled", "rejected"]:
                        self.partial_fills[symbol] = 0
                        self.open_orders[symbol] = None
                        self.logger.warning(f"[{event.upper()}] Order for {symbol} was {event}")

                except Exception as e:
                    self.logger.error(f"[ERROR] Exception in trade update handler: {e}",
                                      exc_info=True)

            @self.trading_stream.subscribe_trade_updates
            async def print_trade_update(*symbols):
                print('trade update', *symbols)

            self.trading_stream.subscribe_trade_updates(handle_trade_updates)
            self.trading_stream.run()

        def run_wss_client():
            async def handle_second_bar(data):
                try:
                    symbol = data.symbol
                    ts = data.timestamp
                    if isinstance(ts, str):
                        ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")  # parse string to datetime

                    ts = ts.replace(second=0, microsecond=0)
                    self.logger.info(f"[BAR] {symbol} @ {ts}")

                    if symbol not in self.minute_history:
                        self.minute_history[symbol] = self.get_1000m_history_data([symbol])[symbol]

                    df = self.minute_history[symbol]
                    if not isinstance(df.index, pd.MultiIndex):
                        df.index = pd.MultiIndex.from_tuples(df.index, names=["symbol", "timestamp"])

                    index_key = (symbol, ts)
                    new_data = [data.open, data.high, data.low, data.close, data.volume, data.trade_count, data.vwap]
                    columns = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

                    if index_key not in df.index:
                        new_row = pd.DataFrame(
                            [new_data],
                            index=pd.MultiIndex.from_tuples([index_key], names=["symbol", "timestamp"]),
                            columns=columns
                        )
                        df = pd.concat([df, new_row])
                    else:
                        df.loc[index_key] = new_data

                    self.minute_history[symbol] = df
                    if len(df) < 60:
                        return

                    df = self.add_all_features(df)
                    latest = df.iloc[-1]
                    prev = df.iloc[-2]

                    rsi, adx, macd, macd_hist = latest['RSI'], latest['ADX'], latest['MACD'], latest['MACD_Hist']
                    short_sma, long_sma = latest['20_MA'], latest['50_MA']
                    volume = latest['volume']
                    volume_sma = df['volume'].rolling(20).mean().iloc[-1]
                    price = latest['close']

                    # Define DTE range for iron condor: 5 to 7 days
                    min_dte, max_dte = 5, 7

                    # Update criteria dynamically for all four legs
                    for leg in ['short_put', 'long_put', 'short_call', 'long_call']:
                        self.criteria[leg]['min_days_to_exp'] = min_dte
                        self.criteria[leg]['max_days_to_exp'] = max_dte

                    self.logger.info(f"[CRITERIA] Targeting expiries between {min_dte}–{max_dte} days out")

                    if market_open + timedelta(minutes=15) <= ts < market_open.replace(hour=16, minute=0):
                        # Refresh before screening
                        self.active_spreads = self.sync_with_alpaca()

                        self.logger.info(f"[DEBUG] Skip list (open legs): {self.symbols_with_open_options}")
                        self.logger.info(f"[DEBUG] Cooldown list (already traded): {self.last_traded}")

                        symbol_base = symbol.split()[0].strip()

                        # Skip if this symbol has *any* open option position
                        if symbol_base in self.symbols_with_open_options:
                            self.logger.info(f"[SKIP] {symbol} already has an open option leg.")
                            return

                        # --- Enforce 5-minute minimum re-screening interval per symbol ---
                        last_check: Optional[datetime] = self.last_screening_minute.get(symbol)

                        if last_check is None or (ts - last_check).total_seconds() >= 300:
                            self.last_screening_minute[symbol] = ts

                            # Scoring for neutral setup
                            score = sum([
                                1 if 40 < rsi < 60 else 0,  # Neutral RSI zone
                                1 if abs(macd_hist) < 0.2 else 0,  # Flat MACD histogram
                                1 if abs(short_sma - long_sma) / price < 0.01 else 0,  # MAs converging
                                1 if adx < 25 else 0,  # Low trend strength
                                1 if volume > (volume_sma * 0.8) else 0  # Healthy liquidity
                            ])

                            self.logger.info(f"[SCREENED] {symbol} Score: {score}/5")

                            if score < 3:
                                self.logger.info(f"[BLOCKED] Neutral signal score too low ({score}/5) for {symbol}")
                                return

                            self.logger.info(f"[ENTRY SIGNAL] Attempting iron condor for {symbol} at ${price:.2f}")

                            # --- Iron Condor Selection ---
                            # --- Iron Condor Selection (Hybrid, production-ready) ---
                            try:
                                today = datetime.now(self.timezone).date()
                                start_date = today + timedelta(days=5)
                                end_date = today + timedelta(days=7)

                                self.logger.debug(
                                    f"[EXPIRY RANGE] Fetching options for {symbol} between {start_date} and {end_date}")

                                # --- Fetch PUT and CALL options ---
                                put_options = self.get_options(symbol, price * 0.9, price * 1.1, start_date, end_date,
                                                               ContractType.PUT)
                                call_options = self.get_options(symbol, price * 0.9, price * 1.1, start_date, end_date,
                                                                ContractType.CALL)

                                self.logger.info(
                                    f"[CHAIN] Retrieved {len(put_options) if put_options is not None else 0} raw PUT options, "
                                    f"{len(call_options) if call_options is not None else 0} raw CALL options for {symbol}"
                                )

                                if not put_options or not call_options:
                                    self.logger.info(f"[CHAIN] Insufficient options retrieved for {symbol}")
                                    return False

                                # --- Expiry discovery using BOTH puts & calls (more correct for asymmetric chains) ---
                                expiries = sorted(set(
                                    self.build_option_dict(opt, price, self.risk_free_rate).get("expiration_date")
                                    for opt in (put_options + call_options) if opt
                                ))

                                self.logger.info(f"[EXPIRIES] Found {len(expiries)} expiry dates: {expiries}")

                                # --- Group ALL options by expiry for logging/resiliency ---
                                all_puts = {}
                                all_calls = {}
                                for expiry in expiries:
                                    expiry_norm = expiry.date() if hasattr(expiry, "date") else expiry

                                    puts_for_exp = [
                                        opt for opt in put_options
                                        if ((opt.expiration_date.date() if hasattr(opt.expiration_date,
                                                                                   "date") else opt.expiration_date)
                                            == expiry_norm)
                                    ]
                                    calls_for_exp = [
                                        opt for opt in call_options
                                        if ((opt.expiration_date.date() if hasattr(opt.expiration_date,
                                                                                   "date") else opt.expiration_date)
                                            == expiry_norm)
                                    ]

                                    self.logger.info(
                                        f"[EXPIRY CHECK] {symbol} expiry {expiry_norm}: {len(puts_for_exp)} PUTs, {len(calls_for_exp)} CALLs available")

                                    if puts_for_exp and calls_for_exp:
                                        all_puts[expiry_norm] = puts_for_exp
                                        all_calls[expiry_norm] = calls_for_exp

                                # --- Pick nearest expiry in 5–7 DTE, fallback to earliest available if none in-range ---
                                chosen_expiry = self.get_expiry_within_range(expiries, min_dte=5, max_dte=7)
                                if not chosen_expiry:
                                    self.logger.warning(f"[EXPIRY] No valid expiries 5–7 DTE for {symbol}")
                                    if expiries:
                                        chosen_expiry = expiries[0]
                                        self.logger.warning(
                                            f"[EXPIRY] Falling back to earliest available expiry {chosen_expiry.date() if hasattr(chosen_expiry, 'date') else chosen_expiry}"
                                        )
                                    else:
                                        self.logger.warning(f"[EXPIRY] No expiries at all for {symbol}")
                                        return False

                                expiry_date_norm = chosen_expiry.date() if hasattr(chosen_expiry,
                                                                                   "date") else chosen_expiry

                                # --- Pull grouped options for chosen expiry ---
                                puts = all_puts.get(expiry_date_norm, [])
                                calls = all_calls.get(expiry_date_norm, [])

                                if puts and calls:
                                    self.logger.info(
                                        f"[CHAIN] Retrieved {len(puts)} PUTs and {len(calls)} CALLs for {symbol} expiring {expiry_date_norm}")
                                else:
                                    self.logger.warning(
                                        f"[CHAIN] Skipping {symbol} exp {expiry_date_norm}: insufficient puts({len(puts)})/calls({len(calls)})")
                                    return False

                                # === Iron Condor Selection (strict -> relaxed) ===
                                condor = None
                                criteria_used = None

                                # Backup original criteria to restore later
                                criteria_backup = getattr(self, "criteria", None)

                                criteria_sets = [("STRICT", self.criteria), ("RELAXED", self.relaxed_criteria)]
                                for label, crit in criteria_sets:
                                    # apply candidate criteria
                                    self.criteria = crit
                                    self.logger.info(f"[IRON CONDOR] Trying {label.lower()} criteria for {symbol}")

                                    # evaluate and find condor using only options for chosen expiry
                                    try:
                                        # If you have an evaluate function that mutates options for scoring, call it
                                        if hasattr(self, "evaluate_iron_condor_options"):
                                            self.evaluate_iron_condor_options(put_options=puts, call_options=calls,
                                                                              price=price)

                                        condor = self.find_iron_condor(
                                            put_options=puts,
                                            call_options=calls,
                                            underlying_price=price,
                                            risk_free_rate=self.risk_free_rate,
                                        )
                                    except Exception as e:
                                        self.logger.exception(
                                            f"[IRON CONDOR] Error during find_iron_condor under {label}: {e}")
                                        condor = None

                                    if condor and all(condor):
                                        criteria_used = label
                                        self.logger.info(
                                            f"[IRON CONDOR] Found condor under {label.lower()} criteria for {symbol}")
                                        break

                                # Restore original criteria (avoid side-effects)
                                self.criteria = criteria_backup

                                if not condor or not all(condor):
                                    self.logger.info(
                                        f"[NO IRON CONDOR] No valid setup found for {symbol} (strict + relaxed)")
                                    # log rejection reasons for diagnostics
                                    for reason, count in getattr(self, "rejection_reasons", {}).items():
                                        self.logger.info(f"[REJECTION] {reason}: {count}")
                                    return False

                                # === Final rejection counts (diagnostic) ===
                                for reason, count in getattr(self, "rejection_reasons", {}).items():
                                    self.logger.info(f"[REJECTION] {reason}: {count}")

                                # === Unpack legs ===
                                short_put, long_put, short_call, long_call = condor
                                legs = {
                                    "SHORT_PUT": short_put,
                                    "LONG_PUT": long_put,
                                    "SHORT_CALL": short_call,
                                    "LONG_CALL": long_call,
                                }

                                # -------- Quote validation --------
                                quotes = {}
                                for label, leg in legs.items():
                                    bid = float(leg.get("bid_price", 0.0))
                                    ask = float(leg.get("ask_price", 0.0))

                                    if min(bid, ask) <= 0:
                                        self.logger.info("[REJECTED] %s invalid quotes (non-positive).", label)
                                        return False
                                    if bid > ask:
                                        self.logger.info("[REJECTED] %s inverted quotes (bid > ask).", label)
                                        return False

                                    # store quotes for later economics
                                    quotes[label] = (bid, ask)

                                # Guard against wide spreads (absolute + relative)
                                for label, (bid, ask) in quotes.items():
                                    spr = ask - bid
                                    if spr > max(0.50, 0.20 * max(bid, 1e-6)):
                                        self.logger.info("[REJECTED] %s quotes too wide (abs+rel).", label)
                                        return False

                                # -------- Economics --------
                                sp_strike = float(short_put["strike_price"])
                                lp_strike = float(long_put["strike_price"])
                                sc_strike = float(short_call["strike_price"])
                                lc_strike = float(long_call["strike_price"])

                                put_width = sp_strike - lp_strike
                                call_width = lc_strike - sc_strike
                                if put_width <= 0 or call_width <= 0:
                                    self.logger.info("[REJECTED] Invalid width(s). Put=%.2f, Call=%.2f", put_width,
                                                     call_width)
                                    return False

                                net_credit = max(0.0, quotes["SHORT_PUT"][0] - quotes["LONG_PUT"][1]) + \
                                             max(0.0, quotes["SHORT_CALL"][0] - quotes["LONG_CALL"][1])

                                multiplier = int(short_put.get("multiplier", 100))
                                per_condor_risk = max(put_width, call_width) * multiplier - net_credit * multiplier
                                if per_condor_risk <= 0:
                                    self.logger.info("[REJECTED] Non-positive per-condor risk.")
                                    return False

                                credit_to_risk = net_credit / max(put_width, call_width)

                                # -------- Final filters (criteria-driven) --------
                                try:
                                    delta_put = abs(float(short_put.get("delta", 0.0)))
                                except (TypeError, ValueError):
                                    delta_put = 0.0
                                try:
                                    delta_call = abs(float(short_call.get("delta", 0.0)))
                                except (TypeError, ValueError):
                                    delta_call = 0.0
                                try:
                                    oi_put = int(short_put.get("open_interest", 0))
                                except (TypeError, ValueError):
                                    oi_put = 0
                                try:
                                    oi_call = int(short_call.get("open_interest", 0))
                                except (TypeError, ValueError):
                                    oi_call = 0

                                crit = self.criteria
                                relaxed = self.relaxed_criteria

                                # index ETF list — treat these as liquid (bypass OI)
                                _index_etf_liquid = {"SPY", "QQQ", "IWM", "DIA"}

                                symbol_base = symbol.split()[0].strip().upper()

                                # --- Quote harvesting & mid-calculation ---
                                quotes = {}
                                bad_quote = False
                                for label, leg in (
                                        ("SHORT_PUT", short_put),
                                        ("LONG_PUT", long_put),
                                        ("SHORT_CALL", short_call),
                                        ("LONG_CALL", long_call)
                                ):
                                    try:
                                        bid = float(leg.get("bid_price", 0.0))
                                        ask = float(leg.get("ask_price", 0.0))
                                    except Exception:
                                        bid, ask = 0.0, 0.0

                                    if bid <= 0 or ask <= 0:
                                        self.logger.info("[REJECTED] %s invalid quotes (non-positive).", label)
                                        bad_quote = True
                                        break
                                    if bid > ask:
                                        self.logger.info("[REJECTED] %s inverted quotes (bid > ask).", label)
                                        bad_quote = True
                                        break

                                    mid = (bid + ask) / 2.0
                                    quotes[label] = {"bid": bid, "ask": ask, "mid": mid, "spread": ask - bid}

                                if bad_quote:
                                    return False

                                # execution-safety: reject only if spreads are extremely wide
                                # Strict absolute thresholds & relaxed multipliers below:
                                strict_spread_abs = 0.75
                                relaxed_spread_abs = 1.50
                                strict_spread_rel = 0.30  # relative to mid
                                relaxed_spread_rel = 0.60

                                # gather spread flags
                                wide_spread_labels = []
                                for label, info in quotes.items():
                                    spr = info["spread"]
                                    mid = info["mid"]
                                    if spr > strict_spread_abs and spr > strict_spread_rel * max(mid, 1e-6):
                                        wide_spread_labels.append(label)

                                # compute mid-based credit & CRR (less noisy than bid-only)
                                mid_credit = max(0.0, quotes["SHORT_PUT"]["mid"] - quotes["LONG_PUT"]["mid"]) + \
                                             max(0.0, quotes["SHORT_CALL"]["mid"] - quotes["LONG_CALL"]["mid"])

                                # compute best-case immediate credit using bid-ask (conservative)
                                bidask_credit = max(0.0, quotes["SHORT_PUT"]["bid"] - quotes["LONG_PUT"]["ask"]) + \
                                                max(0.0, quotes["SHORT_CALL"]["bid"] - quotes["LONG_CALL"]["ask"])

                                multiplier = int(short_put.get("multiplier", 100))
                                put_width = float(short_put["strike_price"]) - float(long_put["strike_price"])
                                call_width = float(long_call["strike_price"]) - float(short_call["strike_price"])
                                max_width = max(put_width, call_width)
                                if put_width <= 0 or call_width <= 0 or max_width <= 0:
                                    self.logger.info("[REJECTED] Invalid widths. Put=%.2f Call=%.2f", put_width,
                                                     call_width)
                                    return False

                                per_condor_risk = max_width * multiplier - mid_credit * multiplier
                                if per_condor_risk <= 0:
                                    self.logger.info("[REJECTED] Non-positive per-condor risk.")
                                    return False

                                credit_to_risk_mid = mid_credit / max_width
                                credit_to_risk_bidask = bidask_credit / max_width

                                # --- OI logic with ETF bypass ---
                                def oi_ok(threshold, oi_val):
                                    return symbol_base in _index_etf_liquid or oi_val >= threshold

                                # --- Strict pass check (uses conservative bidask credit but also requires mid_credit sanity) ---
                                strict_pass = (
                                        crit['short_put']['min_delta'] <= delta_put <= crit['short_put'][
                                    'max_delta'] and
                                        crit['short_call']['min_delta'] <= delta_call <= crit['short_call'][
                                            'max_delta'] and
                                        oi_ok(crit['short_put']['min_oi'], oi_put) and
                                        oi_ok(crit['short_call']['min_oi'], oi_call) and
                                        credit_to_risk_bidask >= crit['min_credit_to_risk'] and
                                        len(wide_spread_labels) == 0  # no wide spreads for strict
                                )

                                # --- Relaxed pass check ---
                                relaxed_pass = (
                                        relaxed['short_put']['min_delta'] <= delta_put <= relaxed['short_put'][
                                    'max_delta'] and
                                        relaxed['short_call']['min_delta'] <= delta_call <= relaxed['short_call'][
                                            'max_delta'] and
                                        oi_ok(relaxed['short_put']['min_oi'], oi_put) and
                                        oi_ok(relaxed['short_call']['min_oi'], oi_call) and
                                        credit_to_risk_mid >= relaxed['min_credit_to_risk']
                                # allow mid-based acceptance
                                )

                                # If neither strict nor relaxed pass, check midpoint-override: allow if bidask credit fails but mid_credit is good and spreads are not catastrophically wide
                                midpoint_override = False
                                if not strict_pass and not relaxed_pass:
                                    # allow midpoint override only if spreads are within the relaxed bounds (not extremely wide)
                                    too_wide_relaxed = any(
                                        (info["spread"] > relaxed_spread_abs and info[
                                            "spread"] > relaxed_spread_rel * max(info["mid"], 1e-6))
                                        for info in quotes.values()
                                    )
                                    if (not too_wide_relaxed) and (credit_to_risk_mid >= relaxed['min_credit_to_risk']):
                                        midpoint_override = True
                                        self.logger.info(
                                            "[OVERRIDE] Using midpoint economics: mid_credit=%.4f CRR_mid=%.4f",
                                            mid_credit, credit_to_risk_mid)

                                if strict_pass:
                                    self.logger.info(
                                        "[ACCEPTED:STRICT] PUT(%s/%s Δ=%.2f OI=%d) CALL(%s/%s Δ=%.2f OI=%d) NetCredit_bidask=%.2f CRR_bidask=%.2f",
                                        sp_strike, lp_strike, delta_put, oi_put,
                                        sc_strike, lc_strike, delta_call, oi_call,
                                        bidask_credit, credit_to_risk_bidask
                                    )
                                elif relaxed_pass or midpoint_override:
                                    self.logger.info(
                                        "[ACCEPTED:RELAXED] PUT(%s/%s Δ=%.2f OI=%d) CALL(%s/%s Δ=%.2f OI=%d) NetCredit_mid=%.2f CRR_mid=%.2f",
                                        sp_strike, lp_strike, delta_put, oi_put,
                                        sc_strike, lc_strike, delta_call, oi_call,
                                        mid_credit, credit_to_risk_mid
                                    )
                                else:
                                    self.logger.info(
                                        "[REJECTED] Failed filters "
                                        "(PUT Δ=%.2f OI=%d, CALL Δ=%.2f OI=%d, CRR_mid=%.2f, CRR_bidask=%.2f) wide_spread_labels=%s",
                                        delta_put, oi_put, delta_call, oi_call, credit_to_risk_mid,
                                        credit_to_risk_bidask, wide_spread_labels
                                    )
                                    return False

                                # -------- Account & sizing --------
                                try:
                                    account = self.trading_client.get_account()
                                except Exception as e:
                                    self.logger.error("[ERROR] get_account failed: %s", e)
                                    return False

                                portfolio = float(account.portfolio_value or 0.0)
                                buying_power = float(account.buying_power or 0.0)

                                per_trade_cap = portfolio * 0.20  # Max 20% portfolio per trade
                                max_risk_allowed = buying_power * float(getattr(self, "buy_power_limit", 1.0))

                                def safe_floor(dividend, divisor):
                                    return max(0, int(dividend / divisor)) if divisor > 0 else 0

                                max_qty_bp = safe_floor(buying_power, per_condor_risk)
                                max_qty_cap = max(1, safe_floor(per_trade_cap, per_condor_risk))
                                max_qty_risk = safe_floor(max_risk_allowed, per_condor_risk)

                                qty = max(0, min(max_qty_bp, max_qty_cap, max_qty_risk))

                                self.logger.info(
                                    "[SIZING] risk/condor=%.2f, max_qty_bp=%d, max_qty_cap=%d, max_qty_risk=%d, final_qty=%d",
                                    per_condor_risk, max_qty_bp, max_qty_cap, max_qty_risk, qty
                                )

                                if qty < 1:
                                    self.logger.info("[SKIP] Not enough BP for 1 condor under risk caps.")
                                    return False

                                if per_trade_cap < per_condor_risk:
                                    self.logger.warning(
                                        "[WARNING] Per-trade cap ($%.2f) < risk per condor ($%.2f). Proceeding with 1 if BP permits.",
                                        per_trade_cap, per_condor_risk
                                    )

                                # Tell why we are capped
                                if qty == max_qty_bp:
                                    self.logger.info("[INFO] Position size limited by Buying Power cap.")
                                elif qty == max_qty_cap:
                                    self.logger.info("[INFO] Position size limited by Per-Trade cap (10%% portfolio).")
                                elif qty == max_qty_risk:
                                    self.logger.info("[INFO] Position size limited by Global Risk cap.")
                                else:
                                    self.logger.info("[INFO] Position sizing not capped (rare).")

                                # -------- Final BP check using user-defined function --------
                                if not self.check_buying_power(short_put, long_put, short_call, long_call):
                                    self.logger.info("[REJECTED] check_buying_power failed for qty=%d.", qty)
                                    return False

                                # -------- Prepare & place multi-leg order (limit) --------
                                underlying_symbol = (
                                        short_put.get("underlying_symbol")
                                        or short_put.get("root_symbol")
                                        or short_call.get("underlying_symbol")
                                )
                                if not underlying_symbol:
                                    self.logger.info("[REJECTED] Missing underlying symbol.")
                                    return False

                                tick = (
                                               short_put.get("tick_size")
                                               or long_put.get("tick_size")
                                               or short_call.get("tick_size")
                                               or long_call.get("tick_size")
                                               or 0.01
                                       ) or 0.01

                                # compute conservative target limit based on tick increments and net_credit
                                try:
                                    if tick and tick > 0:
                                        ticks = int(net_credit / float(tick))
                                    else:
                                        ticks = 0
                                except Exception:
                                    ticks = 0

                                target_limit = round(float(max(0.0, ticks * float(tick))), 2)

                                # build leg dicts with sides
                                short_put_leg = dict(short_put)
                                short_put_leg["side"] = "SELL"
                                long_put_leg = dict(long_put)
                                long_put_leg["side"] = "BUY"
                                short_call_leg = dict(short_call)
                                short_call_leg["side"] = "SELL"
                                long_call_leg = dict(long_call)
                                long_call_leg["side"] = "BUY"

                                order = {
                                    "symbol": underlying_symbol,
                                    "short_put": short_put_leg,
                                    "long_put": long_put_leg,
                                    "short_call": short_call_leg,
                                    "long_call": long_call_leg,
                                }

                                try:
                                    res = self.submit_mleg_open(order, total_qty=qty, limit_price=target_limit)
                                except Exception as e:
                                    self.logger.error("[ERROR] submit_mleg_open failed: %s", e, exc_info=True)
                                    return False

                                if res:
                                    self.logger.info(
                                        "[ORDER PLACED - %s] %s IC %dx @ %.2f | put_width=%.2f | call_width=%.2f | credit=%.2f | risk=%.2f",
                                        criteria_used, underlying_symbol, qty, target_limit, put_width, call_width,
                                        net_credit, per_condor_risk
                                    )
                                    ts = datetime.now(timezone.utc).isoformat()
                                    entry = {
                                        "symbol": underlying_symbol,
                                        "underlying_symbol": underlying_symbol,
                                        "short_put": short_put_leg,
                                        "long_put": long_put_leg,
                                        "short_call": short_call_leg,
                                        "long_call": long_call_leg,
                                        "entered_at": ts,
                                        "expiry": str(short_put["expiration_date"].date()),
                                        "qty": qty
                                    }
                                    # persist state
                                    self.active_spreads.append(entry)
                                    self.entry_log[underlying_symbol] = ts
                                    self.save_spread_state(self.active_spreads)
                                    return True
                                else:
                                    self.logger.warning("[ORDER FAILED] submit_mleg_open returned falsy result.")
                                    return False

                            except Exception as e:
                                self.logger.exception(f"[IRON CONDOR] Unexpected error for {symbol}: {e}")
                                # ensure criteria restored if exception bubbled before restore
                                try:
                                    self.criteria = self.criteria
                                except Exception:
                                    pass
                                return False

                    """
                    Exit/roll logic for Iron Condors.
                    """

                    if market_open + timedelta(hours=4) <= ts < market_close - timedelta(minutes=15):

                        # Step 0: Refresh from persisted state and Alpaca
                        self.active_condors = self.sync_with_alpaca()

                        if not self.active_condors:
                            self.logger.info("[EXIT CHECK] No active Iron Condors to evaluate.")
                            return

                        self.logger.info(f"[EXIT CHECK] Evaluating exit for {len(self.active_condors)} condors:")

                        for condor in self.active_condors:
                            self.logger.info(
                                f"   [REVIEW] {condor['symbol']} | "
                                f"SP={condor.get('short_put', {}).get('symbol')} "
                                f"LP={condor.get('long_put', {}).get('symbol')} "
                                f"SC={condor.get('short_call', {}).get('symbol')} "
                                f"LC={condor.get('long_call', {}).get('symbol')}"
                            )

                        # Step 1: Remove condors no longer live in Alpaca
                        try:
                            positions = self.trading_client.get_all_positions()
                            open_option_symbols = {pos.symbol for pos in positions if pos.asset_class == "us_option"}
                        except Exception as e:
                            self.logger.error(f"[EXIT CHECK] Failed to fetch Alpaca positions: {e}")
                            open_option_symbols = set()

                        for condor in list(self.active_condors):
                            symbols = {
                                condor.get("short_put", {}).get("symbol"),
                                condor.get("long_put", {}).get("symbol"),
                                condor.get("short_call", {}).get("symbol"),
                                condor.get("long_call", {}).get("symbol"),
                            }
                            if not any(sym in open_option_symbols for sym in symbols if sym):
                                self.logger.info(f"[EXIT CHECK] Condor closed externally for {condor['symbol']}")
                                self.active_condors.remove(condor)

                        # Step 2: Process remaining condors
                        try:
                            positions = self.trading_client.get_all_positions()
                            option_qtys = {
                                pos.symbol: abs(int(float(pos.qty)))
                                for pos in positions if pos.asset_class == "us_option"
                            }
                        except Exception as e:
                            self.logger.error(f"[EXIT CHECK] Failed to fetch Alpaca positions for exit/roll logic: {e}")
                            option_qtys = {}

                        price = df['close'].iloc[-1]

                        for condor in self.active_condors.copy():
                            sp, lp = condor.get("short_put", {}), condor.get("long_put", {})
                            sc, lc = condor.get("short_call", {}), condor.get("long_call", {})

                            if not (sp and lp and sc and lc):
                                self.logger.warning(f"[EXIT CHECK] Skipping incomplete condor {condor['symbol']}")
                                continue

                            # === Generic condor evaluation function (you must implement this) ===
                            action, condor_update, reason = self.roll_rinse_iron_condor(
                                short_put=sp, long_put=lp,
                                short_call=sc, long_call=lc,
                                underlying_price=price
                            )

                            # Exit handling
                            if action == "HOLD":
                                self.logger.info(
                                    f"[HOLD] {condor['symbol']} | Reason={reason} | "
                                    f"SP={sp['symbol']} | LP={lp['symbol']} | SC={sc['symbol']} | LC={lc['symbol']}"
                                )

                            elif action == "EXIT":
                                qty = min(
                                    option_qtys.get(sp["symbol"], 0),
                                    option_qtys.get(lp["symbol"], 0),
                                    option_qtys.get(sc["symbol"], 0),
                                    option_qtys.get(lc["symbol"], 0),
                                )
                                if qty > 0:
                                    try:
                                        res = self.submit_mleg_close(condor, qty)
                                        self.logger.info(
                                            f"[EXIT] {condor['symbol']} | Reason={reason} | "
                                            f"SP={sp['symbol']} | LP={lp['symbol']} | SC={sc['symbol']} | LC={lc['symbol']} "
                                            f"| qty={qty}, order_id={getattr(res, 'id', 'N/A')}"
                                        )
                                    except Exception as e:
                                        self.logger.error(
                                            f"[EXIT FAILED] {condor['symbol']} | {reason} | {e}"
                                        )
                                else:
                                    self.logger.warning(
                                        f"[EXIT] {condor['symbol']} | Reason={reason} | "
                                        f"No live qty in Alpaca (already closed?)"
                                    )
                                self.active_condors.remove(condor)
                                self.save_spread_state(self.active_condors)

                            elif action == "ROLL":
                                self.logger.info(
                                    f"[ROLL] {condor['symbol']} | Reason={reason} | "
                                    f"SP={sp['symbol']} | LP={lp['symbol']} | SC={sc['symbol']} | LC={lc['symbol']}"
                                )
                                self.active_condors.remove(condor)
                                self.active_condors.append(condor_update)
                                self.save_spread_state(self.active_condors)

                            elif action == "ERROR":
                                self.logger.warning(
                                    f"[ERROR] {condor['symbol']} | Reason={reason} | "
                                    f"SP={sp['symbol']} | LP={lp['symbol']} | SC={sc['symbol']} | LC={lc['symbol']}"
                                )

                    # Portfolio-level exit check (AFTER all condor-level exits/rolls)
                    if ts.weekday() == 4 and ts >= market_close - timedelta(minutes=30):
                        try:
                            account = self.trading_client.get_account()
                            equity = float(account.equity)  # current equity reported by Alpaca
                            last_equity = float(account.last_equity)  # yesterday's equity (Alpaca-provided)

                            # Method A (Alpaca's own baseline)
                            alpaca_pct = ((equity - last_equity) / last_equity) * 100 if last_equity > 0 else 0.0

                            # Method B (Custom baseline) - define once at strategy startup
                            if not hasattr(self, "base_equity"):
                                self.base_equity = equity  # store initial equity at strategy launch
                            custom_pct = ((
                                                  equity - self.base_equity) / self.base_equity) * 100 if self.base_equity > 0 else 0.0

                            self.logger.info(
                                f"[PORTFOLIO CHECK] Equity={equity:.2f}, "
                                f"Prev={last_equity:.2f}, Change(Alpaca)={alpaca_pct:.2f}%, "
                                f"Change(Custom)={custom_pct:.2f}%"
                            )

                            # Use custom percentage as the true trigger
                            if custom_pct >= 5.0:  # configurable
                                self.logger.info(
                                    f"[PORTFOLIO EXIT] Closing ALL positions (Custom Profit {custom_pct:.2f}%)")

                                try:
                                    positions = self.trading_client.get_all_positions()
                                    for pos in positions:
                                        try:
                                            self.trading_client.close_position(pos.symbol)
                                            self.logger.info(f"[PORTFOLIO EXIT] Closed {pos.symbol}, qty={pos.qty}")
                                        except Exception as e:
                                            self.logger.error(f"[PORTFOLIO EXIT FAILED] {pos.symbol} | {e}")

                                    self.active_condors.clear()
                                    self.save_spread_state(self.active_condors)

                                except Exception as e:
                                    self.logger.error(f"[PORTFOLIO EXIT FAILED] Could not close positions: {e}")

                        except Exception as e:
                            self.logger.error(f"[PORTFOLIO CHECK FAILED] {e}")


                except Exception as e:
                    print(f"[ERROR] Exception in handle_second_bar: {e}")
                    self.logger.error("[TRACEBACK]\n" + traceback.format_exc())

            self.wss_client.subscribe_bars(handle_second_bar, *symbols)
            self.wss_client.run()

        trading_thread = threading.Thread(target=run_trading_stream)
        wss_thread = threading.Thread(target=run_wss_client)

        trading_thread.start()
        wss_thread.start()

        trading_thread.join()
        wss_thread.join()


if __name__ == "__main__":

    # Instantiate your strategy class
    strategy = IronCondorStrategy()

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