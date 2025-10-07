import pandas as pd
import numpy as np
from scipy.stats import norm
from ta.trend import MACD, ADXIndicator
import time
import os, json

from scipy.optimize import brentq
from datetime import date, datetime, timedelta, timezone

from zoneinfo import ZoneInfo
from typing import TypedDict, Optional, Dict, Any, List, Tuple
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
logger = logging.getLogger("bull_put_strategy")
logging.basicConfig(level=logging.INFO)

API_KEY = "PKYDUEDWNH370ZCXYNQY"
SECRET_KEY = "1AMVlgOxXR5j6u1RBjQMrdtkswCIcyNMYlPW6A4c"
PEPER_URL = "https://paper-api.alpaca.markets/v2"
PAPER = True

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

class SpreadData(TypedDict):
    symbol: str
    short_put: Dict[str, Any]
    long_put: Dict[str, Any]
    entered_at: str
    expiry: str
    qty: Optional[float]


class BullPutSpread:
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
        # self.logger.setLevel(logging.DEBUG)
        self.logger.debug("BullPutSpread initialized.")

        self.underlying_assets = [
            "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLI", "XLV", "SMH", "XLE",
            "EEM", "FXI", "GLD", "GDX", "EFA", "EWZ", "TLT", "EWC", "KWEB", "ARKK",
            "TQQQ", "LQD", "BKLN"
        ]

        # === Strategy Parameters ===
        self.strike_range = 0.05  # 5% above/below underlying — better suited for weeklies
        self.buy_power_limit = 0.30  # Use up to 30% of account buying power per spread
        self.risk_free_rate = self.get_risk_free_rate()  # Used in BS model or IV rank calcs
        self.target_profit_percent = 0.35  # 35% profit target suits short hold period
        self.delta_stop_loss = 0.5  # Exit early if short put delta exceeds 0.50
        self.vega_stop_loss_percent = 0.2  # Exit if IV increases more than 20% from entry

        # === Buying Power ===
        self.buying_power = float(self.trading_client.get_account().buying_power)
        self.buying_power_limit = self.buying_power * self.buy_power_limit

        # === Core Spread Selection Criteria ===
        self.criteria = {
            'short_put': {
                'min_delta': -0.35,  # Avoid too deep OTM
                'max_delta': -0.10,  # Moderate premium + high probability OTM
                'min_iv': 0.15,  # Filter for decent premium
                'max_iv': 1.0,
                'min_oi': 25,  # Ensure liquidity
                'max_bid_ask_spread': 0.25,  # Avoid slippage
            },
            'long_put': {
                'min_delta': -0.95,  # Deep OTM to reduce premium cost
                'max_delta': -0.20,  # Not too close to short leg
                'min_iv': 0.10,
                'max_iv': 1.0,
                'min_oi': 25,
                'max_bid_ask_spread': 0.30,
            },
            'min_days_to_exp': 8,  # Target next week's expiry for 5-day hold
            'max_days_to_exp': 14,
            'max_spread_width': 5.0,  # Keep risk defined and capital-efficient
            'max_otm_distance': 10.0  # Avoid strikes too far from underlying
        }

        self.relaxed_criteria = {
            'short_put': {
                'min_delta': -0.45,  # Slightly more aggressive fallback
                'max_delta': -0.10,
                'min_iv': 0.08,  # Allow lower IV environments
                'max_iv': 1.2,
                'min_oi': 10,
                'max_bid_ask_spread': 0.35,
            },
            'long_put': {
                'min_delta': -0.95,
                'max_delta': -0.10,
                'min_iv': 0.05,
                'max_iv': 1.2,
                'min_oi': 10,
                'max_bid_ask_spread': 0.35,
            },
            'min_days_to_exp': 7,
            'max_days_to_exp': 21,
            'max_spread_width': 7.0,
            'max_otm_distance': 15.0
        }

        # === Internal State ===
        self.minute_history: Dict[str, pd.DataFrame] = {}
        self.active_spreads = []
        self.symbols_with_open_options = set()
        self.rejection_reasons = defaultdict(int)
        self.last_checked: Dict[str, int] = {}

        # === Time-to-Expiration Logic ===
        self.min_dte = 5  # Minimum DTE for entry
        self.max_dte = 55  # Maximum DTE for entry
        self.holding_period_days = 5  # Days to hold the position before exit evaluation

        self.logger.debug(f"BullPutSpread initialized on {self.today} with buying power ${self.buying_power:,.2f}.")

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

    def get_expiry_within_range(self, expiries, min_dte=8, max_dte=14, today=None):
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
        expiration = pd.Timestamp(expiration, tz='UTC') if expiration.tzinfo is None else expiration.tz_convert('UTC')
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

    # ------------------------------------------------------------------------------
    # Asset Selection Based on Technical and IV Criteria
    # ------------------------------------------------------------------------------
    def get_underlying_assets(self) -> List[str]:
        """
        Select all bullish index ETFs for Bull Put Spread strategy using two-tier filtering:
        - Primary: Trend + Momentum + Strength + Volume + IV Rank
        - Fallback: Open > 20-day MA (momentum proxy) if none are found
        """
        self.logger.info("[SCREEN] Starting Bull Put Spread asset selection...")

        assets = self.trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
        symbols = [a.symbol for a in assets if a.tradable and a.symbol in self.underlying_assets]

        # Historical data window
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

        # === Step 4: Scoring + Filtering ===
        for symbol in symbols:
            df = bars[bars['symbol'] == symbol].copy().set_index('timestamp')
            if len(df) < 50:
                self.logger.debug(f"[FILTER] {symbol}: Skipped (insufficient history)")
                continue

            try:
                latest = df.iloc[-1]
                score = 0
                reasons = []

                # --- Score-based Filtering ---
                if latest['close'] > latest['20_MA'] > latest['50_MA']:
                    score += 1
                else:
                    reasons.append("trend_fail")

                if latest['MACD_Hist'] > 0:
                    score += 1
                else:
                    reasons.append("macd_fail")

                if 45 < latest['RSI'] < 75:
                    score += 1
                else:
                    reasons.append("rsi_fail")

                if latest['ADX'] > 20:
                    score += 1
                else:
                    reasons.append("adx_fail")

                if latest['volume'] > latest['Volume_SMA20']:
                    score += 1
                else:
                    reasons.append("volume_fail")

                # Pass threshold
                if score >= 3:
                    close_price = latest['close']
                    option_price = close_price * 0.05

                    iv = self.calculate_implied_volatility(
                        option_price, close_price, close_price, 5 / 252,  # 5-day hold
                        self.risk_free_rate, 'put'
                    )

                    if iv is not None and iv > 0.15:
                        selected.append((symbol, iv))
                        self.logger.info(
                            f"[SELECTED] {symbol} | Score={score}/5 | IV={iv:.3f} | "
                            f"Close={close_price:.2f}, RSI={latest['RSI']:.1f}, ADX={latest['ADX']:.1f}, MACD_Hist={latest['MACD_Hist']:.3f}"
                        )
                    else:
                        self.logger.debug(f"[REJECTED] {symbol}: IV too low or invalid (IV={iv})")
                else:
                    self.logger.debug(f"[REJECTED] {symbol} | Score={score}/5 | Reasons: {', '.join(reasons)}")

            except Exception as e:
                self.logger.warning(f"[ERROR] Screening failed for {symbol}: {e}")
                continue

        selected_symbols = [s for s, _ in sorted(selected, key=lambda x: x[1], reverse=True)]

        # === Fallback ===
        if len(selected_symbols) == 0:
            self.logger.warning("[FALLBACK] No assets passed primary filters. Switching to momentum fallback.")
            fallback_symbols = self.get_20day_ma_assets()
            if fallback_symbols:
                self.logger.info(f"[FALLBACK USED] Momentum screen selected: {fallback_symbols}")
            else:
                self.logger.warning("[FALLBACK] No symbols passed fallback screen either.")
            selected_symbols.extend(fallback_symbols)

        return selected_symbols

    def get_20day_ma_assets(self) -> List[str]:
        """Momentum-based fallback: open > 20-day SMA."""
        self.logger.info("[FALLBACK] Running momentum-based filter (open > 20-day MA)...")

        stock_historical_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        assets = self.trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        symbols = [a.symbol for a in assets if a.tradable and a.symbol in self.underlying_assets]

        start_dt = (datetime.now(ZoneInfo('America/New_York')) - timedelta(days=60)).date()
        end_dt = datetime.now(ZoneInfo('America/New_York')).date()

        bars = stock_historical_data_client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_dt, end=end_dt)
        ).df

        if bars.empty:
            self.logger.warning("[FALLBACK] No historical bars retrieved.")
            return []

        if isinstance(bars.index, pd.MultiIndex):
            bars.reset_index(inplace=True)

        bars['20_day_ma'] = bars.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
        latest_bars = bars.sort_values('timestamp').groupby('symbol').tail(1)

        momentum_stocks = latest_bars[latest_bars['open'] > latest_bars['20_day_ma']]
        tickers = momentum_stocks['symbol'].unique().tolist()

        for symbol in latest_bars['symbol'].unique():
            if symbol in tickers:
                self.logger.debug(f"[FALLBACK PASS] {symbol}: open > 20-day MA")
            else:
                self.logger.debug(f"[FALLBACK FAIL] {symbol}: open <= 20-day MA")

        return tickers

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
    def calculate_option_metrics(
            self,
            option_data: dict,
            underlying_price: float,
            risk_free_rate: float
    ) -> dict:
        """
        Calculate option price, IV, and Greeks using mid-price and BSM model.
        """
        symbol = option_data.get('symbol')
        expiration = pd.Timestamp(option_data.get('expiration_date'))

        try:
            quote_req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.option_historical_data_client.get_option_latest_quote(quote_req)[symbol]
            bid = self.safe_float(quote.bid_price)
            ask = self.safe_float(quote.ask_price)

            if bid == 0.0 and ask == 0.0:
                self.logger.warning(f"[QUOTE MISSING] {symbol} skipped due to bid=0 and ask=0.")
                return {
                    "option_price": 0.0,
                    "expiration_date": expiration,
                    "remaining_days": 0,
                    "iv": 0.0,
                    "delta": 0.0,
                    "gamma": 0.0,
                    "theta": 0.0,
                    "vega": 0.0
                }

            if bid > 0 and ask > 0:
                mid_price = (bid + ask) / 2.0
            elif bid > 0:
                mid_price = bid
            elif ask > 0:
                mid_price = ask
            else:
                mid_price = 0.0

        except Exception as e:
            self.logger.error(f"[QUOTE ERROR] {symbol}: {e}")
            return {
                "option_price": 0.0,
                "expiration_date": expiration,
                "remaining_days": 0,
                "iv": 0.0,
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0
            }

        days_to_expiry = max((expiration - pd.Timestamp.now()).days, 1)
        time_to_expiry = days_to_expiry / 252  # trading days in year

        option_type = option_data.get('type').value.lower()

        # Step 1: Calculate Implied Volatility
        try:
            iv = self.calculate_implied_volatility(
                option_price=mid_price,
                S=underlying_price,
                K=float(option_data['strike_price']),
                T=time_to_expiry,
                r=risk_free_rate,
                option_type=option_type
            )
        except Exception as e:
            self.logger.warning(f"[IV ERROR] {symbol}: {e}. Using fallback IV=0.2")
            iv = 0.2

        # Step 2: Calculate Greeks
        try:
            delta, gamma, theta, vega = self.calculate_greeks(
                option_price=mid_price,
                strike_price=float(option_data['strike_price']),
                expiration=expiration,
                underlying_price=underlying_price,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                IV=iv
            )
        except Exception as e:
            self.logger.warning(f"[GREEKS ERROR] {symbol}: {e}")
            delta, gamma, theta, vega = 0.0, 0.0, 0.0, 0.0

        return {
            "option_price": mid_price,
            "expiration_date": expiration,
            "remaining_days": days_to_expiry,
            "iv": iv,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega
        }

    # ------------------------------------------------------------------------------
    # Unified Option Dictionary Constructor
    # ------------------------------------------------------------------------------

    def safe_float(self,val, default=0.0):
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def build_option_dict(
            self,
            option_data,
            underlying_price: float,
            risk_free_rate: float
    ) -> dict:
        option_data = self.ensure_dict(option_data)
        symbol = option_data["symbol"]

        # Get Greeks & metrics from Black-Scholes
        metrics = self.calculate_option_metrics(option_data, underlying_price, risk_free_rate)

        bid_price, ask_price, mid_price = 0.0, 0.0, 0.0

        try:
            quote_req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.option_historical_data_client.get_option_latest_quote(quote_req)[symbol]
            bid_price = self.safe_float(quote.bid_price)
            ask_price = self.safe_float(quote.ask_price)

            if bid_price == 0.0 and ask_price == 0.0:
                self.logger.warning(f"[MISSING QUOTE] Bid/Ask both 0.0 for {symbol}")
            else:
                mid_price = (bid_price + ask_price) / 2.0
        except Exception as e:
            self.logger.warning(f"[QUOTE ERROR] Failed to fetch bid/ask for {symbol}: {e}")

        try:
            open_interest = int(option_data.get('open_interest') or 0)
        except (TypeError, ValueError):
            self.logger.debug(f"[OI] Invalid open_interest for {symbol}. Defaulting to 0.")
            open_interest = 0

        # Extract Greeks from metrics
        iv = self.safe_float(metrics.get('iv'))
        delta = self.safe_float(metrics.get('delta'))
        gamma = self.safe_float(metrics.get('gamma'))
        theta = self.safe_float(metrics.get('theta'))
        vega = self.safe_float(metrics.get('vega'))
        option_price = self.safe_float(metrics.get('option_price'))

        if iv == 0.0:
            self.logger.warning(f"[IV] Zero IV for {symbol}")
        if delta == 0.0:
            self.logger.warning(f"[DELTA] Zero delta for {symbol}")

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
            'remaining_days': metrics['remaining_days'],
            'open_interest': open_interest,
            'open_interest_date': option_data.get('open_interest_date'),
            'size': int(option_data.get('size') or 1),
            'status': option_data.get('status'),
            'style': option_data.get('style'),
            'tradable': option_data.get('tradable'),
            'type': option_data.get('type'),
            'bid_price': bid_price,
            'ask_price': ask_price,
            'mid_price': mid_price,
            'implied_volatility': iv,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'option_price': option_price,
            'initial_option_price': mid_price
        }
        return candidate

    def check_option_conditions(self, candidate: dict, label: str, criteria: dict) -> bool:
        """
        Optimized: Validate option against expiration, delta, IV, OI, liquidity, and bid/ask spread.
        - Reduced redundant type conversions and dict lookups
        - Early exits for failures to avoid unnecessary checks
        - Compact but still logs reasons
        """

        leg_key = {"SHORT": "short_put", "LONG": "long_put"}.get(label)
        if not leg_key:
            self.logger.error(f"[CRITERIA ERROR] Invalid label '{label}'")
            return False

        leg_criteria = criteria.get(leg_key, {})
        symbol = candidate.get("symbol", "unknown")

        # Extract and sanitize core values
        bid, ask = float(candidate.get("bid_price", 0.0)), float(candidate.get("ask_price", 0.0))
        spread = ask - bid
        delta = candidate.get("delta")
        delta = float(delta) if delta not in (None, "") else None
        iv = float(candidate.get("implied_volatility") or 0.0)
        oi = int(candidate.get("open_interest") or 0)
        dte = int(candidate.get("remaining_days") or 0)

        # --- 1. Liquidity ---
        if bid <= 0 or ask <= 0:
            self.logger.debug(f"[REJECT] {label} {symbol} illiquid (bid={bid}, ask={ask})")
            self.rejection_reasons[f"{label}_illiquid"] += 1
            return False

        # --- 2. Bid-ask spread ---
        if spread > leg_criteria.get("max_bid_ask_spread", 0.5):
            self.logger.debug(f"[REJECT] {label} {symbol} spread {spread:.2f} too wide")
            self.rejection_reasons[f"{label}_spread"] += 1
            return False

        # --- 3. Delta ---
        if delta is None:
            self.logger.debug(f"[REJECT] {label} {symbol} missing delta")
            self.rejection_reasons[f"{label}_delta_missing"] += 1
            return False

        if delta < leg_criteria.get("min_delta", float("-inf")):
            self.logger.debug(f"[REJECT] {label} {symbol} delta {delta:.2f} too low")
            self.rejection_reasons[f"{label}_delta_too_low"] += 1
            return False

        if delta > leg_criteria.get("max_delta", float("inf")):
            self.logger.debug(f"[REJECT] {label} {symbol} delta {delta:.2f} too high")
            self.rejection_reasons[f"{label}_delta_too_high"] += 1
            return False

        # --- 4. IV ---
        if iv <= 0:
            self.logger.debug(f"[REJECT] {label} {symbol} IV {iv} non-positive")
            self.rejection_reasons[f"{label}_iv"] += 1
            return False

        if iv < leg_criteria.get("min_iv", 0):
            self.logger.debug(f"[REJECT] {label} {symbol} IV {iv:.2f} too low")
            self.rejection_reasons[f"{label}_iv_too_low"] += 1
            return False

        if iv > leg_criteria.get("max_iv", float("inf")):
            self.logger.debug(f"[REJECT] {label} {symbol} IV {iv:.2f} too high")
            self.rejection_reasons[f"{label}_iv_too_high"] += 1
            return False

        # --- 5. OI ---
        if oi < leg_criteria.get("min_oi", 0):
            self.logger.debug(f"[REJECT] {label} {symbol} OI {oi} too low")
            self.rejection_reasons[f"{label}_oi"] += 1
            return False

        # --- 6. DTE ---
        if not (criteria.get("min_days_to_exp", 0) <= dte <= criteria.get("max_days_to_exp", 999)):
            if dte < criteria.get("min_days_to_exp", 0):
                self.logger.debug(f"[REJECT] {label} {symbol} DTE {dte} too short")
                self.rejection_reasons[f"{label}_dte_too_short"] += 1
            else:
                self.logger.debug(f"[REJECT] {label} {symbol} DTE {dte} too long")
                self.rejection_reasons[f"{label}_dte_too_long"] += 1
            return False

        return True

    # ---------------------------------------------------------------------------
    # Retrieve viable OTM puts with valid Greeks, IV, and OI
    # ---------------------------------------------------------------------------
    def find_bull_put_spread(
            self,
            put_options: List[Any],
            underlying_price: float,
            risk_free_rate: float
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Optimized bull put spread finder:
        - Prebuild candidates in one pass
        - Store SHORT/LONG by expiration efficiently
        - Avoid O(N^2) explosion by pairing only nearest valid long strikes
        - Fail-fast skips for invalid structures
        """

        from bisect import bisect_left

        valid_candidates = []
        for opt in put_options:
            try:
                valid_candidates.append(
                    self.build_option_dict(opt, underlying_price, risk_free_rate)
                )
            except Exception as e:
                self.logger.warning(f"[ERROR] Build failed for {getattr(opt, 'symbol', 'unknown')}: {e}")

        sp_by_exp, lp_by_exp = defaultdict(list), defaultdict(list)

        for c in valid_candidates:
            exp = c["expiration_date"]
            if self.check_option_conditions(c, "SHORT", self.criteria):
                sp_by_exp[exp].append(c)
            if self.check_option_conditions(c, "LONG", self.criteria):
                lp_by_exp[exp].append(c)

        best_spread, best_score = None, float("-inf")

        for exp in sp_by_exp.keys() & lp_by_exp.keys():
            shorts = sorted(sp_by_exp[exp], key=lambda x: float(x["strike_price"]), reverse=True)
            longs = sorted(lp_by_exp[exp], key=lambda x: float(x["strike_price"]))

            # For each short, only consider the *closest* valid long strikes below it
            long_strikes = [float(l["strike_price"]) for l in longs]

            for sp in shorts:
                s_strike = float(sp["strike_price"])
                idx = bisect_left(long_strikes, s_strike)  # longs below short strike
                for lp in longs[:idx][::-1]:  # walk backwards (closest strikes first)
                    l_strike = float(lp["strike_price"])

                    # Spread width constraint
                    if s_strike - l_strike > self.criteria["max_spread_width"]:
                        continue

                    # Distance constraint
                    if l_strike < s_strike - self.criteria.get("max_otm_distance", 5):
                        continue

                    # Risk / BP check
                    if not self.check_buying_power(sp, lp):
                        continue

                    try:
                        score = self.score_spread(sp, lp)
                        if score > best_score:
                            best_score, best_spread = score, (sp, lp)
                    except Exception as e:
                        self.logger.warning(f"[SCORING ERROR] {e}")

                    # Since longs are sorted closest first, break after first good match
                    break

        if best_spread:
            sp, lp = best_spread
            self.logger.info(
                f"[PAIR SELECTED] Short={sp['symbol']}@{sp['strike_price']} | "
                f"Long={lp['symbol']}@{lp['strike_price']} | Score={best_score:.4f}"
            )
            return sp, lp

        self.logger.info("[SPREAD FAILED] No valid bull put spread found.")
        for reason, count in self.rejection_reasons.items():
            self.logger.info(f"    {reason}: {count}")
        return None, None

    # ---------------------------------------------------------------------------
    # Filter options using strategy-defined criteria
    # ---------------------------------------------------------------------------
    def score_spread(self, short_put: dict, long_put: dict) -> float:
        """
        Score a bull put spread based on net credit, max risk, and optional Greeks.
        """
        try:
            short_bid = float(short_put.get("bid_price", 0.0))
            short_ask = float(short_put.get("ask_price", 0.0))
            long_bid = float(long_put.get("bid_price", 0.0))
            long_ask = float(long_put.get("ask_price", 0.0))

            # Mid prices
            short_mid = (short_bid + short_ask) / 2
            long_mid = (long_bid + long_ask) / 2

            net_credit = short_mid - long_mid
            spread_width = float(long_put["strike_price"]) - float(short_put["strike_price"])
            max_loss = spread_width - net_credit

            if max_loss <= 0 or net_credit <= 0:
                return 0.0

            reward_risk = net_credit / max_loss

            # Optional: Factor in theta (time decay benefit)
            short_theta = float(short_put.get("theta", 0.0))
            long_theta = float(long_put.get("theta", 0.0))
            net_theta = short_theta - long_theta  # net decay

            # Optional: Normalize bid/ask spreads and OI
            sp_liquidity = short_bid > 0 and short_ask > 0
            lp_liquidity = long_bid > 0 and long_ask > 0
            sp_oi = float(short_put.get("open_interest", 0))
            lp_oi = float(long_put.get("open_interest", 0))

            # Base score
            score = reward_risk

            # Adjust for theta benefit
            if net_theta > 0:
                score *= (1 + 0.10 * net_theta)  # modest boost for positive theta

            # Penalize illiquidity
            if not sp_liquidity or not lp_liquidity:
                score *= 0.5  # halve score if either leg is illiquid

            # Penalize low open interest
            if sp_oi < 50 or lp_oi < 50:
                score *= 0.7  # apply 30% penalty

            self.logger.debug(
                f"[SPREAD SCORE] Short={short_put['symbol']}, Long={long_put['symbol']} | "
                f"NetCredit={net_credit:.2f}, MaxLoss={max_loss:.2f}, RewardRisk={reward_risk:.2f}, "
                f"Theta={net_theta:.2f}, OI(S/L)={sp_oi}/{lp_oi}, Score={score:.4f}"
            )

            return round(score, 4)

        except Exception as e:
            self.logger.warning(f"[SCORE ERROR] Failed to score spread: {e}")
            return 0.0

    def check_buying_power(self, short_put: Dict[str, Any], long_put: Dict[str, Any]) -> bool:
        """
        Check if the bull put spread fits within the account's buying power.
        Returns True if within acceptable risk limits.
        """
        try:
            # === Account Info ===
            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)
            portfolio_value = float(account.portfolio_value)

            # === Log buying power info ===
            self.logger.info(f"[ACCOUNT] Buying Power: ${buying_power:,.2f} | Portfolio Value: ${portfolio_value:,.2f}")

            # === Risk Limit Calculation ===
            buy_power_limit = buying_power * self.buy_power_limit  # e.g. 5% of buying power
            self.logger.info(f"[ACCOUNT] Buying Power Limit: ${buy_power_limit:,.2f}")

            # === Option Details ===
            option_size = float(short_put.get('size', 100))  # default to 100
            short_strike = float(short_put['strike_price'])
            long_strike = float(long_put['strike_price'])
            short_bid = float(short_put['bid_price'])
            long_ask = float(long_put['ask_price'])

            if short_strike <= long_strike:
                self.logger.warning("[BPS ERROR] Invalid spread: short strike must be higher than long strike.")
                return False

            spread_width = short_strike - long_strike
            net_credit = short_bid - long_ask
            max_loss = (spread_width - net_credit) * option_size

            # === Log risk metrics ===
            self.logger.info(
                f"[BUYING POWER CHECK] Spread Width: {spread_width:.2f}, Net Credit: {net_credit:.2f}, "
                f"Max Risk: ${max_loss:.2f}, Allowed Risk Limit: ${buy_power_limit:.2f}"
            )

            # === Risk Test ===
            if max_loss >= buy_power_limit:
                self.rejection_reasons['buying_power_exceeded'] += 1
                return False
            return True

        except Exception as e:
            self.logger.error(f"[BPS ERROR] check_buying_power failed: {e}")
            return False

    def roll_rinse_bull_put_spread(
            self,
            short_put: Dict[str, Any],
            long_put: Dict[str, Any],
            underlying_price: float
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        """
        Evaluates exit/roll/hold conditions for a bull put spread.

        Returns:
            action: "HOLD", "EXIT", "ROLL", or "ERROR"
            spread_dict: updated spread if applicable
            reason: explanation string for logging
        """

        def extract_underlying_from_option_symbol(option_symbol: str) -> str:
            import re
            m = re.search(r"\d{6}[CP]", option_symbol)
            return option_symbol[:m.start()] if m else option_symbol

        def latest_mid(symbol: str) -> Optional[float]:
            """Best-effort mid for a single option symbol."""
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
            """Find the entry (execution) price for a leg."""
            for k in ("avg_entry_price", "initial_option_price", "option_price", "mid_price"):
                v = leg.get(k)
                if v and float(v) > 0:
                    return float(v)

            # fallback: broker avg entry price
            try:
                positions = self.trading_client.get_all_positions()
                for pos in positions:
                    pos_data = pos.model_dump() if hasattr(pos, "model_dump") else pos.__dict__
                    if pos_data.get("symbol") == leg.get("symbol"):
                        ap = pos_data.get("avg_entry_price")
                        if ap and float(ap) > 0:
                            return float(ap)
            except Exception as e:
                self.logger.warning(f"[ENTRY] lookup in positions failed for {leg.get('symbol')}: {e}")

            # last resort: use mid
            return latest_mid(leg.get("symbol"))

        try:
            # --- 0) Validate inputs ---
            if not underlying_price or underlying_price <= 0:
                return "ERROR", None, f"Invalid underlying price={underlying_price}"

            symbol = short_put.get("underlying_symbol")
            if not symbol and short_put.get("symbol"):
                symbol = extract_underlying_from_option_symbol(short_put["symbol"])
            if not symbol:
                return "ERROR", None, "Missing underlying symbol"

            # --- 1) Entry net credit ---
            short_entry = resolve_entry_price(short_put)
            long_entry = resolve_entry_price(long_put)
            if not short_entry or not long_entry:
                return "ERROR", None, "Missing initial spread prices"

            entry_credit = float(short_entry) - float(long_entry)
            if entry_credit <= 0:
                return "ERROR", None, f"Non-credit entry (entry_credit={entry_credit:.2f})"

            # Profit targets
            base_target = float(getattr(self, "target_profit_percent", 0.35) or 0.35)
            if entry_credit <= 0.20:
                target_profit = 0.50
            elif entry_credit < 0.30:
                target_profit = 0.30
            else:
                target_profit = max(min(base_target, 0.50), 0.30)

            # Always precompute relaxed target as 70% of strict target
            relaxed_target = round(target_profit * 0.7, 2)

            # --- 2) Current PnL ---
            try:
                positions = self.trading_client.get_all_positions()
                short_pos = next((p for p in positions if p.symbol == short_put["symbol"]), None)
                long_pos = next((p for p in positions if p.symbol == long_put["symbol"]), None)

                if short_pos and long_pos:
                    short_pnl = float(short_pos.unrealized_pl)
                    long_pnl = float(long_pos.unrealized_pl)
                    spread_pnl = short_pnl + long_pnl
                    profit_pct = spread_pnl / (entry_credit * 100)  # Alpaca uses $ pnl, contracts *100
                    pnl_source = "broker"
                else:
                    raise ValueError("Positions not found for both legs")
            except Exception as e:
                self.logger.warning(f"[PNL] Broker PnL fetch failed, falling back to mids: {e}")
                short_mid_now = latest_mid(short_put["symbol"]) or short_entry
                long_mid_now = latest_mid(long_put["symbol"]) or long_entry
                close_debit = max(float(short_mid_now) - float(long_mid_now), 0.0)
                realized_profit = entry_credit - close_debit
                profit_pct = realized_profit / entry_credit
                pnl_source = "mid"

            # --- 3) Holding time & exit windows ---
            days_open = 0
            if short_put.get("entry_time"):
                try:
                    days_open = (datetime.now(self.timezone) -
                                 datetime.fromisoformat(short_put["entry_time"])).days
                except Exception:
                    pass

            weekday_today = datetime.now(self.timezone).weekday()  # 0=Mon … 4=Fri
            relaxed_active = (weekday_today >= 2)  # Wed–Fri only
            friday_only = (weekday_today == 4)

            # --- 4) Risk metrics ---
            short_delta = abs(float(short_put.get("delta", 0.0)))
            short_vega = float(short_put.get("vega", 0.0))

            # --- Exit conditions ---
            if profit_pct >= target_profit:
                return "EXIT", None, f"Strict target {int(target_profit * 100)}% hit (Mon–Fri) (pnl={profit_pct:.1%}, src={pnl_source})"

            if relaxed_active and profit_pct >= relaxed_target:
                return "EXIT", None, f"Relaxed target {int(relaxed_target * 100)}% hit (Wed–Fri only, held {days_open}d, pnl={profit_pct:.1%}, src={pnl_source})"

            if friday_only and short_delta >= 0.65:
                return "EXIT", None, f"Delta hard stop (Fri only Δ={short_delta:.2f})"

            if friday_only:
                if short_delta >= float(getattr(self, "delta_stop_loss", 0.5)):
                    return "EXIT", None, f"Delta soft stop (Fri only Δ={short_delta:.2f})"
                if short_vega >= float(getattr(self, "vega_stop_loss_percent", 0.2)):
                    return "EXIT", None, f"Vega stop (Fri only ν={short_vega:.2f})"

            # --- Roll logic (Fri only, under pressure) ---
            if friday_only:
                if profit_pct < -0.40 and short_delta > 0.35:
                    try:
                        current_expiry = datetime.strptime(short_put["expiration_date"], "%Y-%m-%d")
                        new_expiration = current_expiry + timedelta(days=7)
                    except Exception:
                        new_expiration = None

                    try:
                        short_strike = float(short_put["strike_price"])
                        long_strike = float(long_put["strike_price"])
                        width = abs(short_strike - long_strike)
                        new_short_strike = short_strike - 2
                        new_long_strike = new_short_strike - width
                    except Exception as e:
                        return "ERROR", None, f"Failed to compute roll strikes: {e}"

                    new_spread = {
                        "symbol": short_put.get("underlying_symbol"),
                        "short_put": {
                            "symbol": f"{short_put['underlying_symbol']}{new_expiration.strftime('%y%m%d')}P{int(new_short_strike * 1000):08d}",
                            "strike_price": new_short_strike,
                            "expiration_date": new_expiration.strftime("%Y-%m-%d"),
                        },
                        "long_put": {
                            "symbol": f"{long_put['underlying_symbol']}{new_expiration.strftime('%y%m%d')}P{int(new_long_strike * 1000):08d}",
                            "strike_price": new_long_strike,
                            "expiration_date": new_expiration.strftime("%Y-%m-%d"),
                        },
                        "entry_time": datetime.now(self.timezone).isoformat()
                    }

                    est_credit = 0.30
                    if est_credit >= 0.30:
                        return "ROLL", new_spread, f"Rolled forward to next week (Fri only, pnl={profit_pct:.1%}, Δ={short_delta:.2f})"
                    else:
                        return "EXIT", None, "Credit too small to roll (Fri only)"

            # --- HOLD (with structured log) ---
            self.logger.info(
                f"[HOLD] {symbol} | pnl={profit_pct:.1%}, "
                f"strict_target={int(target_profit * 100)}% (Mon–Fri), "
                f"relaxed_target={int(relaxed_target * 100)}% (Wed–Fri, {'active' if relaxed_active else 'inactive'}), "
                f"Δ={short_delta:.2f} (hard≥0.65 Fri only, soft≥0.50 Fri only), "
                f"ν={short_vega:.2f} (stop≥0.20 Fri only), "
                f"roll=Fri only, held={days_open}d, src={pnl_source}"
            )
            return "HOLD", None, "No exit criteria met"

        except Exception as e:
            self.logger.error(f"[ROLLING ERROR] {e}")
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

    def _collect_legs_for_action(self, spread: dict, action: str) -> list[OptionLegRequest]:
        """
        action: 'open' or 'close'
        Spread can contain keys:
          - short_put, long_put
          - short_call, long_call   (for iron condor)
        Each leg dict must have: symbol, side ('buy'/'sell'), qty
        """
        leg_keys = ["short_put", "long_put", "short_call", "long_call"]
        legs = []
        for k in leg_keys:
            leg = spread.get(k)
            if not leg:
                continue
            symbol = leg["symbol"]
            qty = int(leg.get("qty", 1))
            if action == "open":
                side = OrderSide.BUY if leg["side"].lower() == "buy" else OrderSide.SELL
            elif action == "close":
                side = self._leg_close_side(leg["side"])
            else:
                raise ValueError("action must be 'open' or 'close'")

            # ratio_qty is per-contract ratio; we assume 1:1
            legs.append(OptionLegRequest(symbol=symbol, side=side, ratio_qty=1))
        if not legs:
            raise ValueError("No legs found in spread.")
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

    def load_spread_state(self, path="spread_state.json"):
        """
        Load saved spread state from disk (used only at startup).
        Falls back to empty if file is missing or corrupt.
        """
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    state = json.load(f)
                    self.logger.info(f"[STATE LOADED] {len(state)} spreads from {path}")
                    return state
        except Exception as e:
            self.logger.warning(f"[LOAD ERROR] Could not read {path}: {e}")

        return []

    def save_spread_state(self, state, path="spread_state.json"):
        try:
            with open(path, "w") as f:
                json.dump(state, f, default=str)
        except Exception as e:
            self.logger.error(f"[SAVE ERROR] Could not save spread state: {e}")

    def cleanup_expired_spreads(self, spreads):
        """
        Remove only spreads with a real expiry date in the past.
        Keep expiry=None spreads for evaluation.
        """
        today = datetime.now().date()
        cleaned_spreads = []
        for s in spreads:
            expiry_str = s.get("expiry")
            if not expiry_str or expiry_str.lower() == "none":  # Keep None / missing expiry
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

    def evaluate_put_options(self, put_options: list, price: float, label: str = "STRICT") -> None:
        """
        Evaluate short/long put candidates, log rejection reasons, and print option details.

        Args:
            put_options (list): List of option contracts.
            price (float): Current underlying price.
            label (str): Label for logging (e.g., "STRICT", "RELAXED").
        """
        short_c = self.criteria['short_put']
        long_c = self.criteria['long_put']

        short_table, long_table = [], []
        columns = ["Strike", "Delta", "IV", "OI", "DTE", "Spread", "Reason"]

        for opt in put_options:
            underlying_price = price
            opt_dict = self.build_option_dict(opt, underlying_price, self.risk_free_rate)

            delta = float(opt_dict.get('delta', 0))
            iv = float(opt_dict.get('implied_volatility', 0))
            bid = float(opt_dict.get('bid_price', 0))
            ask = float(opt_dict.get('ask_price', 0))
            oi = opt_dict.get('open_interest', 0)
            expiration = opt_dict.get('expiration_date')

            if expiration is None:
                continue  # reject invalid option

            if expiration.tzinfo is None:
                expiration = expiration.tz_localize('UTC')

            now = pd.Timestamp.now(tz='UTC')
            dte = (expiration - now).days
            spread = ask - bid
            strike = float(opt_dict['strike_price'])

            # --- Evaluate SHORT put leg ---
            short_reason = []
            if oi == 0:
                short_reason.append("OI=0")
            # if dte < short_c['min_days_to_exp']:
            #     short_reason.append(f"DTE<{short_c['min_days_to_exp']}")
            if spread > short_c['max_bid_ask_spread']:
                short_reason.append(f"Spread>{short_c['max_bid_ask_spread']}")
            if iv < short_c['min_iv']:
                short_reason.append(f"IV<{short_c['min_iv']}")
            if delta < short_c['min_delta']:
                short_reason.append(f"Delta<{short_c['min_delta']}")
            if delta > short_c['max_delta']:
                short_reason.append(f"Delta>{short_c['max_delta']}")

            if short_reason:
                short_table.append([
                    strike, f"{delta:.2f}", f"{iv:.2f}", oi, dte, f"{spread:.2f}",
                    ", ".join(short_reason)
                ])

            # --- Evaluate LONG put leg ---
            long_reason = []
            if oi == 0:
                long_reason.append("OI=0")
            # if dte < long_c['min_days_to_exp']:
            #     long_reason.append(f"DTE<{long_c['min_days_to_exp']}")
            if spread > long_c['max_bid_ask_spread']:
                long_reason.append(f"Spread>{long_c['max_bid_ask_spread']}")
            if iv < long_c['min_iv']:
                long_reason.append(f"IV<{long_c['min_iv']}")
            if delta < long_c['min_delta']:
                long_reason.append(f"Delta<{long_c['min_delta']}")
            if delta > long_c['max_delta']:
                long_reason.append(f"Delta>{long_c['max_delta']}")

            if long_reason:
                long_table.append([
                    strike, f"{delta:.2f}", f"{iv:.2f}", oi, dte, f"{spread:.2f}",
                    ", ".join(long_reason)
                ])

        # --- Logging rejection tables ---
        if short_table:
            df_short = pd.DataFrame(short_table, columns=columns)
            self.logger.info(f"\n[{label} REJECTED SHORT PUTS]\n{df_short.to_string(index=False)}")
        else:
            self.logger.info(f"[{label} REJECTED SHORT PUTS] None")

        if long_table:
            df_long = pd.DataFrame(long_table, columns=columns)
            self.logger.info(f"\n[{label} REJECTED LONG PUTS]\n{df_long.to_string(index=False)}")
        else:
            self.logger.info(f"[{label} REJECTED LONG PUTS] None")

        # --- Log all option details ---
        for opt in put_options:
            opt_dict = self.build_option_dict(opt, price, self.risk_free_rate)

            delta = opt_dict.get('delta')
            iv = opt_dict.get('implied_volatility')
            bid = opt_dict.get('bid_price', 0)
            ask = opt_dict.get('ask_price', 0)
            oi = opt_dict.get('open_interest', 'N/A')

            self.logger.info(
                f"[{label} OPTION] {opt_dict['symbol']} | Strike: {opt_dict['strike_price']} | "
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
                        new_row = pd.DataFrame([new_data], index=pd.MultiIndex.from_tuples([index_key], names=["symbol",
                                                                                                               "timestamp"]),
                                               columns=columns)
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

                    # Define DTE range: 8 to 14 days
                    min_dte = 8
                    max_dte = 14

                    # Update the criteria dynamically for both legs
                    for leg in ['short_put', 'long_put']:
                        self.criteria[leg]['min_days_to_exp'] = min_dte
                        self.criteria[leg]['max_days_to_exp'] = max_dte

                    self.logger.info(f"[CRITERIA] Targeting expiries between {min_dte}–{max_dte} days out")

                    # === Entry Check ===
                    # Only allow entries on Monday AND within 15 min after open until 4:00 PM
                    if (
                            ts.weekday() == 0  # Monday (0 = Monday, 6 = Sunday)
                            and market_open + timedelta(minutes=15) <= ts < market_open.replace(hour=16, minute=0)
                    ):
                        # Refresh before screening
                        self.active_spreads = self.sync_with_alpaca()

                        self.logger.info(f"[DEBUG] Skip list (open legs): {self.symbols_with_open_options}")

                        symbol_base = symbol.split()[0].strip()

                        # Skip if this symbol has *any* open option position
                        if symbol_base in self.symbols_with_open_options:
                            self.logger.info(f"[SKIP] {symbol} already has an open option leg.")
                            return

                        # --- Enforce 5-minute minimum re-screening interval per symbol ---
                        last_check: Optional[datetime] = self.last_screening_minute.get(symbol)

                        if last_check is None or (ts - last_check).total_seconds() >= 300:
                            self.last_screening_minute[symbol] = ts

                            # === Step 1: Momentum sanity check ===
                            if abs(macd - prev['MACD']) < 0.05 and abs(rsi - prev['RSI']) < 0.75:
                                self.logger.info(f"[SKIP] Weak momentum shift for {symbol}")
                                return

                            score = sum([
                                1 if 45 < rsi < 75 else 0,
                                1 if macd > 0 and macd_hist > 0 else 0,
                                1 if short_sma > long_sma else 0,
                                1 if adx > 20 else 0,
                                1 if volume > (volume_sma * 0.8) else 0
                            ])

                            self.logger.info(f"[SCREENED] {symbol} Score: {score}/5")

                            if score < 3:
                                self.logger.info(f"[BLOCKED] Signal score too low ({score}/5) for {symbol}")
                                return

                            self.logger.info(f"[ENTRY SIGNAL] Attempting spread for {symbol} at ${price:.2f} ")

                            today = datetime.now(self.timezone).date()
                            start_date = today + timedelta(days=8)
                            end_date = today + timedelta(days=14)

                            self.logger.debug(
                                f"[EXPIRY RANGE] Fetching options for {symbol} between {start_date} and {end_date}")

                            put_options = self.get_options(
                                symbol,
                                price * 0.9,
                                price * 1.1,
                                start_date,
                                end_date,
                                ContractType.PUT
                            )

                            self.logger.info(f"[CHAIN] Retrieved {len(put_options)} raw put options for {symbol}")

                            if not put_options:
                                self.logger.info(f"[CHAIN] No put options retrieved for {symbol}")
                                return

                            expiries = sorted(set([
                                self.build_option_dict(opt, price, self.risk_free_rate).get('expiration_date')
                                for opt in put_options if opt
                            ]))

                            self.logger.info(f"[EXPIRIES] Found {len(expiries)} expiry dates: {expiries}")

                            expiry_date = self.get_expiry_within_range(expiries, min_dte=8, max_dte=14)
                            if not expiry_date:
                                self.logger.warning(f"[EXPIRY] No valid expiry 8–14 DTE for {symbol}")
                                return False

                            # Filter to only options matching the selected expiry
                            put_options = [opt for opt in put_options if opt.expiration_date == expiry_date]

                            self.logger.info(
                                f"[CHAIN] Retrieved {len(put_options)} filtered put contracts for {symbol} expiring {expiry_date}")

                            if self.criteria is None:
                                self.criteria = self.relaxed_criteria

                            self.evaluate_put_options(put_options, price, label="STRICT")

                            # First attempt using strict criteria
                            spread = self.find_bull_put_spread(
                                put_options,
                                underlying_price=price,
                                risk_free_rate=self.risk_free_rate
                            )

                            if not all(spread):
                                self.logger.info(f"[NO SPREAD] No valid bull put spread found for {symbol}")
                                self.logger.info(f"[RETRY] Trying relaxed criteria for {symbol}")

                                # Backup strict criteria and apply relaxed ones
                                strict_criteria = self.criteria
                                self.criteria = self.relaxed_criteria

                                # Retry evaluation + spread finding with relaxed criteria
                                self.evaluate_put_options(put_options, price, label="RELAXED")

                                # Retry with relaxed parameters and lower OI threshold
                                spread = self.find_bull_put_spread(
                                    put_options,
                                    underlying_price=price,
                                    risk_free_rate=self.risk_free_rate
                                )

                                # Restore strict criteria to avoid side effects
                                self.criteria = strict_criteria

                            if not all(spread):
                                self.logger.info(
                                    f"[NO SPREAD] No valid bull put spread found for {symbol} (even with relaxed criteria)")
                                return

                            for reason, count in self.rejection_reasons.items():
                                self.logger.info(f"[REJECTION] {reason}: {count}")

                            # === After spread selection ===
                            short_put, long_put = spread

                            try:
                                # -------- Live quotes --------
                                s_bid = float(short_put.get("bid_price", 0.0))
                                s_ask = float(short_put.get("ask_price", 0.0))
                                l_bid = float(long_put.get("bid_price", 0.0))
                                l_ask = float(long_put.get("ask_price", 0.0))

                                if min(s_bid, s_ask, l_bid, l_ask) <= 0:
                                    self.logger.info("[REJECTED] Invalid live quotes (non-positive).")
                                    return

                                # Inversion guard
                                if s_bid > s_ask or l_bid > l_ask:
                                    self.logger.info("[REJECTED] Inverted quotes (bid > ask).")
                                    return

                                s_spread = s_ask - s_bid
                                l_spread = l_ask - l_bid

                                # -------- Spread sanity (single place) --------
                                # Absolute+relative limits; unify with thresholds you want to enforce globally
                                if s_spread > max(0.50, 0.20 * max(s_bid, 1e-6)) or l_spread > max(0.50,
                                                                                                   0.20 * max(l_bid,
                                                                                                              1e-6)):
                                    self.logger.info("[REJECTED] Quotes too wide (abs+rel).")
                                    return

                                # -------- Economics --------
                                short_strike = float(short_put["strike_price"])
                                long_strike = float(long_put["strike_price"])
                                width = short_strike - long_strike
                                if width <= 0:
                                    self.logger.info("[REJECTED] Invalid width.")
                                    return

                                # Conservative natural credit; optionally compute mid for diagnostics
                                net_credit = max(0.0, s_bid - l_ask)

                                # Use contract multiplier if present
                                multiplier = int(short_put.get("multiplier", long_put.get("multiplier", 100)))
                                per_spread_risk = (width - net_credit) * multiplier
                                if per_spread_risk <= 0:
                                    self.logger.info("[REJECTED] Non-positive per-spread risk.")
                                    return

                                credit_to_risk = net_credit / width  # unitless; width already in $ terms

                                # -------- Filters (strict / relaxed) --------
                                try:
                                    delta = abs(float(short_put.get("delta", 0.0)))
                                except (TypeError, ValueError):
                                    delta = 0.0
                                try:
                                    oi = int(short_put.get("open_interest", 0))
                                except (TypeError, ValueError):
                                    oi = 0

                                strict_pass = (
                                        0.15 <= delta <= 0.35 and
                                        oi >= 50 and
                                        s_spread <= 0.20 and
                                        credit_to_risk >= 0.25
                                )
                                relaxed_pass = (
                                        0.10 <= delta <= 0.40 and
                                        oi >= 20 and
                                        s_spread <= max(0.20, 0.10 * max(s_bid, 1e-6)) and
                                        credit_to_risk >= 0.15
                                )

                                if strict_pass:
                                    self.logger.info(
                                        "[ACCEPTED:STRICT] SP=%s, LP=%s, Δ=%.2f, OI=%d, Credit=%.2f, CRR=%.2f",
                                        short_strike, long_strike, delta, oi, net_credit, credit_to_risk
                                    )
                                elif relaxed_pass:
                                    self.logger.info(
                                        "[ACCEPTED:RELAXED] SP=%s, LP=%s, Δ=%.2f, OI=%d, Credit=%.2f, CRR=%.2f",
                                        short_strike, long_strike, delta, oi, net_credit, credit_to_risk
                                    )
                                else:
                                    self.logger.info("[REJECTED] Failed filters (Δ=%.2f, CRR=%.2f, OI=%d)", delta,
                                                     credit_to_risk, oi)
                                    return

                                # -------- Account & sizing --------
                                try:
                                    account = self.trading_client.get_account()
                                except Exception as e:
                                    self.logger.error("[ERROR] get_account failed: %s", e)
                                    return

                                portfolio = float(account.portfolio_value)
                                buying_power = float(account.buying_power)

                                # Be consistent: set 0.3 for 30% or fix comment
                                per_trade_cap = portfolio * 0.3 # Max 20% of portfolio per trade
                                max_risk_allowed = buying_power * float(getattr(self, "buy_power_limit", 1.0))

                                # Compute integer max quantities under each constraint
                                def safe_floor(dividend, divisor):
                                    return max(0, int(dividend / divisor)) if divisor > 0 else 0

                                max_qty_bp = safe_floor(buying_power, per_spread_risk)
                                max_qty_cap = max(1, safe_floor(per_trade_cap,
                                                                per_spread_risk))  # allow at least 1 if cap >= risk
                                max_qty_risk = safe_floor(max_risk_allowed, per_spread_risk)

                                qty = max(0, min(max_qty_bp, max_qty_cap, max_qty_risk))

                                self.logger.info(
                                    "[SIZING] risk/spread=%.2f, max_qty_bp=%d, max_qty_cap=%d, max_qty_risk=%d, final_qty=%d",
                                    per_spread_risk, max_qty_bp, max_qty_cap, max_qty_risk, qty
                                )

                                if qty < 1:
                                    self.logger.info("[SKIP] Not enough BP for 1 contract under risk caps.")
                                    return
                                if per_trade_cap < per_spread_risk:
                                    self.logger.warning(
                                        "[WARNING] Per-trade cap ($%.2f) < risk per spread ($%.2f). Proceeding with 1 contract if BP permits.",
                                        per_trade_cap, per_spread_risk
                                    )

                                if qty == max_qty_bp:
                                    self.logger.info("[INFO] Position size limited by Buying Power cap.")
                                elif qty == max_qty_cap:
                                    self.logger.info("[INFO] Position size limited by Per-Trade cap (10%% portfolio).")
                                elif qty == max_qty_risk:
                                    self.logger.info("[INFO] Position size limited by Global Risk cap.")
                                else:
                                    self.logger.info("[INFO] Position sizing not capped (rare).")

                                # -------- Final BP check using sized qty --------
                                if not self.check_buying_power(short_put, long_put):
                                    self.logger.info("[REJECTED] check_buying_power failed for qty=%d.", qty)
                                    return

                                # -------- Prepare & place order --------
                                symbol = (
                                        short_put.get("underlying_symbol")
                                        or short_put.get("root_symbol")
                                        or long_put.get("underlying_symbol")
                                )
                                if not symbol:
                                    self.logger.info("[REJECTED] Missing underlying symbol.")
                                    return

                                # Tick-aware rounding (defaults to 0.01)
                                tick = (
                                        short_put.get("tick_size")
                                        or long_put.get("tick_size")
                                        or 0.01
                                )
                                # Round down to the closest tick to avoid broker rejections
                                ticks = int(net_credit / tick)
                                target_limit = round(ticks * tick, 2)

                                # Avoid mutating shared dicts if reused elsewhere
                                short_leg = dict(short_put)
                                short_leg["side"] = "SELL"
                                long_leg = dict(long_put)
                                long_leg["side"] = "BUY"

                                order = {"symbol": symbol, "short_put": short_leg, "long_put": long_leg}

                                try:
                                    res = self.submit_mleg_open(order, total_qty=qty, limit_price=target_limit)
                                except Exception as e:
                                    self.logger.error("[ERROR] submit_mleg_open failed: %s", e)
                                    return

                                if res:
                                    tag = "STRICT" if strict_pass else "RELAXED"
                                    self.logger.info(
                                        "[ORDER PLACED - %s] %s BPS %dx @ %.2f | width %.2f | credit %.2f | risk %.2f",
                                        tag, symbol, qty, target_limit, width, net_credit, per_spread_risk
                                    )
                                    ts = datetime.now(timezone.utc).isoformat()
                                    # Persist state
                                    entry = {
                                        "symbol": symbol,
                                        "underlying_symbol": symbol,
                                        "short_put": short_leg,
                                        "long_put": long_leg,
                                        "entered_at": ts,
                                        "expiry": str(short_put["expiration_date"].date()),
                                        "qty": qty
                                    }
                                    self.active_spreads.append(entry)
                                    self.entry_log[symbol] = ts
                                    self.save_spread_state(self.active_spreads)

                            except Exception as e:
                                self.logger.error("[BPS ERROR] Exception during spread check: %s", e)
                                return

                    # === Exit Check ===
                    if market_open + timedelta(hours=1) <= ts < market_close - timedelta(minutes=15):

                        # Step 0: Refresh from persisted state and Alpaca
                        self.active_spreads = self.sync_with_alpaca()

                        if not self.active_spreads:
                            self.logger.info("[EXIT CHECK] No active spreads to evaluate.")
                        else:
                            self.logger.info(f"[EXIT CHECK] Evaluating exit for {len(self.active_spreads)} spreads:")

                            # Log all spreads under review
                            for spread in self.active_spreads:
                                short = spread.get("short_put", {}).get("symbol")
                                long = spread.get("long_put", {}).get("symbol")
                                self.logger.info(f"   [REVIEW] {spread['symbol']} | Short={short} | Long={long}")

                            # Step 1: Remove spreads no longer live in Alpaca
                            try:
                                positions = self.trading_client.get_all_positions()
                                open_option_symbols = {pos.symbol for pos in positions if
                                                       pos.asset_class == "us_option"}
                            except Exception as e:
                                self.logger.error(f"[EXIT CHECK] Failed to fetch Alpaca positions: {e}")
                                open_option_symbols = set()

                            for spread in list(self.active_spreads):
                                short_symbol = spread["short_put"].get("symbol")
                                long_symbol = spread["long_put"].get("symbol")

                                if short_symbol not in open_option_symbols and long_symbol not in open_option_symbols:
                                    self.logger.info(f"[EXIT CHECK] Spread closed externally for {spread['symbol']}")
                                    self.active_spreads.remove(spread)

                            # Step 2: Process remaining spreads
                            try:
                                positions = self.trading_client.get_all_positions()
                                option_qtys = {
                                    pos.symbol: abs(int(float(pos.qty)))
                                    for pos in positions if pos.asset_class == "us_option"
                                }
                            except Exception as e:
                                self.logger.error(
                                    f"[EXIT CHECK] Failed to fetch Alpaca positions for exit/roll logic: {e}")
                                option_qtys = {}

                            price = df['close'].iloc[-1]

                            for spread in self.active_spreads.copy():
                                short_put = spread.get("short_put", {})
                                long_put = spread.get("long_put", {})
                                if not short_put or not long_put:
                                    self.logger.warning(f"[EXIT CHECK] Skipping incomplete spread {spread['symbol']}")
                                    continue

                                short_sym = short_put.get("symbol")
                                long_sym = long_put.get("symbol")

                                action, spread_update, reason = self.roll_rinse_bull_put_spread(
                                    short_put=short_put,
                                    long_put=long_put,
                                    underlying_price=price
                                )

                                if action == "HOLD":
                                    self.logger.info(
                                        f"[HOLD] {spread['symbol']} | Reason={reason} | Short={short_sym} | Long={long_sym}"
                                    )

                                elif action == "EXIT":
                                    qty = option_qtys.get(short_sym, 0)
                                    if qty > 0:
                                        try:
                                            res = self.submit_mleg_close(spread, qty)
                                            self.logger.info(
                                                f"[EXIT] {spread['symbol']} | Reason={reason} | "
                                                f"Short={short_sym} | Long={long_sym} "
                                                f"| qty={qty}, order_id={getattr(res, 'id', 'N/A')}"
                                            )
                                        except Exception as e:
                                            self.logger.error(
                                                f"[EXIT SUBMIT FAILED] {spread['symbol']} | Reason={reason} | "
                                                f"Short={short_sym} | Long={long_sym} | {e}"
                                            )
                                    else:
                                        self.logger.warning(
                                            f"[EXIT] {spread['symbol']} | Reason={reason} | "
                                            f"Short={short_sym} | Long={long_sym} | No live qty in Alpaca (already closed?)"
                                        )

                                    self.active_spreads.remove(spread)
                                    self.save_spread_state(self.active_spreads)

                                elif action == "ROLL":
                                    self.logger.info(
                                        f"[ROLL] {spread['symbol']} | Reason={reason} | Short={short_sym} | Long={long_sym}"
                                    )
                                    self.active_spreads.remove(spread)
                                    self.active_spreads.append(spread_update)
                                    self.save_spread_state(self.active_spreads)

                                elif action == "ERROR":
                                    self.logger.warning(
                                        f"[ERROR] {spread['symbol']} | Reason={reason} | Short={short_sym} | Long={long_sym}"
                                    )

                            # === Step 3: Portfolio-level profit exit ===
                            try:
                                account = self.trading_client.get_account()
                                equity = float(account.equity)
                                last_equity = float(account.last_equity)

                                # Alpaca’s daily comparison
                                alpaca_pct = ((
                                                      equity - last_equity) / last_equity) * 100 if last_equity > 0 else 0.0

                                self.logger.info(
                                    f"[PORTFOLIO CHECK] Equity={equity:.2f}, Prev={last_equity:.2f}, Change(Alpaca)={alpaca_pct:.2f}%"
                                )

                                # Trigger exit if portfolio up 5%+
                                if alpaca_pct >= 5.0:
                                    self.logger.info(
                                        f"[PORTFOLIO EXIT] Closing ALL spreads (Alpaca Profit {alpaca_pct:.2f}%)")

                                    # Step 0: Refresh from persisted state and Alpaca
                                    self.active_spreads = self.sync_with_alpaca()

                                    if not self.active_spreads:
                                        self.logger.info("[PORTFOLIO EXIT] No active spreads to close.")
                                    else:
                                        try:
                                            positions = self.trading_client.get_all_positions()
                                            option_qtys = {
                                                pos.symbol: abs(int(float(pos.qty)))
                                                for pos in positions
                                                if getattr(pos, "asset_class", "").lower() == "us_option"
                                            }

                                            for spread in list(self.active_spreads):
                                                short_put = spread.get("short_put", {})
                                                long_put = spread.get("long_put", {})
                                                short_symbol = short_put.get("symbol")
                                                long_symbol = long_put.get("symbol")

                                                # Safety: skip incomplete spreads
                                                if not short_put or not long_put:
                                                    self.logger.warning(
                                                        f"[PORTFOLIO EXIT] Skipping incomplete spread {spread.get('symbol', '?')}"
                                                    )
                                                    continue

                                                qty = option_qtys.get(short_symbol, 0)

                                                if qty > 0:
                                                    try:
                                                        res = self.submit_mleg_close(spread, qty)
                                                        self.logger.info(
                                                            f"[PORTFOLIO EXIT] Closed {spread['symbol']} | "
                                                            f"Short={short_symbol} | Long={long_symbol} | "
                                                            f"qty={qty}, order_id={getattr(res, 'id', 'N/A')}"
                                                        )
                                                    except Exception as e:
                                                        self.logger.error(
                                                            f"[PORTFOLIO EXIT FAILED] {spread['symbol']} | "
                                                            f"Short={short_symbol} | Long={long_symbol} | {e}"
                                                        )
                                                else:
                                                    self.logger.info(
                                                        f"[PORTFOLIO EXIT] {spread['symbol']} already closed (no live qty in Alpaca)."
                                                    )

                                                # Remove from active spreads regardless
                                                self.active_spreads.remove(spread)
                                                self.save_spread_state(self.active_spreads)

                                        except Exception as e:
                                            self.logger.error(f"[PORTFOLIO EXIT ERROR] {e}")

                            except Exception as e:
                                self.logger.error(f"[PORTFOLIO CHECK FAILED] {e}")

                    # # === Portfolio-level profit exit ===
                    # # if market_open + timedelta(hours=2) <= ts < market_close - timedelta(minutes=15):
                    # if ts.weekday() == 4 and ts >= market_open + timedelta(hours=1):
                    #     try:
                    #         account = self.trading_client.get_account()
                    #         equity = float(account.equity)
                    #         last_equity = float(account.last_equity)
                    #
                    #         # Alpaca’s daily comparison
                    #         alpaca_pct = ((
                    #                                   equity - last_equity) / last_equity) * 100 if last_equity > 0 else 0.0
                    #
                    #         self.logger.info(
                    #             f"[PORTFOLIO CHECK] Equity={equity:.2f}, Prev={last_equity:.2f}, Change(Alpaca)={alpaca_pct:.2f}%"
                    #         )
                    #
                    #         # Trigger exit if portfolio up 5%+
                    #         if alpaca_pct >= 2.0:
                    #             self.logger.info(
                    #                 f"[PORTFOLIO EXIT] Closing ALL spreads (Alpaca Profit {alpaca_pct:.2f}%)")
                    #
                    #             # Step 0: Refresh from persisted state and Alpaca
                    #             self.active_spreads = self.sync_with_alpaca()
                    #
                    #             if not self.active_spreads:
                    #                 self.logger.info("[PORTFOLIO EXIT] No active spreads to close.")
                    #             else:
                    #                 try:
                    #                     positions = self.trading_client.get_all_positions()
                    #                     option_qtys = {
                    #                         pos.symbol: abs(int(float(pos.qty)))
                    #                         for pos in positions
                    #                         if getattr(pos, "asset_class", "").lower() == "us_option"
                    #                     }
                    #
                    #                     for spread in list(self.active_spreads):
                    #                         short_put = spread.get("short_put", {})
                    #                         long_put = spread.get("long_put", {})
                    #                         short_symbol = short_put.get("symbol")
                    #                         long_symbol = long_put.get("symbol")
                    #
                    #                         # Safety: skip incomplete spreads
                    #                         if not short_put or not long_put:
                    #                             self.logger.warning(
                    #                                 f"[PORTFOLIO EXIT] Skipping incomplete spread {spread.get('symbol', '?')}"
                    #                             )
                    #                             continue
                    #
                    #                         qty = option_qtys.get(short_symbol, 0)
                    #
                    #                         if qty > 0:
                    #                             try:
                    #                                 res = self.submit_mleg_close(spread, qty)
                    #                                 self.logger.info(
                    #                                     f"[PORTFOLIO EXIT] Closed {spread['symbol']} | "
                    #                                     f"Short={short_symbol} | Long={long_symbol} | "
                    #                                     f"qty={qty}, order_id={getattr(res, 'id', 'N/A')}"
                    #                                 )
                    #                             except Exception as e:
                    #                                 self.logger.error(
                    #                                     f"[PORTFOLIO EXIT FAILED] {spread['symbol']} | "
                    #                                     f"Short={short_symbol} | Long={long_symbol} | {e}"
                    #                                 )
                    #                         else:
                    #                             self.logger.info(
                    #                                 f"[PORTFOLIO EXIT] {spread['symbol']} already closed (no live qty in Alpaca)."
                    #                             )
                    #
                    #                         # Remove from active spreads regardless
                    #                         self.active_spreads.remove(spread)
                    #                         self.save_spread_state(self.active_spreads)
                    #
                    #                 except Exception as e:
                    #                     self.logger.error(f"[PORTFOLIO EXIT ERROR] {e}")
                    #
                    #     except Exception as e:
                    #         self.logger.error(f"[PORTFOLIO CHECK FAILED] {e}")

                    # Exit all positions
                    # if ts.weekday() == 4 and ts >= market_open + timedelta(hours=1):
                    #
                    #     # Step 0: Refresh from persisted state and Alpaca
                    #     self.active_spreads = self.sync_with_alpaca()
                    #
                    #     if not self.active_spreads:
                    #         self.logger.info("[FRIDAY EXIT] No active spreads to close.")
                    #     else:
                    #         try:
                    #             positions = self.trading_client.get_all_positions()
                    #             option_qtys = {
                    #                 pos.symbol: abs(int(float(pos.qty)))
                    #                 for pos in positions
                    #                 if getattr(pos, "asset_class", "").lower() == "us_option"
                    #             }
                    #
                    #             for spread in list(self.active_spreads):
                    #                 short_put = spread.get("short_put", {})
                    #                 long_put = spread.get("long_put", {})
                    #                 short_symbol = short_put.get("symbol")
                    #                 long_symbol = long_put.get("symbol")
                    #
                    #                 # Safety: skip incomplete spreads
                    #                 if not short_put or not long_put:
                    #                     self.logger.warning(
                    #                         f"[FRIDAY EXIT] Skipping incomplete spread {spread.get('symbol', '?')}"
                    #                     )
                    #                     continue
                    #
                    #                 qty = option_qtys.get(short_symbol, 0)
                    #
                    #                 if qty > 0:
                    #                     try:
                    #                         res = self.submit_mleg_close(spread, qty)
                    #                         self.logger.info(
                    #                             f"[FRIDAY EXIT] Closed {spread['symbol']} | Short={short_symbol} | Long={long_symbol} "
                    #                             f"| qty={qty}, order_id={getattr(res, 'id', 'N/A')}"
                    #                         )
                    #                     except Exception as e:
                    #                         self.logger.error(
                    #                             f"[FRIDAY EXIT FAILED] {spread['symbol']} | Short={short_symbol} | Long={long_symbol} | {e}"
                    #                         )
                    #                 else:
                    #                     self.logger.info(
                    #                         f"[FRIDAY EXIT] {spread['symbol']} already closed (no live qty in Alpaca)."
                    #                     )
                    #
                    #                 # Remove from active spreads regardless
                    #                 self.active_spreads.remove(spread)
                    #                 self.save_spread_state(self.active_spreads)
                    #
                    #         except Exception as e:
                    #             self.logger.error(f"[FRIDAY EXIT ERROR] {e}")

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
    strategy = BullPutSpread()

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
