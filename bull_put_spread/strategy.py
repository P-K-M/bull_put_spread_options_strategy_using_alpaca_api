"""
strategy.py
--------------

Author: Paul K. Mwangi
"""

# Imports and basic configuration
# Purpose: group and document all imports and top-level configuration used across
# the Bull Put Spread strategy.

# -------------------------
# Standard library imports
# -------------------------
import threading
import logging
import time
from zoneinfo import ZoneInfo
from datetime import date, datetime

# -------------------------
# Numerical & statistics
# -------------------------
# pandas / numpy are used for data manipulation and numeric arrays. scipy provides
# probability functions and solvers used by pricing/IV calculations.
import pandas as pd

# -------------------------
# Typing and developer helpers
# -------------------------
# Used to provide type hints which improves readability and helps code review.
from typing import TypedDict, Optional, Dict, Any, List, Tuple
from collections import defaultdict

# -------------------------
# Alpaca (data + trading)
# -------------------------
# Alpaca packages provide both historical & live data access and trading client
# objects. They are central for live/paper trading and option chain retrieval.
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.requests import (
    GetCalendarRequest,
    OptionLegRequest
)
from alpaca.trading.enums import (
    OrderSide,
    ContractType,
    PositionSide
)

# -------------------------
# Logging configuration
# -------------------------
# Configure a named logger for the strategy.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("bull_put_strategy")

API_KEY = "[ENTER API KEY]"
SECRET_KEY = "[ENTER SECRET KEY]"
PEPER_URL = "https://paper-api.alpaca.markets/v2"
PAPER = True # Use paper trading by default; change to False for live trading

if not API_KEY or not SECRET_KEY:
    logger.warning("ALPACA_API_KEY or ALPACA_SECRET_KEY not set. Replace with secure environment variables before running live.")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

# -------------------------
# Strategy configuration (tunable parameters)
# -------------------------

TRADING_CONFIG: Dict[str, Any] = {
    # Market assumptions
    "RISK_FREE_RATE": 0.04,  # Annual risk-free rate used in option pricing (e.g. 4%)
    "TRADING_YEAR_SIZE": 252,  # Trading days per year for annualization

    # Indicator parameters (technical signal generation)
    "RSI_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "ADX_PERIOD": 14,
    "SMA_SHORT": 20,
    "SMA_LONG": 50,
}

# Strategy class and core state
# Purpose: Define the data structures and primary BullPutSpread class responsible
# for holding state, configuration, and API clients.

# -------------------------
# Data structures
# -------------------------
# TypedDict provides explicit structure for spread records persisted to state
# or sent to reporting dashboards.
class SpreadData(TypedDict):
    """Schema for a single Bull Put Spread trade record.

    Fields:
        symbol: Underlying symbol (e.g., 'SPY').
        short_put: Serialized option contract details for the short leg.
        long_put: Serialized option contract details for the long leg.
        entered_at: ISO timestamp when the spread was entered.
        expiry: Expiration date (ISO string) of the spread.
        qty: Number of option contracts (None if not yet determined).
    """
    symbol: str
    short_put: Dict[str, Any]
    long_put: Dict[str, Any]
    entered_at: str
    expiry: str
    qty: Optional[float]


# -------------------------
# BullPutSpread class
# -------------------------
class BullPutSpread:
    """Encapsulates state and configuration for the bull put spread strategy.

        Responsibilities:
          - Manage connections to exchange/data providers (Alpaca clients).
          - Hold strategy parameters and selection criteria.
          - Track internal runtime state (active spreads, minute-level history).
          - Provide methods for spread discovery, scoring, placement and exits
            (implementation in later sections).

        """
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
        self.logger.debug("BullPutSpread initialized.")

        # === Universe of underlying assets ===
        self.underlying_assets = [
            "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLI", "XLV", "SMH", "XLE",
            "EEM", "FXI", "GLD", "GDX", "EFA", "EWZ", "TLT", "EWC", "KWEB", "ARKK",
            "TQQQ", "LQD", "BKLN"
        ]

        # === Strategy Parameters ===
        self.strike_range = 0.05
        self.buy_power_limit = 0.30
        self.risk_free_rate = self.get_risk_free_rate()
        self.target_profit_percent = 0.35
        self.delta_stop_loss = 0.5
        self.vega_stop_loss_percent = 0.2

        # === Buying Power ===
        self.buying_power = float(self.trading_client.get_account().buying_power)
        self.buying_power_limit = self.buying_power * self.buy_power_limit

        # === Core Spread Selection Criteria ===
        self.criteria = {

        }

        # === Relaxed criteria (fallback) ===
        # Used when the market environment is thin or no spreads match strict
        # criteria. These values are deliberately more permissive.
        self.relaxed_criteria = {

        }

        # === Internal runtime state ===
        # Collections used by the strategy during execution and for reporting.
        self.minute_history: Dict[str, pd.DataFrame] = {}
        self.active_spreads = []
        self.symbols_with_open_options = set()
        self.rejection_reasons = defaultdict(int)
        self.last_checked: Dict[str, int] = {}

        self.logger.debug(f"BullPutSpread initialized on {self.today} with buying power ${self.buying_power:,.2f}.")

    # -----------------------------------------------------------------------------
    # === Utility functions (risk-free rate, expiry selection, historical data) ===
    # -----------------------------------------------------------------------------

    # Purpose: Implement reusable utility methods for market data retrieval and
    # strategy parameterization.
    def get_risk_free_rate(self) -> float:
        """
        Fetches the current risk-free rate (symbol: ^IRX).
        If unavailable, returns the default configured rate.

        Returns:
            float: Annualized risk-free interest rate (e.g., 0.045 for 4.5%)
        """

    def get_expiry_within_range(self, expiries, min_dte=8, max_dte=14, today=None):
        """Select an expiry within [min_dte, max_dte] trading days.

        Args:
            expiries (List[str]): Available expiry dates as strings.
            min_dte (int): Minimum days-to-expiration.
            max_dte (int): Maximum days-to-expiration.
            today (date, optional): Anchor date for DTE calculation. Defaults to now (EST).

        Returns:
            date | None: The nearest valid expiry date, or None if none found.
        """

        return

    def get_1000m_history_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Retrieve the last 1000 minutes of historical data for a list of symbols.

        Args:
            symbols (List[str]): List of ticker symbols to fetch minute-level data for.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their OHLCV history.
        """


    # ==============================================================================
    # Implied Volatility, Greeks, and Technical Indicators
    # ------------------------------------------------------------------------------
    # This section equips the strategy with:
    #   (A) Options valuation tools (Implied Volatility estimation + Greeks)
    #   (B) Technical indicators for screening underlying assets
    #
    # Purpose:
    # - Estimate IV from market prices to assess option mispricing.
    # - Compute Black-Scholes Greeks (Delta, Gamma, Theta, Vega) to evaluate
    #   option sensitivity to market factors.
    # - Apply TA indicators (RSI, SMA, MACD, ADX) for entry/exit signals.
    # ==============================================================================

    # ================================
    # (A) Option Valuation & Greeks
    # ================================

    def calculate_implied_volatility(self, option_price, S, K, T, r, option_type):
        """
        Estimate the implied volatility using the Black-Scholes model via Brent's method.
        Handles options close to intrinsic value gracefully.
        """

    def calculate_greeks(self, option_price, strike_price, expiration, underlying_price,
                         risk_free_rate, option_type, IV=None):
        """
        Calculate Black-Scholes Greeks (Delta, Gamma, Theta, Vega) with fallback handling.
        """


        return

    # ================================
    # (B) Technical Indicators
    # ================================

    def calculate_rsi(self, series, period=14):
        """
        Calculate Relative Strength Index (RSI) using exponential smoothing.
        Args:
            series (pd.Series): Series of closing prices.
            period (int): Lookback period (default=14).

        Returns:
            pd.Series: RSI values scaled 0–100.
        """

        return

    def calculate_moving_averages(self, df, short=20, long=50):
        """
        Compute short- and long-term Simple Moving Averages (SMA).

        Args:
            df (pd.DataFrame): Price data with 'close' column.
            short (int): Short-term window size.
            long (int): Long-term window size.

        Returns:
            pd.DataFrame: Original DataFrame with added SMA columns.
        """

        return df

    def calculate_macd_features(self, df, fast=12, slow=26, signal=9):
        """
        Compute Moving Average Convergence Divergence (MACD).

        Args:
            df (pd.DataFrame): Price data with 'close' column.
            fast (int): Fast EMA window.
            slow (int): Slow EMA window.
            signal (int): Signal line EMA window.

        Returns:
            pd.DataFrame: DataFrame with MACD, Signal, and Histogram columns.
        """


    def calculate_adx_features(self, df, period=14):
        """
        Compute Average Directional Index (ADX) to measure trend strength.

        Args:
            df (pd.DataFrame): Price data with 'high', 'low', 'close' columns.
            period (int): Lookback period for ADX.

        Returns:
            pd.DataFrame: DataFrame with added ADX column.
        """

        return df

    def add_all_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and append all technical indicators used for signal generation.

        Args:
            dataframe (pd.DataFrame): OHLCV price data.

        Returns:
            pd.DataFrame: DataFrame enriched with:
                          - SMA(20), SMA(50)
                          - MACD, Signal, MACD Histogram
                          - ADX (trend strength)
                          - RSI (momentum)
                          - Volume SMA(20)
        """

    # ==============================================================================
    # Asset Selection Based on Technical and IV Criteria
    # ------------------------------------------------------------------------------
    # Purpose:
    #   - Identify bullish index ETFs or stocks suitable for Bull Put Spreads.
    #   - Use a two-tier screening approach:
    #       (1) Primary screen: Technical trend + momentum + strength + volume + IV rank.
    #       (2) Fallback: Simple momentum rule (open > 20-day SMA) if no symbols pass.
    #
    # Why this matters:
    #   - Ensures the strategy only sells puts on strong/bullish assets.
    #   - Adds robustness: if no symbols meet strict criteria, fallback ensures
    #     continuity of trading opportunities.
    # ==============================================================================
    def get_underlying_assets(self) -> List[str]:
        """
        Screen tradable assets for bullish setups.

        Workflow:
            1. Retrieve active tradable assets from Alpaca.
            2. Fetch 6-month historical data for each candidate symbol.
            3. Compute technical indicators (SMA, MACD, RSI, ADX, Volume).
            4. Score each symbol (0–5) based on bullish criteria.
            5. Filter by implied volatility (IV > 0.15 ensures premium worth selling).
            6. Rank and return top candidates.
            7. If none pass → fallback to momentum filter.

        Returns:
            List[str]: Ranked list of selected bullish symbols.
        """
    def get_20day_ma_assets(self) -> List[str]:
        """
        Momentum fallback filter.
        Selects assets where: today's OPEN > 20-day SMA(close).
        Used if no assets pass the stricter screening.

        Returns:
            List[str]: List of momentum-based fallback tickers.
        """


    # ==============================================================================
    # OPTIONS CHAIN HANDLING: Retrieval, Validation, Pricing, and Greeks Calculation
    # ==============================================================================
    """
    This module handles the complete option processing pipeline:
    1. Retrieval of option chains from Alpaca (within strike/expiry bounds).
    2. Validation of liquidity (open interest, tradability).
    3. Pricing using bid/ask quotes and mid-price logic.
    4. Implied Volatility (IV) estimation via Black-Scholes.
    5. Greeks calculation (Δ, Γ, Θ, Vega).
    6. Unified option dictionary construction for consistency across the strategy.

    This ensures that each option contract is transformed into a normalized,
    fully enriched data structure before being used for spread construction.
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

        Args:
            symbol (str): Underlying ticker (e.g., SPY, QQQ).
            min_strike (float): Minimum strike price.
            max_strike (float): Maximum strike price.
            min_exp (date): Minimum expiration date.
            max_exp (date): Maximum expiration date.
            option_type (ContractType): PUT (default) or CALL.

        Returns:
            list: A list of option contracts retrieved from Alpaca.
        """"""
        Fetch options within strike and expiration bounds for a given symbol.
        """

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

    def ensure_dict(self, option_data) -> dict:
        """
        Ensure option data is in dictionary format (convert Pydantic-style models).

        Args:
            option_data: Option contract object (dict or Alpaca model).

        Returns:
            dict: Option data as a plain dictionary.
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
        Calculate option pricing metrics and Greeks using Black-Scholes.

        Steps:
            1. Retrieve live bid/ask from Alpaca.
            2. Compute mid-price as the best estimate of option value.
            3. Estimate implied volatility (IV) from mid-price.
            4. Calculate Greeks (Δ, Γ, Θ, Vega) using BSM model.

        Args:
            option_data (dict): Option contract details.
            underlying_price (float): Current stock/ETF price.
            risk_free_rate (float): Annualized risk-free rate (e.g., 0.045).

        Returns:
            dict: Metrics including option_price, IV, delta, gamma, theta, vega.
        """


    # ------------------------------------------------------------------------------
    # Unified Option Dictionary Constructor
    # ------------------------------------------------------------------------------

    def safe_float(self,val, default=0.0):
        """Safe conversion to float (fallbacks on invalid/None values)."""
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
        """
        Build a normalized dictionary with all option attributes + calculated metrics.

        Args:
            option_data (dict or model): Raw option contract from Alpaca.
            underlying_price (float): Current underlying asset price.
            risk_free_rate (float): Risk-free rate for Greeks/IV.

        Returns:
            dict: Enriched option object with standardized fields for strategy use.
        """

    # ---------------------------------------------------------------------------
    # OPTION FILTERING & SPREAD CONSTRUCTION LOGIC
    # ---------------------------------------------------------------------------

    def check_option_conditions(self, candidate: dict, label: str, criteria: dict) -> bool:
        """
        Validate whether an option candidate (put) meets the strategy-defined criteria.

        This function applies multiple layers of filtering to ensure that only
        liquid, reasonably priced, and strategically viable options are considered
        for spread construction.

        Args:
            candidate (dict): Option contract details.
            label (str): "SHORT" for short put leg, "LONG" for long put leg.
            criteria (dict): Strategy rules defining min/max thresholds.

        Returns:
            bool: True if candidate passes all checks, False otherwise.
        """


    # ---------------------------------------------------------------------------
    # Spread Construction: Identify best bull put spread candidates
    # ---------------------------------------------------------------------------
    def find_bull_put_spread(
            self,
            put_options: List[Any],
            underlying_price: float,
            risk_free_rate: float
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Main bull put spread finder.

        Workflow:
        1. Build standardized option dictionaries for all candidates.
        2. Filter each candidate using check_option_conditions().
        3. Group valid SHORT and LONG puts by expiry date.
        4. Pair only the closest strikes to minimize risk.
        5. Score each spread and select the best candidate.

        Args:
            put_options (List[Any]): List of raw option contracts.
            underlying_price (float): Current underlying stock price.
            risk_free_rate (float): Annualized risk-free interest rate.

        Returns:
            Tuple[Optional[dict], Optional[dict]]:
                Best (short_put, long_put) pair if found, otherwise (None, None).
        """

        return None, None

    # ---------------------------------------------------------------------------
    # Spread Scoring: Evaluate profitability & risk of a candidate spread
    # ---------------------------------------------------------------------------
    def score_spread(self, short_put: dict, long_put: dict) -> float:
        """
        Score a bull put spread candidate.

        Metrics:
        - Reward/Risk ratio (primary driver)
        - Net credit (premium received)
        - Max risk (spread width - net credit)
        - Positive theta (benefit from time decay)
        - Liquidity and open interest penalties

        Returns:
            float: Composite score, higher is better.
        """


    # ---------------------------------------------------------------------------
    # Buying Power Check: Ensure spread fits account risk profile
    # ---------------------------------------------------------------------------
    def check_buying_power(self, short_put: Dict[str, Any], long_put: Dict[str, Any]) -> bool:
        """
        Validate whether the proposed spread fits within account buying power.

        Ensures the max risk of the spread does not exceed a configurable
        fraction of available buying power (e.g., 5%).

        Returns:
            bool: True if spread risk is acceptable, False otherwise.
        """


    # ---------------------------------------------------------------------------
    # Bull Put Spread Roll & Exit Management
    # ---------------------------------------------------------------------------
    def roll_rinse_bull_put_spread(
            self,
            short_put: Dict[str, Any],
            long_put: Dict[str, Any],
            underlying_price: float
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        """
        Manages open Bull Put Spreads by evaluating whether to:
        - HOLD the current spread,
        - EXIT the position (take profit, stop loss, or risk control),
        - ROLL the spread forward to a new expiry/strikes, or
        - return ERROR if inputs/data are invalid.

        Args:
            short_put (Dict[str, Any]): Dictionary representing the short put option leg.
            long_put (Dict[str, Any]): Dictionary representing the long put option leg.
            underlying_price (float): Current market price of the underlying stock/ETF.

        Returns:
            Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
                - action: "HOLD", "EXIT", "ROLL", or "ERROR"
                - spread_dict: updated spread dictionary if rolled, otherwise None
                - reason: explanation string for logging/diagnostics
        """

    # ===============================================================
    # Spread Execution, Order Management & State Tracking
    # ===============================================================
    # This section handles all aspects of executing multi-leg option
    # strategies (bull put spreads, iron condors, etc.) via Alpaca.
    # It includes:
    #   - Multi-leg order submission (open/close) with market or limit prices
    #   - Spread state persistence (saving/loading from disk)
    #   - Automatic cleanup of expired spreads
    #   - Synchronization with Alpaca to ensure live positions match state
    #
    # This ensures trades are executed atomically, tracked reliably,
    # and resilient across restarts or broker/API discrepancies.
    # ---------------------------------------------------------------
    def _leg_close_side(self, side_str: str | PositionSide) -> OrderSide:
        """
        Internal utility to determine the correct order side
        when closing an existing option leg.

        Args:
            side_str (str | PositionSide): Original opening side of the leg.

        Returns:
            OrderSide: Opposite side needed to close the position.

        Raises:
            ValueError: If side cannot be recognized.
        """


    def _collect_legs_for_action(self, spread: dict, action: str) -> list[OptionLegRequest]:
        """
        Build the list of legs (OptionLegRequest) for a given spread
        and desired action (open/close).

        Args:
            spread (dict): Spread dictionary containing legs (short_put, long_put, etc.).
            action (str): "open" to initiate spread, "close" to exit spread.

        Returns:
            list[OptionLegRequest]: Valid option leg requests for order submission.

        Raises:
            ValueError: If no valid legs found or invalid action passed.
        """

    def submit_mleg_open(self, spread: dict, total_qty: int, limit_price: float | None = None):
        """
        Submit a multi-leg OPEN order for a spread (bull put, iron condor, etc.).

        Args:
            spread (dict): Spread dictionary containing legs.
            total_qty (int): Number of spreads to trade (scales all legs equally).
            limit_price (float | None): Net credit (positive) or debit (negative).
                                        If None, executes as MARKET order.

        Returns:
            API response from Alpaca submit_order.
        """

    def submit_mleg_close(self, spread: dict, total_qty: int, limit_price: float | None = None):
        """
        Submit a multi-leg CLOSE order to exit a spread atomically.

        Args:
            spread (dict): Spread dictionary containing legs.
            total_qty (int): Number of spreads to close.
            limit_price (float | None): Net debit/credit. If None, executes as MARKET.

        Returns:
            API response from Alpaca submit_order.
        """

    def load_spread_state(self, path="spread_state.json"):
        """
        Load previously saved spread state from disk.

        Args:
            path (str): JSON file path where state is stored.

        Returns:
            list: List of spreads, or [] if file missing/corrupt.
        """

    def save_spread_state(self, state, path="spread_state.json"):
        """
        Persist current spread state to disk for recovery across restarts.

        Args:
            state (list): Spread list to save.
            path (str): Destination JSON file.
        """
    def cleanup_expired_spreads(self, spreads):
        """
        Remove expired spreads (expiry < today).
        Keep spreads with None/missing expiry for later evaluation.

        Args:
            spreads (list): Spread list with expiry metadata.

        Returns:
            list: Cleaned list containing only active/valid spreads.
        """

    def sync_with_alpaca(self):
        """
        Synchronize algorithm's spread state with Alpaca's live positions.

        Steps:
            1. Fetch all open option positions from Alpaca.
            2. Group them into spreads (bull put spreads, iron condors, etc.).
            3. Preserve historical metadata like entry timestamp where possible.
            4. Persist cleaned, verified spreads to local state.

        Returns:
            list: Live spreads currently tracked in Alpaca.
        """

    def evaluate_put_options(self, put_options: list, price: float, label: str = "STRICT") -> None:
        """
        Evaluate short/long put candidates, log rejection reasons, and print option details.

        Args:
            put_options (list): List of option contracts.
            price (float): Current underlying price.
            label (str): Label for logging (e.g., "STRICT", "RELAXED").
        """


    # ===============================================================
    # Main entry point for running the Bull Put Spread algorithm.
    # ===============================================================
    def run(self, market_open, market_close):
        """
        Args:
            market_open (datetime): The official U.S. market open time.
            market_close (datetime): The official U.S. market close time.

        Workflow:
            1. Initialize trading & data streams (Alpaca API).
            2. Screen and load bullish candidate assets.
            3. Fetch intraday historical data for technical indicators.
            4. Start threaded event loops for:
                - Trade updates (fills, cancels, rejections).
                - Market data streaming (1-minute bars).
            5. Continuously evaluate technicals and DTE windows
               to identify and place Bull Put Spread opportunities.
        """


        def run_trading_stream():
            async def handle_trade_updates(data):
                """
                Respond to Alpaca trade update events:
                - new, partial_fill, fill, canceled, rejected
                Updates internal state dictionaries accordingly.
                """


            self.trading_stream.subscribe_trade_updates(handle_trade_updates)
            self.trading_stream.run()

        def run_wss_client():
            async def handle_second_bar(data):
                """
                Process each incoming 1-minute bar:
                - Update rolling price/volume history.
                - Enrich with technical indicators (RSI, MACD, ADX, MAs).
                - Dynamically adjust expiry (DTE) selection window.
                - Signal Bull Put Spread entries when criteria are met.
                """


                    # ---------------------------------------------------------------------------
                    # === Entry Logic: Screening, Signal Validation, Spread Selection & Execution
                    # ---------------------------------------------------------------------------
                    # This block executes *after* each new 1-minute bar is processed and ensures
                    # we only attempt entries during a controlled window (15 min after open until 11 AM).
                    #
                    # Workflow:
                    #   1. Risk controls: skip if symbol already has open options or is in cooldown.
                    #   2. Signal validation: confirm momentum and technical score (RSI, MACD, ADX, SMA, Volume).
                    #   3. Option chain retrieval: fetch puts within 8–14 DTE.
                    #   4. Spread construction: attempt strict criteria first, fallback to relaxed.
                    #   5. Quote validation: reject illiquid or inverted quotes.
                    #   6. Risk & sizing: ensure position respects buying power and portfolio caps.
                    #   7. Execution: place multi-leg limit order and persist state.
                    # ---------------------------------------------------------------------------

                    # === Step 0: Time-based entry window (after open until 11:00 AM) ===


                    # === Exit Logic, Spread Management, and Portfolio Safeguards ===
                    # This section continuously monitors all active Bull Put Spreads during the session
                    # and applies structured exit logic to manage risk and lock in profits.
                    #
                    # Responsibilities:
                    #   1. Spread-Level Exit Logic
                    #      - Sync active spreads with Alpaca to reconcile state.
                    #      - Remove spreads that were externally closed.
                    #      - Apply trade-level exit, hold, or roll decisions via roll_rinse_bull_put_spread().
                    #
                    #   2. Portfolio-Level Exit Logic
                    #      - On Fridays, near market close, close all spreads to avoid weekend risk.
                    #      - Exit all spreads if weekly portfolio profit exceeds +5% to lock in gains.
                    #
                    # These safeguards ensure disciplined risk management at both the individual trade
                    # and portfolio level, consistent with professional trading standards.

                    # === Exit Check (spread-level monitoring) ===



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