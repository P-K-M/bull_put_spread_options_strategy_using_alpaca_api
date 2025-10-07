"""
utils.py.

Author: Paul K. Mwangi
"""

# Imports and basic configuration
# Purpose: group and document all imports and top-level configuration used across
# the Bull Put Spread strategy.

# -------------------------
# Standard library imports
# -------------------------

from datetime import date


# -------------------------
# Typing and developer helpers
# -------------------------
# Used to provide type hints which improves readability and helps code review.
from typing import Optional, Dict, Any, List, Tuple


# -------------------------
# Alpaca (data + trading)
# -------------------------
# Alpaca packages provide both historical & live data access and trading client
# objects. They are central for live/paper trading and option chain retrieval.

from alpaca.trading.requests import (
    OptionLegRequest
)
from alpaca.trading.enums import (
    OrderSide,
    ContractType,
    PositionSide
)

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