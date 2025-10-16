Bull Put Spread Options Trading Strategy
Prepared for: Potential Investor Prepared by: Algorithm Development Team
Date: September 11, 2025
Executive Summary
This document describes a professional, industry-standard implementation of a Bull Put Spread algorithmic trading
strategy. The strategy targets consistent option premium income by selling out-of-the-money (OTM) put credit
spreads on U.S. equities and ETFs. Positions are entered early in the trading week and managed for a short time
horizon (typical holding period: 5 trading days) with rigorous risk controls.
Strategy Overview
Objective: Generate repeatable income with a defined downside using Bull Put Spreads (credit spreads).
Instruments: U.S. equity and ETF options.
Holding Period: Short-term; primary cadence is Monday entry ® Friday exit (5-day cycle).
Order Type: Limit orders only; multi-leg order handling for clean fills.
Universe: Pre-screened bullish instruments by liquidity and technical indicators.
Mathematical Framework & Payoff
A Bull Put Spread consists of:
- Sell 1 put at strike K_s (short put).
- Buy 1 put at strike K_l < K_s (long put) to cap downside.
Net credit at entry: C = premium_short - premium_long.
Payoff at expiration (S_T):
Profit(S_T) = C - max(K_s - S_T, 0) + max(K_l - S_T, 0).
Key quantities:
- Maximum Profit = C (credit received), realized if S_T >= K_s.
- Maximum Loss = (K_s - K_l) - C, realized if S_T <= K_l.
- Break-even = K_s - C.
Pricing & Greeks (used in selection and risk management):
Use Black-Scholes-Merton (for European-style approximation) to estimate option fair value and Greeks:
- d1 = [ln(S/K) + (r + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
- d2 = d1 - sigma * sqrt(T)
- Call and put prices derived from standard BSM formulas (we use consistent implied volatility inputs from market
data).
We prioritize: delta (short put target ~ -0.08 to -0.15), theta (positive portfolio theta), and IV Rank to ensure
adequate premium.
Illustrative Example: Payoff Diagram
Illustrative Example Trade (Numbers)
Underlying (S0) $100.00
Short Put Strike (K_s) $95.00
Long Put Strike (K_l) $90.00
Short Premium (received) $1.50
Long Premium (paid) $0.30
Net Credit (C) $1.20
Max Profit $1.20
Max Loss $3.80
Break-even $93.80
Operational Workflow
The following flowchart summarizes the algorithm's operational steps from screening to exit.
Scoring, Selection & Risk Controls
Scoring heuristic (example weights):
- Technical Score (RSI/MACD/SMA/ADX): 35%
- Option Liquidity & Spread Width (OI, bid-ask): 20%
- Implied Volatility & IV Rank: 20%
- Risk/Reward (Credit relative to width): 15%
- Portfolio Diversification / Correlation impact: 10%
Risk Controls include:
- No naked shorts; always paired with long put.
- Position sizing capped by buying power and a per-symbol allocation limit.
- Forced weekly exit to avoid weekend gap risk.
- Pre-trade checks for spreads that can be closed in a single multi-leg order.
- Logging, state persistence, and reconciliation with broker fills.
Appendices & Assumptions
Assumptions & Limitations:
- Options are American-style in US markets; Black-Scholes used as approximation for pricing and Greeks.
- Execution may differ from theoretical prices due to bid-ask, partial fills, and slippage.
- Historical backtest results should be appended when available; this document focuses on strategy design and
operational controls.
Next steps for investor review:
1. Provide historical performance (backtest) to populate expected returns and Sharpe ratios.
2. Operational readiness checklist: paper/live broker test, monitoring, and failover procedures.
3. Compliance review and capital allocation proposal.
