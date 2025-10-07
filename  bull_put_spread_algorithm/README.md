# Bull Put Spread Options Trading Strategy (Teaser Repo)

This repository provides a **teaser overview** of my professional-grade Bull Put Spread trading algorithm.  
The **full implementation** (with complete signal generation, spread construction, risk controls, and live trading integration) is available for **licensed purchase**.

---

## Strategy Overview
- **Approach:** Sell out-of-the-money put credit spreads on U.S. equities/ETFs.  
- **Objective:** Generate consistent weekly premium income with defined risk.  
- **Cycle:** Enter early in the week (Mon–Tue), exit within 5 trading days (Fri).  
- **Risk Controls:** Strict position sizing, forced exits before weekends, no naked shorts.  
- **Tools:** Black-Scholes Greeks, IV Rank, and technical indicator filters.

---

## Example Backtest (Teaser)

![Sample Equity Curve](./teaser/sample_backtest_results.png)

| Trade # | Symbol | Short Put | Long Put | Net Credit | P/L |
|---------|--------|-----------|----------|------------|-----|
| 1       | AAPL   | 95 Put    | 90 Put   | $1.20      | $1.20 |
| 2       | MSFT   | 310 Put   | 305 Put  | $0.95      | $0.95 |

---

## What You’ll Find in This Repo
High-level strategy documentation  
Sample backtest results  
Teaser code (`strategy.py`) showing **how trades are logged**  
Full proprietary algorithm (reserved for licensed clients)  

---

## How to Access the Full Strategy
The complete implementation includes:
- Advanced option chain filtering  
- Automated spread construction  
- Position management & rolling logic  
- Live Alpaca API trading integration  
- Comprehensive logging and state persistence  

To purchase a **license** for the full codebase, please contact:  
**Paul K. Mwangi**  
Email: [karugumwangih@gmail.com]  
LinkedIn: [https://www.linkedin.com/in/paul-karugu-967534255/]  


