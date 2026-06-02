#  Stock Bubble Detector
### Detecting Overheated Stocks Before They Crash — Powered by Yahoo Finance

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![yfinance](https://img.shields.io/badge/Data-Yahoo%20Finance-purple.svg)](https://pypi.org/project/yfinance/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Enter any stock ticker. Get real price data. Receive a **Bubble Score (0–100)** and personalized avoidance strategies — in seconds.

---

##  The Problem This Solves

Financial bubbles have destroyed trillions of dollars in wealth. In every major crash, four warning signs appeared **before** the collapse:

| Bubble | Year | Crash | Key Signal |
|--------|------|-------|------------|
| Dot-Com | 2000 | NASDAQ −78% | RSI > 85, parabolic price acceleration |
| Housing Crisis | 2008 | S&P 500 −57% | Extreme volatility, momentum collapse |
| Crypto Mania | 2021 | Bitcoin −77% | Price 3× above 200-day MA |
| Meme Stocks | 2021 | GameStop −90% | RSI > 90, extreme acceleration |

This tool quantifies all four signals in real time, with real data.

---

##  Project Structure

```
stock-bubble-detector/
│
├── stock_bubble_detector.py       # Main Python script — 7 sections, 20+ functions
├── stock_bubble_detector.ipynb    # Jupyter Notebook — step-by-step walkthrough
├── requirements.txt               # pip dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

##  Quick Start

### 1. Clone the Repo
```bash
git clone https://github.com/YOUR_USERNAME/stock-bubble-detector.git
cd stock-bubble-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3a. Run the Interactive CLI
```bash
python stock_bubble_detector.py
```

```
======================================================
    STOCK BUBBLE DETECTOR
    Powered by Yahoo Finance and Python
======================================================

  STOCK BUBBLE ANALYZER -- INPUT
  Enter stock ticker symbol (e.g. AAPL, TSLA, NVDA): NVDA
  How many years of data to analyze? (1-5, default 2): 2

  Downloading data for NVDA ...
  OK -- NVDA: 503 trading days loaded.

  =============================================
  STOCK SUMMARY: NVDA
  =============================================
  Current Price       : $875.40
  Total Return        : +185.3%
  Annualized Volatility: 58.2%
  Max Drawdown        : -31.4%
  Current RSI (14-day): 74.2
  Price / 200-day MA  : 1.87x
  =============================================

  BUBBLE REPORT: NVDA
  Bubble Score : 75 / 100
  Risk Level   : CRITICAL BUBBLE RISK

  RSI Score              30 pts  [##############################]
  Moving Avg Score       30 pts  [##############################]
  Volatility Score       10 pts  [##########--------------------]
  Acceleration Score      5 pts  [#####-------------------------]
```

### 3b. Run the Jupyter Notebook
```bash
jupyter notebook stock_bubble_detector.ipynb
```

---

##  The Four Bubble Indicators

The bubble score combines four independent statistical checks into one number (0–100).

| # | Indicator | What It Detects | Max Score |
|---|-----------|----------------|-----------|
| 1 | **RSI Analysis** | Overbought momentum — RSI > 70 is warning, > 80 is critical | 30 pts |
| 2 | **Price vs 200-Day MA** | Price far above long-term "fair value" baseline | 30 pts |
| 3 | **Volatility** | Annualized price swings — panic and speculation | 20 pts |
| 4 | **Price Acceleration** | Parabolic moves — recent 30 days vs prior 30 days | 20 pts |

### Risk Level Guide

| Bubble Score | Risk Level | Recommended Action |
|-------------|-----------|-------------------|
| 0–29 |  Healthy | Hold or accumulate (dollar-cost average) |
| 30–49 |  Moderate | Monitor weekly, no new buys |
| 50–69 |  High | Trim 25–40%, set 15% trailing stop-loss |
| 70–100 |  Critical | Reduce 50–75% urgently, hedge with put options |

---

##  Code Structure — 7 Sections

```
SECTION 1  load_stock_data()             Download OHLCV data from Yahoo Finance
SECTION 2  get_current_price()           Latest closing price
           get_total_return()            % gain/loss start → end
           get_daily_returns()           Day-by-day % changes
           get_volatility()              Annualized standard deviation
           get_moving_average()          N-day simple moving average
           get_rsi()                     Relative Strength Index (0–100)
           get_price_to_200ma_ratio()    Price ÷ 200-day MA
           get_max_drawdown()            Largest peak-to-trough fall
SECTION 3  check_rsi()                   RSI check → 0–30 pts + warnings
           check_moving_average()        MA ratio check → 0–30 pts + warnings
           check_volatility()            Volatility check → 0–20 pts + warnings
           check_acceleration()          Parabolic check → 0–20 pts + warnings
           run_bubble_analysis()         Combine all four → result dict
SECTION 4  generate_strategies()         Risk-scaled avoidance strategy list
SECTION 5  print_stock_summary()         Formatted metric printout
           print_bubble_report()         Formatted bubble score report
           print_strategies()            Formatted strategy list
SECTION 6  plot_stock_detail()           Two-panel Price + RSI chart
SECTION 7  get_user_input()              Interactive CLI input with validation
           ask_another()                 Prompt to analyze another stock
           main()                        Full interactive analysis loop
```

---

##  Charts Produced

| Chart | How To Generate | What It Shows |
|-------|----------------|---------------|
| Price + MA + RSI | `plot_stock_detail(ticker, data)` | Price trend, moving averages, RSI with overbought/oversold zones |
| Return Distribution | Notebook Section 2.3 | Daily return histogram with normal distribution overlay |
| RSI Over Time | Notebook Section 2.4 | Full RSI history with shaded zones |
| Price vs Moving Averages | Notebook Section 2.5 | Price, 50-day MA, 200-day MA, danger zone shading |
| Drawdown Chart | Notebook Section 2.6 | Peak-to-trough falls annotated with maximum |
| Score Breakdown | Notebook Section 3.3 | Bar chart of each indicator's contribution |
| Watchlist Leaderboard | Notebook Section 6.2 | Color-coded scores for multiple stocks |
| Risk vs Return Scatter | Notebook Section 6.3 | Volatility vs return with bubble score as dot size |

---

##  Python Concepts Demonstrated

| Concept | Where Used |
|---------|-----------|
| Variables and data types | Throughout all functions |
| Lists and dictionaries | Strategy generation, result storage |
| For loops and conditionals | Bubble checks, CLI loop, strategy selection |
| Functions with return values | All 20+ functions |
| `try / except` error handling | `load_stock_data()` |
| Input validation with `while` | `get_user_input()`, `ask_another()` |
| f-strings and string formatting | All print functions |
| pandas DataFrame operations | Rolling averages, pct_change, cummax |
| NumPy math | Volatility annualization, RSI formula |
| Matplotlib multi-panel charts | `plot_stock_detail()`, notebook charts |

---

##  Requirements

```
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
jupyterlab>=3.0.0
```

Install everything at once:
```bash
pip install -r requirements.txt
```

---

##  License

MIT License — free to use, modify, and distribute for any purpose.

---

*Built with Python 3 · yfinance · pandas · numpy · matplotlib*  
