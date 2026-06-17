# =============================================================
# stock_analyzer.py
# Stock Bubble Detector 
# =============================================================
# This script lets you enter any stock ticker, downloads its
# price history from Yahoo Finance, computes key risk metrics,
# and tells you whether the stock shows signs of a bubble.
#
# Libraries required (install once with pip):
#   pip install yfinance pandas numpy matplotlib
# =============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# -------------------------------------------------------------
# SECTION 1: DATA LOADING
# -------------------------------------------------------------

def load_stock_data(ticker, start_date, end_date):
    """
    Download historical daily price data from Yahoo Finance.

    Parameters:
        ticker     : str -- Stock symbol, e.g. 'AAPL'
        start_date : str -- Start date in 'YYYY-MM-DD' format
        end_date   : str -- End date in 'YYYY-MM-DD' format

    Returns:
        pandas DataFrame with columns [Open, High, Low, Close, Volume],
        or None if the download failed or the ticker was invalid.
    """
    ticker = ticker.upper().strip()
    print(f"\n  Downloading data for {ticker} ...")

    try:
        ticker_obj = yf.Ticker(ticker)

        # .history() fetches OHLCV data for the given date range
        data = ticker_obj.history(start=start_date, end=end_date)

        if data.empty:
            print(f"  WARNING: No data returned for '{ticker}'.")
            print("  Check that the ticker symbol is correct and try again.")
            return None

        print(f"  OK -- {ticker}: {len(data)} trading days loaded "
              f"({start_date} to {end_date})")
        return data

    except Exception as error:
        print(f"  ERROR downloading {ticker}: {error}")
        return None


# -------------------------------------------------------------
# SECTION 2: METRIC CALCULATIONS
# -------------------------------------------------------------

def get_current_price(data):
    """
    Return the most recent closing price.

    Parameters:
        data : DataFrame -- Stock price data from load_stock_data()

    Returns:
        float -- Latest closing price rounded to 2 decimal places.
    """
    return round(float(data['Close'].iloc[-1]), 2)


def get_total_return(data):
    """
    Calculate the percentage gain/loss from first to last price.

    Formula: ((end_price - start_price) / start_price) * 100

    Example: Bought at $100, now $130 -> total return = +30.0%

    Returns:
        float -- Total return as a percentage (e.g. 30.0 means +30%).
    """
    if len(data) < 2:
        return 0.0

    start_price = float(data['Close'].iloc[0])
    end_price   = float(data['Close'].iloc[-1])
    return round(((end_price - start_price) / start_price) * 100, 2)


def get_daily_returns(data):
    """
    Calculate the percentage price change for each trading day.

    Example: Price goes $100 -> $103, daily return = +3.0%

    Returns:
        pandas Series of daily return values (decimal, not percent).
    """
    return data['Close'].pct_change().dropna()


def get_volatility(data):
    """
    Calculate annualized volatility -- a standard measure of price risk.

    High volatility means prices swing wildly (risky).
    Low volatility means prices are stable.

    We annualize by multiplying the daily standard deviation by
    sqrt(252), because there are 252 trading days per year.

    Returns:
        float -- Annualized volatility as a percentage (e.g. 35.0 = 35%).
    """
    daily_returns = get_daily_returns(data)

    if len(daily_returns) == 0:
        return 0.0

    # std() = standard deviation -- measures how spread out returns are
    daily_std  = daily_returns.std()
    annual_std = daily_std * (252 ** 0.5)
    return round(float(annual_std * 100), 2)


def get_moving_average(data, window):
    """
    Calculate a simple moving average over a rolling window of days.

    A moving average smooths out short-term noise to reveal the trend.
    Example: The 50-day MA is the average of the last 50 closing prices,
    recalculated each day.

    Parameters:
        data   : DataFrame -- Stock price data
        window : int       -- Number of days in the rolling window (e.g. 50)

    Returns:
        pandas Series of moving average values.
    """
    return data['Close'].rolling(window=window).mean()


def get_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI).

    RSI is a momentum indicator scaled from 0 to 100:
        RSI > 70  -- Overbought (prices rose too fast, potential bubble)
        RSI < 30  -- Oversold  (prices fell too fast, potential bargain)
        RSI 40-60 -- Normal range

    Parameters:
        data   : DataFrame -- Stock price data
        period : int       -- Lookback period in days (default 14)

    Returns:
        pandas Series of RSI values.
    """
    price_change = data['Close'].diff()

    # Separate positive days (gains) from negative days (losses)
    gains  = price_change.copy()
    losses = price_change.copy()
    gains[gains < 0]   = 0        # Zero out losses in the gains series
    losses[losses > 0] = 0        # Zero out gains in the losses series
    losses = abs(losses)          # Make losses positive for the formula

    # Rolling average of gains and losses over the lookback period
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses..ewm(com=period - 1, min_periods=period).mean()

    # Guard against division by zero
    avg_loss = avg_loss.replace(0, 0.0001)

    # RSI formula
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_price_to_200ma_ratio(data):
    """
    Calculate how far the current price is above the 200-day moving average.

    This ratio is a widely used bubble indicator:
        1.0  -- Price equals the 200-day MA (normal)
        1.30 -- Price is 30% above the 200-day MA (warning)
        1.50 -- Price is 50% above the 200-day MA (danger)

    During the dot-com bubble, many tech stocks were 3x-5x above their
    200-day MAs just before they collapsed.

    Returns:
        pandas Series of ratio values, or None if not enough data.
    """
    if len(data) < 200:
        print("  NOTE: Less than 200 days of data -- MA ratio unavailable.")
        return None

    ma_200 = get_moving_average(data, 200)
    return data['Close'] / ma_200


def get_max_drawdown(data):
    """
    Calculate the maximum drawdown -- the largest peak-to-trough decline.

    Example: Stock peaks at $200 then falls to $120 -> drawdown = -40%.

    This tells you the worst-case loss you would have suffered if you
    bought at the peak and sold at the trough.

    Returns:
        float -- Maximum drawdown as a negative percentage (e.g. -40.0).
    """
    prices      = data['Close']
    rolling_max = prices.cummax()              # Highest price seen so far
    drawdown    = (prices - rolling_max) / rolling_max * 100
    return round(float(drawdown.min()), 2)     # Most negative value


# -------------------------------------------------------------
# SECTION 3: BUBBLE DETECTION
# -------------------------------------------------------------

# Threshold constants used by the bubble detection checks
RSI_WARNING          = 70    # RSI above this is overbought
RSI_DANGER           = 80    # RSI above this is severely overbought
MA_RATIO_WARNING     = 1.30  # 30% above 200-day MA
MA_RATIO_DANGER      = 1.50  # 50% above 200-day MA
VOLATILITY_HIGH      = 40    # 40% annualized volatility
VOLATILITY_EXTREME   = 65    # 65% annualized volatility


def check_rsi(data):
    """
    Score the stock's RSI for overbought conditions.

    Scoring:
        RSI > 80 (danger)   : 30 points
        RSI > 70 (warning)  : 15 points
        RSI > 70 for 20+ of the last 30 days (sustained) : +10 points

    Parameters:
        data : DataFrame -- Stock price data

    Returns:
        score    : int  -- Points contributed to the bubble score (0-30)
        warnings : list -- Warning message strings triggered
    """
    rsi_series  = get_rsi(data)
    current_rsi = float(rsi_series.iloc[-1])
    score       = 0
    warnings    = []

    if current_rsi > RSI_DANGER:
        score = 30
        warnings.append(
            f"EXTREME RSI: {current_rsi:.1f} -- severely overbought "
            f"(danger threshold: {RSI_DANGER})"
        )
    elif current_rsi > RSI_WARNING:
        score = 15
        warnings.append(
            f"HIGH RSI: {current_rsi:.1f} -- overbought "
            f"(warning threshold: {RSI_WARNING})"
        )

    # Check how many of the last 30 trading days had RSI above 70
    recent_30      = rsi_series.tail(30)
    days_overbought = int((recent_30 > 70).sum())

    if days_overbought >= 20:
        score += 10
        warnings.append(
            f"SUSTAINED OVERBOUGHT: RSI above 70 for "
            f"{days_overbought} of the last 30 trading days"
        )

    return min(30, score), warnings


def check_moving_average(data):
    """
    Score how far the price has extended above its 200-day moving average.

    Scoring:
        Price > 50% above 200-day MA : 30 points
        Price > 30% above 200-day MA : 15 points

    Returns:
        score    : int  -- Points contributed to the bubble score (0-30)
        warnings : list -- Warning message strings triggered
    """
    ratio_series = get_price_to_200ma_ratio(data)

    if ratio_series is None:
        return 0, []

    current_ratio = float(ratio_series.iloc[-1])
    score         = 0
    warnings      = []
    pct_above     = (current_ratio - 1) * 100

    if current_ratio > MA_RATIO_DANGER:
        score = 30
        warnings.append(
            f"DANGER: Price is {pct_above:.1f}% above 200-day MA "
            f"(threshold: 50%)"
        )
    elif current_ratio > MA_RATIO_WARNING:
        score = 15
        warnings.append(
            f"WARNING: Price is {pct_above:.1f}% above 200-day MA "
            f"(threshold: 30%)"
        )

    return score, warnings


def check_volatility(data):
    """
    Score the stock's annualized volatility for extreme readings.

    Scoring:
        Volatility > 65% : 20 points
        Volatility > 40% : 10 points

    Returns:
        score    : int  -- Points contributed to the bubble score (0-20)
        warnings : list -- Warning message strings triggered
    """
    annual_vol = get_volatility(data)
    score      = 0
    warnings   = []

    if annual_vol > VOLATILITY_EXTREME:
        score = 20
        warnings.append(
            f"EXTREME VOLATILITY: {annual_vol:.1f}% annualized "
            f"(threshold: {VOLATILITY_EXTREME}%)"
        )
    elif annual_vol > VOLATILITY_HIGH:
        score = 10
        warnings.append(
            f"HIGH VOLATILITY: {annual_vol:.1f}% annualized "
            f"(threshold: {VOLATILITY_HIGH}%)"
        )

    return score, warnings


def check_acceleration(data):
    """
    Detect parabolic (accelerating) price moves -- a classic bubble sign.

    During bubbles, prices tend to rise faster and faster over time.
    This function compares returns from the last 30 days against
    returns from the 30 days before that.

    If the recent 30-day gain is more than 2x the prior 30-day gain
    and both windows are positive, that is a parabolic warning.

    Scoring:
        Recent 30 days > 2x prior 30 days and > +20% : 20 points
        Recent 30 days > 1.5x prior 30 days and > +10%: 10 points

    Returns:
        score    : int  -- Points contributed to the bubble score (0-20)
        warnings : list -- Warning message strings triggered
    """
    if len(data) < 60:
        return 0, []

    prices = data['Close']

    # Split the last 60 trading days into two 30-day windows
    recent_window = prices.tail(30)
    prior_window  = prices.tail(60).head(30)

    recent_return = ((float(recent_window.iloc[-1]) - float(recent_window.iloc[0]))
                     / float(recent_window.iloc[0])) * 100
    prior_return  = ((float(prior_window.iloc[-1])  - float(prior_window.iloc[0]))
                     / float(prior_window.iloc[0]))  * 100

    score    = 0
    warnings = []

    if (recent_return > 20
            and prior_return > 0
            and recent_return > (prior_return * 2)):
        score = 20
        warnings.append(
            f"PARABOLIC MOVE: +{recent_return:.1f}% last 30 days vs "
            f"+{prior_return:.1f}% prior 30 days -- accelerating sharply"
        )
    elif (recent_return > 10
            and prior_return > 0
            and recent_return > (prior_return * 1.5)):
        score = 10
        warnings.append(
            f"ACCELERATING: +{recent_return:.1f}% last 30 days vs "
            f"+{prior_return:.1f}% prior 30 days"
        )

    return score, warnings


def assign_risk_level(bubble_score):
    """
    Convert a numeric bubble score (0-100) to a text risk label.

    Parameters:
        bubble_score : int -- Overall bubble score

    Returns:
        str -- Human-readable risk level.
    """
    if bubble_score >= 70:
        return "CRITICAL BUBBLE RISK"
    elif bubble_score >= 50:
        return "HIGH BUBBLE RISK"
    elif bubble_score >= 30:
        return "MODERATE BUBBLE RISK"
    elif bubble_score >= 10:
        return "LOW BUBBLE RISK"
    else:
        return "HEALTHY -- No Bubble Detected"


def run_bubble_analysis(ticker, data):
    """
    Run all four bubble checks and compute the overall bubble score.

    The four checks are:
        1. RSI overbought check         (max 30 points)
        2. Price vs 200-day MA check    (max 30 points)
        3. Volatility check             (max 20 points)
        4. Price acceleration check     (max 20 points)

    Total bubble score is capped at 100.

    Parameters:
        ticker : str       -- Stock symbol (used for display only)
        data   : DataFrame -- Stock price data from load_stock_data()

    Returns:
        dict with keys: ticker, bubble_score, risk_level,
                        warnings, sub_scores
    """
    print(f"\n  Running bubble analysis on {ticker} ...")

    rsi_score,   rsi_warnings   = check_rsi(data)
    ma_score,    ma_warnings    = check_moving_average(data)
    vol_score,   vol_warnings   = check_volatility(data)
    accel_score, accel_warnings = check_acceleration(data)

    all_warnings  = rsi_warnings + ma_warnings + vol_warnings + accel_warnings
    bubble_score  = min(100, rsi_score + ma_score + vol_score + accel_score)
    risk_level    = assign_risk_level(bubble_score)

    sub_scores = {
        'RSI Score':          rsi_score,
        'Moving Avg Score':   ma_score,
        'Volatility Score':   vol_score,
        'Acceleration Score': accel_score,
    }

    return {
        'ticker':       ticker,
        'bubble_score': bubble_score,
        'risk_level':   risk_level,
        'warnings':     all_warnings,
        'sub_scores':   sub_scores,
    }


# -------------------------------------------------------------
# SECTION 4: AVOIDANCE STRATEGIES
# -------------------------------------------------------------

def generate_strategies(ticker, current_price, bubble_score):
    """
    Return a list of actionable strategies based on the bubble score.

    Each strategy is a dict with:
        priority : str -- 'URGENT', 'HIGH', 'MEDIUM', 'LOW', or 'INFO'
        action   : str -- Short action headline
        detail   : str -- Longer explanation

    Parameters:
        ticker        : str   -- Stock ticker symbol
        current_price : float -- Most recent closing price
        bubble_score  : int   -- Overall bubble score (0-100)

    Returns:
        list of strategy dicts.
    """
    strategies = []

    if bubble_score >= 70:
        strategies = [
            {
                'priority': 'URGENT',
                'action':   'REDUCE POSITION BY 50-75%',
                'detail':   (f"Sell 50-75% of your {ticker} shares now to lock "
                             f"in gains. Current price: ${current_price}.")
            },
            {
                'priority': 'URGENT',
                'action':   f'SET STOP-LOSS AT ${current_price * 0.90:.2f}',
                'detail':   (f"Place an automatic sell order at "
                             f"${current_price * 0.90:.2f} (10% below current price). "
                             f"This caps your worst-case downside.")
            },
            {
                'priority': 'HIGH',
                'action':   'HEDGE WITH PUT OPTIONS',
                'detail':   (f"Buy put options on {ticker} as insurance. "
                             f"A put gains value if the stock drops -- "
                             f"a standard hedging technique at risk desks.")
            },
            {
                'priority': 'HIGH',
                'action':   'ROTATE INTO DEFENSIVE SECTORS',
                'detail':   ("Move proceeds into utilities (XLU), healthcare (XLV), "
                             "or consumer staples (XLP) -- sectors that historically "
                             "hold up better during market crashes.")
            },
            {
                'priority': 'MEDIUM',
                'action':   f'PLAN RE-ENTRY NEAR ${current_price * 0.65:.2f}',
                'detail':   (f"Bubbles often mean-revert fully to the 200-day MA. "
                             f"Consider re-entering {ticker} near that level.")
            },
            {
                'priority': 'MEDIUM',
                'action':   'RAISE CASH RESERVE TO 20%',
                'detail':   ("Holding 20% cash lets you buy quality assets cheaply "
                             "after the bubble bursts.")
            },
        ]

    elif bubble_score >= 50:
        strategies = [
            {
                'priority': 'HIGH',
                'action':   'TRIM POSITION BY 25-40%',
                'detail':   (f"Sell 25-40% of {ticker} to reduce exposure and "
                             f"book partial profits at ${current_price}.")
            },
            {
                'priority': 'HIGH',
                'action':   'SET 15% TRAILING STOP-LOSS',
                'detail':   (f"A trailing stop follows the price upward, triggering "
                             f"a sell if it drops 15% from peak. "
                             f"Currently that would be ${current_price * 0.85:.2f}.")
            },
            {
                'priority': 'MEDIUM',
                'action':   'CAP POSITION AT 10% OF PORTFOLIO',
                'detail':   ("No single stock should exceed 10% of your total "
                             "portfolio. Concentration in a bubble stock magnifies loss.")
            },
            {
                'priority': 'MEDIUM',
                'action':   'MONITOR RSI WEEKLY',
                'detail':   ("If RSI crosses 80, escalate to the CRITICAL protocol. "
                             "Set a weekly reminder to rerun this analysis.")
            },
            {
                'priority': 'LOW',
                'action':   'DO NOT ADD TO POSITION',
                'detail':   (f"Avoid buying more {ticker} until the bubble score "
                             f"drops below 25. Never average up into a bubble.")
            },
        ]

    elif bubble_score >= 30:
        strategies = [
            {
                'priority': 'MEDIUM',
                'action':   'MONITOR WEEKLY',
                'detail':   (f"{ticker} shows early warning signs. Not critical yet, "
                             f"but requires weekly review.")
            },
            {
                'priority': 'MEDIUM',
                'action':   f'MENTAL STOP-LOSS AT ${current_price * 0.88:.2f}',
                'detail':   (f"If {ticker} falls below ${current_price * 0.88:.2f} "
                             f"(12% below current), treat it as a sell signal.")
            },
            {
                'priority': 'LOW',
                'action':   'HOLD -- DO NOT ADD YET',
                'detail':   ("Hold your current position but do not add more until "
                             "the bubble score falls back below 20.")
            },
            {
                'priority': 'LOW',
                'action':   'KEEP PORTFOLIO DIVERSIFIED',
                'detail':   ("Hold stocks across at least 5 different sectors so that "
                             "one sector crash does not devastate the whole portfolio.")
            },
        ]

    else:
        strategies = [
            {
                'priority': 'LOW',
                'action':   'HOLD OR ACCUMULATE',
                'detail':   (f"{ticker} shows no significant bubble signals. "
                             f"Normal buy-and-hold or dollar-cost averaging applies.")
            },
            {
                'priority': 'LOW',
                'action':   'DOLLAR-COST AVERAGE (DCA)',
                'detail':   ("Invest a fixed amount on a regular schedule regardless "
                             "of price. This reduces the impact of bad timing.")
            },
            {
                'priority': 'INFO',
                'action':   'RERUN THIS ANALYSIS MONTHLY',
                'detail':   ("Conditions change. Schedule a monthly check to catch "
                             "developing bubble signals early.")
            },
        ]

    return strategies


# -------------------------------------------------------------
# SECTION 5: PRINTING / REPORTING
# -------------------------------------------------------------

def print_stock_summary(ticker, data):
    """
    Print a clean summary of key metrics for a single stock.

    Parameters:
        ticker : str       -- Stock symbol
        data   : DataFrame -- Stock price data
    """
    price    = get_current_price(data)
    ret      = get_total_return(data)
    vol      = get_volatility(data)
    max_dd   = get_max_drawdown(data)
    rsi_val  = float(get_rsi(data).iloc[-1])

    ratio_series = get_price_to_200ma_ratio(data)
    ratio_str = (f"{float(ratio_series.iloc[-1]):.2f}x"
                 if ratio_series is not None else "N/A (< 200 days)")

    print(f"\n  {'=' * 45}")
    print(f"  STOCK SUMMARY: {ticker}")
    print(f"  {'=' * 45}")
    print(f"  Current Price       : ${price}")
    print(f"  Total Return        : {ret:+.1f}%")
    print(f"  Annualized Volatility: {vol:.1f}%")
    print(f"  Max Drawdown        : {max_dd:.1f}%")
    print(f"  Current RSI (14-day): {rsi_val:.1f}")
    print(f"  Price / 200-day MA  : {ratio_str}")
    print(f"  {'=' * 45}")


def print_bubble_report(result):
    """
    Print the full bubble analysis report to the console.

    Parameters:
        result : dict -- Output from run_bubble_analysis()
    """
    print(f"\n  {'=' * 50}")
    print(f"  BUBBLE REPORT: {result['ticker']}")
    print(f"  {'=' * 50}")
    print(f"  Bubble Score : {result['bubble_score']} / 100")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"\n  Score Breakdown:")

    for indicator, score in result['sub_scores'].items():
        bar = '#' * score + '-' * (30 - score)
        print(f"    {indicator:<22} {score:>3} pts  [{bar}]")

    print(f"\n  Warning Flags ({len(result['warnings'])} found):")
    if result['warnings']:
        for warning in result['warnings']:
            print(f"    [!]  {warning}")
    else:
        print("    [OK] No warning flags triggered.")

    print(f"  {'=' * 50}")


def print_strategies(ticker, bubble_score, strategies):
    """
    Print all recommended avoidance strategies to the console.

    Parameters:
        ticker       : str  -- Stock symbol
        bubble_score : int  -- Overall bubble score
        strategies   : list -- Output from generate_strategies()
    """
    priority_labels = {
        'URGENT': '[URGENT]',
        'HIGH':   '[HIGH  ]',
        'MEDIUM': '[MEDIUM]',
        'LOW':    '[LOW   ]',
        'INFO':   '[INFO  ]',
    }

    print(f"\n  {'=' * 50}")
    print(f"  AVOIDANCE STRATEGIES: {ticker}")
    print(f"  Bubble Score: {bubble_score} / 100")
    print(f"  {'=' * 50}")

    for i, strat in enumerate(strategies, start=1):
        label = priority_labels.get(strat['priority'], '[     ]')
        print(f"\n  {i}. {label} {strat['action']}")

        # Word-wrap the detail text at 60 characters
        words = strat['detail'].split()
        line  = "     "
        for word in words:
            if len(line) + len(word) > 62:
                print(line)
                line = "     " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    print(f"\n  {'=' * 50}")


# -------------------------------------------------------------
# SECTION 6: CHARTING
# -------------------------------------------------------------

def plot_stock_detail(ticker, data):
    """
    Produce a two-panel chart for one stock:
        Top panel    : Closing price with 50-day and 200-day moving averages.
                       Red shading marks where price is >30% above 200-day MA.
        Bottom panel : RSI with overbought (70) and oversold (30) zones.

    The chart is saved as a .png file and displayed on screen.

    Parameters:
        ticker : str       -- Stock symbol (used for title and filename)
        data   : DataFrame -- Stock price data
    """
    # gridspec_kw sets the height ratio: price panel is 3x taller than RSI panel
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )
    fig.suptitle(f'{ticker} -- Price and RSI Analysis',
                 fontsize=16, fontweight='bold')

    # ---- Top panel: price and moving averages ----
    prices = data['Close']
    ma_50  = get_moving_average(data, 50)
    ma_200 = get_moving_average(data, 200)

    ax_price.plot(prices.index, prices, color='#1565C0',
                  linewidth=1.2, label='Close Price')
    ax_price.plot(ma_50.index, ma_50, color='#FF9800',
                  linewidth=1.5, linestyle='--', label='50-Day MA')
    ax_price.plot(ma_200.index, ma_200, color='#F44336',
                  linewidth=1.8, linestyle='-.', label='200-Day MA')

    # Shade "danger zone" where price is more than 30% above the 200-day MA
    ratio = get_price_to_200ma_ratio(data)
    if ratio is not None:
        bubble_zone = ratio > MA_RATIO_WARNING
        ax_price.fill_between(
            prices.index, prices, ma_200,
            where=(bubble_zone & prices.notna() & ma_200.notna()),
            alpha=0.12, color='red',
            label=f'Danger Zone (>{int((MA_RATIO_WARNING - 1)*100)}% above 200MA)'
        )

    ax_price.set_ylabel('Price (USD)', fontsize=12)
    ax_price.legend(loc='upper left', fontsize=10)

    # ---- Bottom panel: RSI ----
    rsi = get_rsi(data)

    ax_rsi.plot(rsi.index, rsi, color='#7B1FA2',
                linewidth=1.5, label='RSI (14-day)')
    ax_rsi.axhline(70, color='#F44336', linestyle='--',
                   linewidth=1.2, label='Overbought (70)')
    ax_rsi.axhline(30, color='#388E3C', linestyle='--',
                   linewidth=1.2, label='Oversold (30)')
    ax_rsi.axhline(50, color='gray', linestyle=':', linewidth=0.8)

    ax_rsi.fill_between(rsi.index, 70, rsi, where=(rsi > 70),
                        alpha=0.25, color='#F44336')
    ax_rsi.fill_between(rsi.index, 30, rsi, where=(rsi < 30),
                        alpha=0.25, color='#388E3C')

    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('RSI', fontsize=12)
    ax_rsi.set_xlabel('Date', fontsize=12)
    ax_rsi.legend(loc='upper left', fontsize=9)

    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_rsi.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_rsi.xaxis.get_majorticklabels(), rotation=45, ha='right')

    filename = f'{ticker}_analysis.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved: {filename}")


# -------------------------------------------------------------
# SECTION 7: USER INPUT AND MAIN LOOP
# -------------------------------------------------------------

def get_user_input():
    """
    Prompt the user to enter a ticker symbol and analysis period.

    Validates that:
        - The ticker is not blank.
        - The number of years is a positive integer.

    Returns:
        ticker     : str -- Uppercased stock symbol
        years_back : int -- Number of years of history to pull
    """
    print("\n  -----------------------------------------------")
    print("  STOCK BUBBLE ANALYZER -- INPUT")
    print("  -----------------------------------------------")

    # Get ticker symbol
    while True:
        ticker = input("  Enter stock ticker symbol (e.g. AAPL, TSLA, NVDA): ").strip()
        if ticker:
            ticker = ticker.upper()
            break
        print("  Ticker cannot be blank. Please try again.")

    # Get analysis period
    while True:
        years_input = input("  How many years of data to analyze? (1-5, default 2): ").strip()

        if years_input == "":
            years_back = 2
            break

        if years_input.isdigit():
            years_back = int(years_input)
            if 1 <= years_back <= 5:
                break
            else:
                print("  Please enter a number between 1 and 5.")
        else:
            print("  Invalid input. Please enter a whole number.")

    return ticker, years_back


def ask_another():
    """
    Ask the user if they want to analyze another stock.

    Returns:
        bool -- True if the user wants to continue, False to exit.
    """
    while True:
        answer = input("\n  Analyze another stock? (yes / no): ").strip().lower()
        if answer in ('yes', 'y'):
            return True
        elif answer in ('no', 'n'):
            return False
        else:
            print("  Please type 'yes' or 'no'.")


def main():
    """
    Main entry point. Runs an interactive loop that:
        1. Prompts the user for a ticker and date range.
        2. Downloads the stock data.
        3. Prints a metric summary.
        4. Runs the bubble analysis and prints the report.
        5. Prints avoidance strategies scaled to the risk level.
        6. Plots the price/RSI chart.
        7. Asks whether to analyze another stock.
    """
    print("\n" + "=" * 55)
    print("    STOCK BUBBLE DETECTOR")
    print("    Powered by Yahoo Finance and Python")
    print("=" * 55)

    while True:
        # Step 1: Collect user input
        ticker, years_back = get_user_input()

        # Step 2: Calculate the date range
        end_date   = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today()
                      - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')

        print(f"\n  Fetching {years_back} year(s) of data "
              f"({start_date} to {end_date}) ...")

        # Step 3: Download data
        data = load_stock_data(ticker, start_date, end_date)

        if data is None:
            print("  Could not load data. Please check the ticker and try again.")
            if not ask_another():
                break
            continue

        # Step 4: Print metric summary
        print_stock_summary(ticker, data)

        # Step 5: Run bubble analysis and print report
        result = run_bubble_analysis(ticker, data)
        print_bubble_report(result)

        # Step 6: Generate and print avoidance strategies
        current_price = get_current_price(data)
        strategies    = generate_strategies(ticker, current_price, result['bubble_score'])
        print_strategies(ticker, result['bubble_score'], strategies)

        # Step 7: Plot the detail chart
        show_chart = input("\n  Show price and RSI chart? (yes / no): ").strip().lower()
        if show_chart in ('yes', 'y'):
            plot_stock_detail(ticker, data)

        # Step 8: Continue or exit
        if not ask_another():
            break

    print("\n  Analysis session ended. Goodbye.")
    print("=" * 55)


# Run the program when executed directly: python stock_analyzer.py
if __name__ == "__main__":
    main()
