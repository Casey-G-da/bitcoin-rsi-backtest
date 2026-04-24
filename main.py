"""Bitcoin 4-hour RSI backtest.

Run this file from the project folder:

    python main.py

The script downloads BTC-USD data, calculates RSI, runs a long-only
backtest, and saves CSV files, a text report, and charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


SYMBOL = "BTC-USD"
START_DATE = "2017-01-01"
INITIAL_BALANCE = 100_000.00
RISK_PER_TRADE = 0.02
RSI_PERIOD = 14
BUY_RSI_LEVEL = 30
SELL_RSI_LEVEL = 72
TAX_RATE_ON_PROFITS = 0.02

PROJECT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = PROJECT_DIR / "charts"


@dataclass
class OpenTrade:
    """Stores information about the currently open trade."""

    entry_date: pd.Timestamp
    entry_price: float
    position_size: float
    equity_at_entry: float


def fetch_price_data() -> pd.DataFrame:
    """Download 4-hour BTC-USD data from Yahoo Finance using yfinance."""

    end_date = (date.today() + timedelta(days=1)).isoformat()
    print(f"Downloading {SYMBOL} 4-hour data from {START_DATE} to {date.today().isoformat()}...")

    data = download_with_yfinance(START_DATE, end_date, "4h")
    data_note = "Direct 4-hour candles downloaded from yfinance."

    if data.empty:
        print(
            "Yahoo Finance did not return the full 4-hour history. "
            "Falling back to daily BTC-USD data expanded to 4-hour rows for older history."
        )
        data = fetch_daily_data_expanded_to_four_hours(end_date)
        data_note = (
            "Yahoo Finance did not provide 4-hour BTC-USD data back to 2017, "
            "so older daily yfinance candles were expanded into approximate 4-hour rows, "
            "then recent available 4-hour yfinance candles were appended."
        )

    if data.empty:
        raise RuntimeError("No data was downloaded from yfinance.")

    data = clean_price_data(data)
    data.attrs["data_note"] = data_note
    return data


def download_with_yfinance(start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Small wrapper around yfinance.download."""

    return yf.download(
        SYMBOL,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )


def clean_price_data(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output and keep the columns this project needs."""

    # yfinance can return MultiIndex columns. Flatten them for beginner-friendly code.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [column[0] for column in data.columns]

    if "Date" not in data.columns:
        data = data.rename_axis("Date").reset_index()

    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    data = data.dropna(subset=["Close"])
    data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.tz_convert(None)

    return data


def fetch_daily_data_expanded_to_four_hours(end_date: str) -> pd.DataFrame:
    """Fetch daily data and expand each day into six 4-hour rows.

    Yahoo Finance currently limits direct 4-hour BTC-USD downloads to recent
    history. This fallback keeps the project runnable from 2017 to today while
    still using yfinance as the data source. Older generated rows use the daily
    OHLC values, so those candles are approximate 4-hour bars, not true
    intraday candles. Recent rows use true yfinance 4-hour candles when Yahoo
    provides them.
    """

    daily = download_with_yfinance(START_DATE, end_date, "1d")

    if daily.empty:
        return daily

    daily = clean_price_data(daily)
    expanded_rows = []

    for row in daily.itertuples(index=False):
        day_start = pd.Timestamp(row.Date).normalize()
        four_hour_times = pd.date_range(day_start, periods=6, freq="4h")

        for candle_time in four_hour_times:
            expanded_rows.append(
                {
                    "Date": candle_time,
                    "Open": row.Open,
                    "High": row.High,
                    "Low": row.Low,
                    "Close": row.Close,
                    "Volume": row.Volume / 6,
                }
            )

    expanded = pd.DataFrame(expanded_rows)

    # Yahoo usually allows recent 4-hour candles. Use them where available and
    # keep expanded daily rows only for the older unavailable history.
    recent_start = (date.today() - timedelta(days=729)).isoformat()
    recent_four_hour = download_with_yfinance(recent_start, end_date, "4h")

    if recent_four_hour.empty:
        return expanded

    recent_four_hour = clean_price_data(recent_four_hour)
    expanded_before_recent = expanded[expanded["Date"] < recent_four_hour["Date"].min()].copy()

    return pd.concat([expanded_before_recent, recent_four_hour], ignore_index=True).sort_values("Date")


def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI using Wilder's smoothing method."""

    price_change = prices.diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add RSI and signal columns to the price data."""

    data = data.copy()
    data["RSI"] = calculate_rsi(data["Close"])

    previous_rsi = data["RSI"].shift(1)
    data["Buy_Signal"] = (previous_rsi >= BUY_RSI_LEVEL) & (data["RSI"] < BUY_RSI_LEVEL)
    data["Sell_Signal"] = (previous_rsi <= SELL_RSI_LEVEL) & (data["RSI"] > SELL_RSI_LEVEL)

    return data


def calculate_position_size(account_equity: float, entry_price: float) -> float:
    """Calculate BTC position size using 2% account risk.

    The strategy does not have a stop loss, so this project uses the 30% RSI
    oversold threshold as a simple sizing reference. That means the trade size
    is based on the dollars risked divided by 30% of the BTC entry price.
    """

    dollars_at_risk = account_equity * RISK_PER_TRADE
    risk_per_btc = entry_price * (BUY_RSI_LEVEL / 100)
    return dollars_at_risk / risk_per_btc


def run_backtest(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the RSI strategy and return trades plus an equity curve."""

    trades = []
    equity_rows = []
    account_equity = INITIAL_BALANCE
    after_tax_equity = INITIAL_BALANCE
    cumulative_tax = 0.0
    open_trade: OpenTrade | None = None

    for row in data.itertuples(index=False):
        current_date = row.Date
        current_price = float(row.Close)

        if open_trade is not None:
            unrealized_pl = (current_price - open_trade.entry_price) * open_trade.position_size
            mark_to_market_equity = account_equity + unrealized_pl
            mark_to_market_after_tax = after_tax_equity + unrealized_pl
        else:
            mark_to_market_equity = account_equity
            mark_to_market_after_tax = after_tax_equity

        equity_rows.append(
            {
                "date": current_date,
                "close": current_price,
                "rsi": row.RSI,
                "equity": mark_to_market_equity,
                "after_tax_equity": mark_to_market_after_tax,
                "drawdown": 0.0,
            }
        )

        # Enter only one long trade at a time.
        if open_trade is None and bool(row.Buy_Signal):
            position_size = calculate_position_size(account_equity, current_price)
            open_trade = OpenTrade(
                entry_date=current_date,
                entry_price=current_price,
                position_size=position_size,
                equity_at_entry=account_equity,
            )
            continue

        # Exit the open long trade when RSI crosses above the sell level.
        if open_trade is not None and bool(row.Sell_Signal):
            profit_loss = (current_price - open_trade.entry_price) * open_trade.position_size
            account_equity += profit_loss

            tax_estimate = max(profit_loss, 0) * TAX_RATE_ON_PROFITS
            cumulative_tax += tax_estimate
            after_tax_equity += profit_loss - tax_estimate

            percent_return = profit_loss / open_trade.equity_at_entry
            cumulative_return = (account_equity / INITIAL_BALANCE) - 1

            trades.append(
                {
                    "entry_date": open_trade.entry_date,
                    "entry_price": open_trade.entry_price,
                    "exit_date": current_date,
                    "exit_price": current_price,
                    "position_size": open_trade.position_size,
                    "profit_loss": profit_loss,
                    "account_equity": account_equity,
                    "percent_return": percent_return,
                    "cumulative_return": cumulative_return,
                    "tax_estimate": tax_estimate,
                    "after_tax_equity": after_tax_equity,
                }
            )

            open_trade = None

            # Update the just-added equity row so the exit candle reflects realized equity.
            equity_rows[-1]["equity"] = account_equity
            equity_rows[-1]["after_tax_equity"] = after_tax_equity

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows)

    if not equity_curve.empty:
        equity_curve["running_peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = equity_curve["equity"] / equity_curve["running_peak"] - 1
        equity_curve["cumulative_return"] = equity_curve["equity"] / INITIAL_BALANCE - 1
        equity_curve["after_tax_cumulative_return"] = equity_curve["after_tax_equity"] / INITIAL_BALANCE - 1
        equity_curve["cumulative_tax_paid"] = cumulative_tax

    return trades_df, equity_curve


def save_outputs(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> None:
    """Save trades and equity curve CSV files."""

    trades.to_csv(PROJECT_DIR / "trades.csv", index=False)
    equity_curve.to_csv(PROJECT_DIR / "equity_curve.csv", index=False)


def create_summary_report(data: pd.DataFrame, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> None:
    """Create a plain-English text summary of the backtest."""

    if equity_curve.empty:
        raise RuntimeError("Equity curve is empty. Cannot create report.")

    total_trades = len(trades)
    winning_trades = int((trades["profit_loss"] > 0).sum()) if total_trades else 0
    losing_trades = int((trades["profit_loss"] <= 0).sum()) if total_trades else 0
    win_rate = winning_trades / total_trades if total_trades else 0
    total_profit_loss = trades["profit_loss"].sum() if total_trades else 0
    total_tax = trades["tax_estimate"].sum() if total_trades else 0
    final_equity = float(equity_curve["equity"].iloc[-1])
    final_after_tax_equity = float(equity_curve["after_tax_equity"].iloc[-1])
    max_drawdown = float(equity_curve["drawdown"].min())
    best_trade = trades["profit_loss"].max() if total_trades else 0
    worst_trade = trades["profit_loss"].min() if total_trades else 0

    report = f"""Bitcoin 4-Hour RSI Backtest Summary
====================================

Symbol: {SYMBOL}
Requested date range: {START_DATE} to {date.today().isoformat()}
Downloaded first candle: {data["Date"].min()}
Downloaded last candle: {data["Date"].max()}
Data note: {data.attrs.get("data_note", "No data note available.")}
RSI period: {RSI_PERIOD}
Buy signal: RSI crosses below {BUY_RSI_LEVEL}
Sell signal: RSI crosses above {SELL_RSI_LEVEL}
Starting account balance: ${INITIAL_BALANCE:,.2f}
Risk per trade: {RISK_PER_TRADE:.0%}
Tax estimate: {TAX_RATE_ON_PROFITS:.0%} of profitable trade gains

Performance
-----------
Total trades: {total_trades}
Winning trades: {winning_trades}
Losing trades: {losing_trades}
Win rate: {win_rate:.2%}
Total profit/loss before tax: ${total_profit_loss:,.2f}
Estimated taxes: ${total_tax:,.2f}
Final equity before tax: ${final_equity:,.2f}
Final equity after tax: ${final_after_tax_equity:,.2f}
Total return before tax: {(final_equity / INITIAL_BALANCE - 1):.2%}
Total return after tax: {(final_after_tax_equity / INITIAL_BALANCE - 1):.2%}
Maximum drawdown: {max_drawdown:.2%}
Best trade: ${best_trade:,.2f}
Worst trade: ${worst_trade:,.2f}

Files Created
-------------
trades.csv
equity_curve.csv
summary_report.txt
charts/btc_price_signals.png
charts/rsi_chart.png
charts/equity_curve_before_tax.png
charts/equity_curve_after_tax.png
charts/drawdown.png
charts/pl_per_trade.png
charts/trade_returns_scatter.png
"""

    (PROJECT_DIR / "summary_report.txt").write_text(report, encoding="utf-8")


def save_chart(path: Path) -> None:
    """Apply tight layout, save the active chart, and close it."""

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def create_charts(data: pd.DataFrame, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> None:
    """Create all required matplotlib charts."""

    CHARTS_DIR.mkdir(exist_ok=True)

    buy_points = trades[["entry_date", "entry_price"]] if not trades.empty else pd.DataFrame()
    sell_points = trades[["exit_date", "exit_price"]] if not trades.empty else pd.DataFrame()

    plt.figure(figsize=(14, 7))
    plt.plot(data["Date"], data["Close"], label="BTC-USD Close", color="#1f77b4", linewidth=1)
    if not buy_points.empty:
        plt.scatter(buy_points["entry_date"], buy_points["entry_price"], label="Buy", color="green", marker="^", s=70)
        plt.scatter(sell_points["exit_date"], sell_points["exit_price"], label="Sell", color="red", marker="v", s=70)
    plt.title("BTC-USD Price with Buy and Sell Markers")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_chart(CHARTS_DIR / "btc_price_signals.png")

    plt.figure(figsize=(14, 5))
    plt.plot(data["Date"], data["RSI"], label="RSI 14", color="#6f42c1", linewidth=1)
    plt.axhline(BUY_RSI_LEVEL, color="green", linestyle="--", label="Buy Level 30")
    plt.axhline(SELL_RSI_LEVEL, color="red", linestyle="--", label="Sell Level 72")
    plt.title("RSI Chart")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_chart(CHARTS_DIR / "rsi_chart.png")

    plt.figure(figsize=(14, 6))
    plt.plot(equity_curve["date"], equity_curve["equity"], color="#2ca02c", linewidth=1.5)
    plt.title("Equity Curve Before Tax")
    plt.xlabel("Date")
    plt.ylabel("Account Equity")
    plt.grid(True, alpha=0.3)
    save_chart(CHARTS_DIR / "equity_curve_before_tax.png")

    plt.figure(figsize=(14, 6))
    plt.plot(equity_curve["date"], equity_curve["after_tax_equity"], color="#17becf", linewidth=1.5)
    plt.title("Equity Curve After Tax")
    plt.xlabel("Date")
    plt.ylabel("After-Tax Equity")
    plt.grid(True, alpha=0.3)
    save_chart(CHARTS_DIR / "equity_curve_after_tax.png")

    plt.figure(figsize=(14, 5))
    plt.fill_between(equity_curve["date"], equity_curve["drawdown"] * 100, color="#d62728", alpha=0.35)
    plt.plot(equity_curve["date"], equity_curve["drawdown"] * 100, color="#d62728", linewidth=1)
    plt.title("Drawdown Chart")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    save_chart(CHARTS_DIR / "drawdown.png")

    plt.figure(figsize=(14, 5))
    if not trades.empty:
        bar_colors = ["green" if value > 0 else "red" for value in trades["profit_loss"]]
        plt.bar(range(1, len(trades) + 1), trades["profit_loss"], color=bar_colors)
    plt.title("Profit/Loss per Trade")
    plt.xlabel("Trade Number")
    plt.ylabel("Profit/Loss")
    plt.grid(True, axis="y", alpha=0.3)
    save_chart(CHARTS_DIR / "pl_per_trade.png")

    plt.figure(figsize=(10, 6))
    if not trades.empty:
        scatter_colors = ["green" if value > 0 else "red" for value in trades["percent_return"]]
        plt.scatter(range(1, len(trades) + 1), trades["percent_return"] * 100, c=scatter_colors, s=60)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Scatter Plot of Trade Returns")
    plt.xlabel("Trade Number")
    plt.ylabel("Return (%)")
    plt.grid(True, alpha=0.3)
    save_chart(CHARTS_DIR / "trade_returns_scatter.png")


def main() -> None:
    """Run the full project from start to finish."""

    CHARTS_DIR.mkdir(exist_ok=True)

    data = fetch_price_data()
    data = add_indicators(data)
    trades, equity_curve = run_backtest(data)

    save_outputs(trades, equity_curve)
    create_summary_report(data, trades, equity_curve)
    create_charts(data, trades, equity_curve)

    print("Backtest complete.")
    print(f"Trades saved to: {PROJECT_DIR / 'trades.csv'}")
    print(f"Equity curve saved to: {PROJECT_DIR / 'equity_curve.csv'}")
    print(f"Summary report saved to: {PROJECT_DIR / 'summary_report.txt'}")
    print(f"Charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
