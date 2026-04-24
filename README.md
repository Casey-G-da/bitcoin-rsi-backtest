# Bitcoin 4-Hour RSI Backtest

This project backtests a simple long-only Bitcoin RSI strategy using 4-hour BTC-USD data from Yahoo Finance.

## Strategy

- Asset: `BTC-USD`
- Timeframe: 4-hour candles
- Start date: `2017-01-01`
- End date: today, based on the date when the script runs
- RSI period: 14
- Starting account balance: `$100,000`
- Risk per trade: `2%` of current account equity
- Buy signal: RSI crosses below `30`
- Sell signal: RSI crosses above `72`
- Direction: long-only

The backtest enters a long trade when RSI moves from at least 30 to below 30. It exits the open long trade when RSI moves from at most 72 to above 72.

Position size is calculated so that 2% of account equity is allocated as risk. Because this RSI strategy does not define a stop loss, the code uses the distance from entry price to the RSI oversold level as a practical sizing reference:

```text
risk dollars = account equity * 0.02
risk per bitcoin = entry price * 0.30
position size = risk dollars / risk per bitcoin
```

This keeps sizing beginner-friendly while still making each trade scale with account equity.

## Outputs

Running `main.py` creates:

- `trades.csv`
- `equity_curve.csv`
- `summary_report.txt`
- `charts/btc_price_signals.png`
- `charts/rsi_chart.png`
- `charts/equity_curve_before_tax.png`
- `charts/equity_curve_after_tax.png`
- `charts/drawdown.png`
- `charts/pl_per_trade.png`
- `charts/trade_returns_scatter.png`

## Setup

```bash
cd bitcoin-rsi-backtest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The script downloads BTC-USD data, runs the strategy, saves CSV/report outputs, and writes all charts into the `charts` folder.

## Notes

Yahoo Finance intraday history is limited by Yahoo's data availability rules. The script requests 4-hour data directly with yfinance first. If Yahoo refuses the full 2017-to-present 4-hour range, the script falls back to daily BTC-USD candles from yfinance, expands older daily candles into six approximate 4-hour rows, and appends recent true 4-hour candles where Yahoo allows them. The summary report includes a data note so this limitation is visible in the output.
