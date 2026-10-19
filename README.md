# Stock Market Trend Analysis with LSTM

A Python educational project that downloads historical market prices, trains a Long Short-Term Memory (LSTM) network and compares out-of-sample predicted closing prices with actual closing prices.

> **Disclaimer:** Predictions are experimental. This is not financial advice, a trading strategy, or a validated forecasting service. Market prices are volatile and models can fail.

## Features

- Download daily historical prices from Yahoo Finance using [yfinance](https://github.com/ranaroussi/yfinance).
- Select the ticker, comparison index, historical start date, sequence length and training epochs through command-line arguments.
- Train a two-layer LSTM on the first 80% of available observations.
- Fit MinMaxScaler on *training data only* to avoid leaking future observations into training.
- Evaluate held-out observations using MAE, MAPE and median absolute percentage error.
- Display optional charts for market history and the prediction/actual comparison.
- Produce one experimental next-trading-day closing-price forecast.

## Requirements

- Python 3.11 recommended.
- Internet access for Yahoo Finance downloads.
- TensorFlow and a CPU/GPU capable of running Keras training.
- A graphical desktop for plotting, unless the no-plot option is used.

## Installation

~~~bash
git clone https://github.com/DemonJ01/Stock-Market.git
cd Stock-Market
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
~~~

## Usage

~~~bash
python Main.py --symbol AAPL --benchmark "^NSEI" --start 2023-01-10 --epochs 10
~~~

For a headless run:

~~~bash
python Main.py --symbol "^GSPC" --no-plot
~~~

See all command-line options with: python Main.py --help

Yahoo Finance ticker formats may differ from company names. For example, AAPL is Apple, ^GSPC is the S&P 500 index, and ^NSEI is Nifty 50. The comparison chart displays raw values, not normalized investment returns; do not directly compare absolute levels as performance.

## How the model works

1. Download the selected ticker's closing-price history.
2. Reserve the last 20% for chronological validation, not random shuffling.
3. Fit normalization on training observations only.
4. Create rolling windows of 50 previous prices by default.
5. Fit the LSTM and predict the reserved observations.
6. Invert normalization to compute metrics in price units.
7. Use the latest 50 available prices to produce one experimental forecast.

The model is a simple univariate baseline. It does not incorporate sentiment, fundamentals, transaction costs, or market regimes. Forecasting accuracy is **not guaranteed**. The model is not a walk-forward backtest and it is not production-trading ready.

## Testing

~~~bash
python -m pytest -q
python -m compileall -q Main.py tests
~~~

Tests cover deterministic data preparation and metric utilities. End-to-end training requires historical downloads and a TensorFlow runtime.

## Repository layout

~~~text
Main.py             Command-line application, training and plotting
requirements.txt    Python dependencies
tests/              Unit tests (do not require model training)
.github/workflows/  Automated syntax/unit-test checks
~~~

## Troubleshooting

- **No prices returned:** verify the ticker, network, and Yahoo Finance service.
- **TensorFlow installation error:** use a supported Python version and platform.
- **Too few data points:** increase the historical date range or reduce the sequence window.
- **Plotting unavailable:** pass --no-plot.

## Responsible use

Do not use model forecasts as the sole basis for real-money trading. Verify data licenses and service terms for your intended use.
