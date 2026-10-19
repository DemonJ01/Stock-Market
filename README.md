# Stock Market Prediction using LSTM

A Python project that analyzes historical stock-market prices and uses a Long Short-Term Memory (LSTM) neural network to predict closing prices.

## Features
- Download historical stock data for a selected ticker.
- Compare price trends with a market index.
- Prepare time-series data using sliding windows and normalization.
- Train an LSTM model and evaluate it on later observations.
- Display predicted and actual closing prices.
- Estimate the next trading day's closing price.

## Technologies
Python, NumPy, pandas, Matplotlib, scikit-learn, TensorFlow/Keras, and yfinance.

## Installation
Use Python 3.11 or another compatible Python version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run
```bash
python Main.py --symbol AAPL --benchmark "^NSEI"
```

For a run without charts:
```bash
python Main.py --symbol AAPL --no-plot
```

Use `python Main.py --help` for additional options.

## Model
The script splits data chronologically into training and validation sets, fits the scaler using training observations, trains a two-layer LSTM, and calculates MAE, MAPE, and median percentage error on the validation period.

## Tests
```bash
python -m pytest -q
```

## Project Structure
```text
Main.py           Data processing, LSTM model, evaluation and plots
requirements.txt  Python dependencies
tests/            Unit tests
```

**Note:** This is an educational forecasting project; predictions are not financial advice.
