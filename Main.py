"""Download market data, evaluate an LSTM, and forecast one closing price.

Research/educational use only; not investment advice.
"""
from __future__ import annotations

import argparse
from datetime import date
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def close_prices(symbol: str, start: str) -> pd.Series:
    """Return a consistently shaped closing-price Series for one ticker."""
    result = yf.download(symbol, start=start, end=date.today().isoformat(),
                         auto_adjust=True, progress=False)
    if result.empty or "Close" not in result:
        raise ValueError(f"No historical closing prices returned for {symbol!r}.")
    close = result["Close"]
    # Recent yfinance releases can return a one-column DataFrame with a
    # MultiIndex, even when downloading a single ticker.
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError(f"Expected one closing-price column for {symbol!r}.")
        close = close.iloc[:, 0]
    clean = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if clean.empty:
        raise ValueError(f"No valid closing prices for {symbol!r}.")
    return clean.astype(float)


def make_sequences(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (samples, timesteps, features) inputs and next-day targets."""
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if window < 1:
        raise ValueError("Window must be at least 1.")
    if len(values) <= window:
        return np.empty((0, window, 1)), np.empty((0,))
    x = np.array([values[i - window:i] for i in range(window, len(values))])
    y = values[window:, 0]
    return x, y


def safe_mape(actual: Sequence[float], prediction: Sequence[float]) -> float:
    """MAPE percentage excluding zero actual prices."""
    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    predicted_array = np.asarray(prediction, dtype=float).reshape(-1)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted series must have equal lengths.")
    valid = actual_array != 0
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(
        (actual_array[valid] - predicted_array[valid]) / actual_array[valid]
    )) * 100)


def analyze(symbol: str, benchmark: str, start: str, window: int,
            epochs: int, show_plots: bool = True) -> None:
    # Import TensorFlow only when training: simple utility tests can run without it.
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input, LSTM

    prices = close_prices(symbol, start)
    if window < 1:
        raise ValueError('Window must be at least 1')
    if len(prices) < window + 12:
        raise ValueError(
            f"Need at least {window + 12} trading days for a train/test split; "
            f"received {len(prices)}."
        )
    split = int(len(prices) * 0.8)
    if split <= window or split >= len(prices):
        raise ValueError("Not enough data for the selected sequence window.")

    # Prevent data leakage: learn scaling parameters using TRAINING prices only.
    scaler = MinMaxScaler()
    scaler.fit(prices.iloc[:split].to_numpy().reshape(-1, 1))
    scaled = scaler.transform(prices.to_numpy().reshape(-1, 1))
    x_train, y_train = make_sequences(scaled[:split], window)
    x_test, y_test = make_sequences(scaled[split - window:], window)
    if not len(x_train) or not len(x_test):
        raise ValueError("No training or validation sequences were generated.")

    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(8, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=epochs, batch_size=16, verbose=1)

    predictions = scaler.inverse_transform(
        model.predict(x_test, verbose=0)
    ).reshape(-1)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    validation = pd.DataFrame(
        {"Actual": actual, "Predicted": predictions},
        index=prices.index[split:],
    )
    validation["Difference"] = validation["Predicted"] - validation["Actual"]
    print(f"\nTicker: {symbol}; validation samples: {len(validation)}")
    print(f"MAE: {mean_absolute_error(actual, predictions):.3f}")
    print(f"MAPE: {safe_mape(actual, predictions):.2f}%")
    valid_nonzero = actual != 0
    median_ape = float(np.median(np.abs((actual[valid_nonzero] - predictions[valid_nonzero]) / actual[valid_nonzero])) * 100) if valid_nonzero.any() else float('nan')
    print(f"Median APE: {median_ape:.2f}%")

    next_input = scaled[-window:].reshape(1, window, 1)
    next_price = float(scaler.inverse_transform(
        model.predict(next_input, verbose=0)
    )[0, 0])
    last_price = float(prices.iloc[-1])
    change_percent = (next_price / last_price - 1) * 100
    print(f"Last observed close: {last_price:.2f}")
    print(f"Forecast next trading-day close: {next_price:.2f} ({change_percent:+.2f}%)")

    if show_plots:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
        axes[0].plot(prices.index, prices, label=symbol)
        if benchmark:
            try:
                reference = close_prices(benchmark, start)
                axes[0].plot(reference.index, reference, label=benchmark)
            except ValueError as exc:
                print(f"Benchmark unavailable: {exc}")
        axes[0].set_title("Historical adjusted closing prices (raw values; not normalized returns)")
        axes[0].legend()
        axes[1].plot(validation.index, validation["Actual"], label="Actual")
        axes[1].plot(validation.index, validation["Predicted"], label="LSTM prediction")
        axes[1].set_title("Out-of-sample validation")
        axes[1].legend()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="^GSPC", help="Yahoo Finance ticker")
    parser.add_argument("--benchmark", default="^NSEI", help="Comparison ticker")
    parser.add_argument("--start", default="2023-01-10", help="YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=50, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive plots")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    try:
        analyze(args.symbol, args.benchmark, args.start,
                args.window, args.epochs, not args.no_plot)
    except (ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
