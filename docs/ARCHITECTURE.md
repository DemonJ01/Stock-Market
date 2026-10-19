# Architecture — Stock-Market

`Main.py` downloads daily closes through yfinance, splits records chronologically, fits normalization on the training partition, forms rolling price windows, trains a two-layer LSTM, predicts held-out closes, computes error metrics, and provides a next-period estimate. `tests/` verifies deterministic data preparation. External services and model training are not covered by unit tests.

## Technology

Python, yfinance, NumPy, pandas, scikit-learn, TensorFlow/Keras and Matplotlib.

## Trust boundaries and dependencies

Predictions are experimental, not investment advice. No walk-forward evaluation, reproducible random seeds, live-trading support, execution engine, or production monitoring. Yahoo Finance data can change or become temporarily inaccessible. The comparison chart shows raw prices, not comparable normalized performance.

Changes to external services, persistence, authentication, model inference or exchange integrations should include tests for failure cases, error handling and authorization rather than relying only on compilation.
