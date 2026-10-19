# Stock-Market

Educational LSTM market forecasting with chronological hold-out evaluation.

**Current scope:** Python, yfinance, NumPy, pandas, scikit-learn, TensorFlow/Keras and Matplotlib.

> Please review the [known limitations](docs/KNOWN_LIMITATIONS.md) before using or presenting this project. Documentation describes the implementation actually available in this repository; additional functionality is not implied.

## Project quick start

Clone the repository, change into its directory, then execute:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Main.py --symbol AAPL --benchmark '^NSEI' --no-plot
~~~

The examples target a Unix-like shell. On Windows, activate the virtual environment with the Windows script and configure environment variables using PowerShell syntax where applicable.

## Validation

~~~bash
python -m pytest -q
python -m compileall -q Main.py tests
~~~

Checks require the relevant Python/Node.js dependencies and, for some functionality, external services. Continuous integration may cover a smaller subset than full manual runtime validation. See the [technical architecture guide](docs/ARCHITECTURE.md) and [test strategy](docs/TESTING.md).

## What is implemented

`Main.py` downloads daily closes through yfinance, splits records chronologically, fits normalization on the training partition, forms rolling price windows, trains a two-layer LSTM, predicts held-out closes, computes error metrics, and provides a next-period estimate. `tests/` verifies deterministic data preparation. External services and model training are not covered by unit tests.

## Known limitations

Predictions are experimental, not investment advice. No walk-forward evaluation, reproducible random seeds, live-trading support, execution engine, or production monitoring. Yahoo Finance data can change or become temporarily inaccessible. The comparison chart shows raw prices, not comparable normalized performance.

For more information, see [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

## Project documentation

- [Architecture](docs/ARCHITECTURE.md) — components and data flow.
- [Testing](docs/TESTING.md) — checks, limits and manual verification.
- [Known limitations](docs/KNOWN_LIMITATIONS.md) — outstanding risks and missing features.
- [Contributing](CONTRIBUTING.md) — local development and pull request guidelines.
- [Security](SECURITY.md) — privacy and vulnerability reporting guidance.

## Contribution and license

Please contribute through a feature branch and a reviewable pull request. Never add secrets, private data or unverifiable feature claims to commits. A repository-wide license has not been selected in this improvement pass; ownership and third-party licensing must be reviewed before publishing one.

