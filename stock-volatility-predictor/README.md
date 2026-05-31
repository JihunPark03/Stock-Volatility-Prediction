# NASDAQ Stock Volatility Risk Prediction

This beginner-friendly Python project predicts whether a NASDAQ stock is likely
to enter a high-volatility period over the next five trading days. It downloads
daily OHLCV market data from [Stooq](https://stooq.com/) through
`pandas-datareader`, builds historical features, and trains a LightGBM binary
classifier.

## Supported tickers

The CLI accepts one or more US tickers. For example:

```text
AAPL MSFT NVDA AMZN GOOGL
```

Ticker symbols are converted to Stooq's `.US` format automatically.

## Stooq API key

Stooq currently requires a captcha-issued API key for CSV downloads. Open a URL
such as [the AAPL download page](https://stooq.com/q/d/?s=AAPL.US&get_apikey),
complete the captcha, copy the `apikey` value from the generated CSV link, and
place it in a local `.env` file before training:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
STOOQ_API_KEY=your-key-from-stooq
```

The `.env` file is ignored by Git. Do not commit your API key.

The loader extends pandas-datareader's Stooq reader only to pass this required
query parameter.

## Features

Each model row uses information available at or before its date:

- Daily return
- 5-day and 20-day rolling volatility
- Close-price ratios to 5-day and 20-day moving averages
- Daily volume change
- Intraday high-low range
- Intraday open-close range

The outcome is the standard deviation of returns over the next five trading
days. The model target is `1` when that outcome meets or exceeds the training
period's 75th-percentile volatility threshold and `0` otherwise.

## Leakage prevention

- Stooq rows are sorted ascending by date before features are calculated.
- Rolling feature windows use only current and prior rows.
- Future volatility uses `.rolling(5).std().shift(-5)` and is never a feature.
- Data is split chronologically, never randomly.
- The top-25% threshold is learned from training-period outcomes only.
- Rolling calculations are isolated per ticker when multiple tickers are used.

## Setup

```bash
cd stock-volatility-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS with Homebrew, install OpenMP if LightGBM reports a missing
`libomp.dylib`:

```bash
brew install libomp
```

## Train and evaluate

Train the default AAPL example:

```bash
python -m src.train --tickers AAPL
```

Train a combined model on multiple NASDAQ tickers:

```bash
python -m src.train --tickers AAPL MSFT NVDA AMZN GOOGL
```

The command saves:

- `models/lightgbm_volatility.joblib`: model bundle for later predictions
- `reports/metrics.json`: evaluation metrics
- `reports/confusion_matrix.png`: confusion matrix chart
- `reports/shap_feature_importance.png`: SHAP importance chart
- `reports/shap_feature_importance.csv`: SHAP importance values

## Predict the latest available day

```bash
python -m src.predict --ticker AAPL
```

The output is a risk probability and a binary prediction. This example is for
education and experimentation, not investment advice.

## Project structure

```text
stock-volatility-predictor/
  README.md
  requirements.txt
  src/
    data_loader.py
    features.py
    train.py
    evaluate.py
    predict.py
    explain.py
  models/
  notebooks/
  reports/
```
