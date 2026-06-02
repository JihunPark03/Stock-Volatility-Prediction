# NASDAQ Stock Volatility Prediction

A machine learning project for monitoring NASDAQ stock volatility risk. The
main application predicts whether a stock is likely to enter a high-volatility
period over the next five trading days and displays the result in a web
dashboard.

## Website Screenshots

### Dashboard Overview

![Dashboard overview](assets/screenshots/Screenshot%202026-06-02%20at%2010.49.29.png)

### Prediction Results

![Prediction results](assets/screenshots/Screenshot%202026-06-02%20at%2010.50.13.png)

## Overview

The project uses daily OHLCV market data from [Stooq](https://stooq.com/) and a
LightGBM classifier. Users can enter multiple NASDAQ ticker symbols, such as
`AAPL`, `MSFT`, or `NVDA`, and receive an individual risk probability for each
stock.

The model is designed as an educational financial risk-monitoring project. It
is not investment advice.

## Main Features

- Fetches daily stock data from Stooq using `pandas-datareader`
- Supports multiple NASDAQ ticker symbols
- Builds leakage-resistant historical features from OHLCV data
- Predicts high-volatility risk over the next five trading days
- Evaluates the model with precision, recall, F1-score, ROC-AUC, and a
  confusion matrix
- Generates SHAP-based feature importance reports
- Saves the trained LightGBM model for later inference
- Provides a FastAPI backend and a Next.js dashboard

## Technology Stack

- Python, pandas, NumPy, scikit-learn
- LightGBM, SHAP, matplotlib
- FastAPI, Uvicorn
- Next.js, React, TypeScript

## Repository Structure

```text
Stock-Volatility-Prediction/
  README.md
  stock-volatility-predictor/
    src/                  # Data loading, features, training, API, prediction
    models/               # Saved LightGBM model
    reports/              # Metrics, confusion matrix, SHAP reports
    web/                  # Next.js dashboard
    README.md             # Detailed application documentation
    requirements.txt
  basic_of_light_gbm/
    main.py               # Small LightGBM regression learning example
    data/aapl.us.txt      # Local AAPL sample data
```

`basic_of_light_gbm` is a simple learning prototype. The production-style
classifier, API, and dashboard are in `stock-volatility-predictor`.

## Model Features

The classifier uses information available at or before each trading date:

- Daily return
- 5-day and 20-day rolling volatility
- 5-day and 20-day moving-average ratios
- Daily volume change
- Intraday high-low range
- Intraday open-close range

The target is `1` when the future five-day volatility is within the top 25% of
the training-period distribution and `0` otherwise. Training and testing use a
chronological split to reduce data leakage.

## Setup

```bash
cd stock-volatility-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, install OpenMP if LightGBM reports a missing `libomp.dylib`:

```bash
brew install libomp
```

## Stooq API Key

Create the local environment file:

```bash
cd stock-volatility-predictor
cp .env.example .env
```

Open the [Stooq API key page](https://stooq.com/q/d/?s=AAPL.US&get_apikey),
complete the captcha, and place the generated key in `.env`:

```dotenv
STOOQ_API_KEY=your-key-from-stooq
```

## Train the Model

```bash
cd stock-volatility-predictor
python -m src.train --tickers AAPL MSFT NVDA AMZN GOOGL
```

## Run a CLI Prediction

```bash
cd stock-volatility-predictor
python -m src.predict --ticker AAPL
```

## Run the Website

Start the FastAPI backend:

```bash
cd stock-volatility-predictor
source .venv/bin/activate
uvicorn src.api:app --reload --port 8000
```

In a second terminal, start the Next.js dashboard:

```bash
cd stock-volatility-predictor/web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Detailed Documentation

See [stock-volatility-predictor/README.md](stock-volatility-predictor/README.md)
for implementation details, leakage-prevention notes, output artifacts, and
additional commands.
