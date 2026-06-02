"""Load daily OHLCV data from Stooq."""

from __future__ import annotations

from datetime import date
import os

import pandas as pd
from pandas.errors import ParserError
from pandas_datareader.stooq import StooqDailyReader
from dotenv import load_dotenv

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
load_dotenv()


class ApiKeyStooqDailyReader(StooqDailyReader):
    """Extend pandas-datareader's Stooq reader for Stooq's current API key."""

    def __init__(self, *args, api_key: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = api_key

    def _get_params(self, symbol, country="US"):
        params = super()._get_params(symbol, country)
        if self.api_key:
            params["apikey"] = self.api_key
        return params


def to_stooq_symbol(ticker: str) -> str:
    """Convert a NASDAQ ticker such as AAPL to Stooq's US symbol format."""
    ticker = ticker.strip().upper()
    return ticker if ticker.endswith(".US") else f"{ticker}.US"


def fetch_stock_data(
    ticker: str,
    start: str = "2015-01-01",
    end: str | date | None = None,
) -> pd.DataFrame:
    """Fetch one ticker from Stooq and return rows sorted oldest to newest."""
    symbol = to_stooq_symbol(ticker)
    api_key = os.getenv("STOOQ_API_KEY")
    reader = ApiKeyStooqDailyReader(
        symbols=symbol,
        start=start,
        end=end,
        api_key=api_key,
    )
    try:
        frame = reader.read()
    except ParserError as exc:
        if not api_key:
            raise RuntimeError(
                "Stooq CSV downloads require an API key. Visit "
                f"https://stooq.com/q/d/?s={symbol}&get_apikey, complete the "
                "captcha, and place STOOQ_API_KEY from the CSV download URL "
                "in the project's .env file."
            ) from exc
        raise
    if frame.empty:
        raise ValueError(f"No Stooq data returned for {symbol}.")

    missing = sorted(set(OHLCV_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Stooq response for {symbol} is missing columns: {missing}")

    frame = frame.loc[:, OHLCV_COLUMNS].copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame["Ticker"] = ticker.removesuffix(".US").upper()
    return frame


def fetch_multiple_stocks(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | date | None = None,
) -> pd.DataFrame:
    """Fetch and combine multiple tickers while retaining each ticker label."""
    frames = [fetch_stock_data(ticker, start=start, end=end) for ticker in tickers]
    return pd.concat(frames).sort_index()
