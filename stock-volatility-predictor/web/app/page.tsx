"use client";

import { FormEvent, useMemo, useState } from "react";

type Prediction = {
  ticker: string;
  date: string;
  high_volatility_probability: number;
  predicted_high_volatility: number;
  risk_level: string;
};

type PredictionError = {
  ticker: string;
  message: string;
};

type ApiResponse = {
  predictions: Prediction[];
  errors: PredictionError[];
};

const quickTickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"];

function parseTickers(value: string) {
  return Array.from(
    new Set(
      value
        .split(/[\s,]+/)
        .map((ticker) => ticker.trim().toUpperCase())
        .filter(Boolean),
    ),
  );
}

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function RiskCard({ prediction }: { prediction: Prediction }) {
  const probability = prediction.high_volatility_probability;
  const highRisk = prediction.predicted_high_volatility === 1;
  const circumference = 2 * Math.PI * 45;
  const offset = circumference * (1 - probability);

  return (
    <article className={`risk-card ${highRisk ? "risk-high" : "risk-normal"}`}>
      <div className="card-topline">
        <div>
          <p className="card-symbol">{prediction.ticker}</p>
          <p className="card-date">Market data: {prediction.date}</p>
        </div>
        <span className="risk-pill">
          <span className="status-dot" />
          {prediction.risk_level}
        </span>
      </div>

      <div className="risk-content">
        <div className="gauge">
          <svg viewBox="0 0 112 112" aria-label={`${percentage(probability)} risk`}>
            <circle className="gauge-track" cx="56" cy="56" r="45" />
            <circle
              className="gauge-fill"
              cx="56"
              cy="56"
              r="45"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
            />
          </svg>
          <div className="gauge-value">
            <strong>{percentage(probability)}</strong>
            <span>risk score</span>
          </div>
        </div>

        <div className="risk-details">
          <p className="detail-label">5-day outlook</p>
          <p className="detail-copy">
            {highRisk
              ? "Elevated volatility conditions are likely. Review exposure and risk limits."
              : "Volatility conditions remain within the model's normal-risk range."}
          </p>
          <div className="threshold-row">
            <span>Classification threshold</span>
            <strong>50.0%</strong>
          </div>
          <div className="threshold-track">
            <span style={{ width: `${probability * 100}%` }} />
            <i />
          </div>
        </div>
      </div>
    </article>
  );
}

export default function Home() {
  const [input, setInput] = useState("AAPL, MSFT, NVDA");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [errors, setErrors] = useState<PredictionError[]>([]);
  const [loading, setLoading] = useState(false);
  const tickers = useMemo(() => parseTickers(input), [input]);

  function addTicker(ticker: string) {
    setInput((current) => {
      const currentTickers = parseTickers(current);
      return currentTickers.includes(ticker)
        ? currentTickers.filter((item) => item !== ticker).join(", ")
        : [...currentTickers, ticker].join(", ");
    });
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!tickers.length) return;

    setLoading(true);
    setErrors([]);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers }),
      });
      const data = (await response.json()) as ApiResponse;
      setPredictions(data.predictions ?? []);
      setErrors(data.errors ?? []);
    } catch {
      setPredictions([]);
      setErrors([{ ticker: "API", message: "Unable to reach the prediction service." }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <nav className="nav">
        <div className="brand">
          <span className="brand-mark">V</span>
          <span>VOLATILITY WATCH</span>
        </div>
        <div className="nav-meta">
          <span className="live-dot" />
          MODEL ONLINE
          <span className="nav-rule" />
          NASDAQ
        </div>
      </nav>

      <section className="hero">
        <p className="eyebrow">LIGHTGBM RISK INTELLIGENCE / 5-DAY OUTLOOK</p>
        <h1>
          See volatility
          <br />
          <em>before it moves.</em>
        </h1>
        <p className="hero-copy">
          Enter NASDAQ symbols to estimate the probability of a high-volatility
          period over the next five trading days.
        </p>
      </section>

      <section className="terminal">
        <form onSubmit={submit}>
          <label htmlFor="tickers">
            <span>01</span> ENTER STOCK SYMBOLS
          </label>
          <div className="input-row">
            <input
              id="tickers"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="AAPL, MSFT, NVDA"
              autoComplete="off"
            />
            <button type="submit" disabled={loading || !tickers.length}>
              {loading ? "ANALYZING..." : "RUN ANALYSIS"}
              <span>→</span>
            </button>
          </div>
          <div className="quick-row">
            <p>QUICK SELECT</p>
            {quickTickers.map((ticker) => (
              <button
                className={tickers.includes(ticker) ? "selected" : ""}
                key={ticker}
                onClick={() => addTicker(ticker)}
                type="button"
              >
                {ticker}
              </button>
            ))}
          </div>
        </form>
      </section>

      <section className="model-strip">
        <div>
          <span>MODEL</span>
          <strong>LIGHTGBM CLASSIFIER</strong>
        </div>
        <div>
          <span>FORECAST WINDOW</span>
          <strong>5 TRADING DAYS</strong>
        </div>
        <div>
          <span>FEATURE SIGNALS</span>
          <strong>8 OHLCV INDICATORS</strong>
        </div>
      </section>

      {(predictions.length > 0 || errors.length > 0) && (
        <section className="results">
          <div className="section-heading">
            <div>
              <p className="eyebrow">LATEST MODEL OUTPUT</p>
              <h2>Risk predictions</h2>
            </div>
            <p>{predictions.length} SYMBOLS ANALYZED</p>
          </div>

          {errors.length > 0 && (
            <div className="error-panel">
              {errors.map((error) => (
                <p key={`${error.ticker}-${error.message}`}>
                  <strong>{error.ticker}</strong> {error.message}
                </p>
              ))}
            </div>
          )}

          <div className="card-grid">
            {predictions.map((prediction) => (
              <RiskCard key={prediction.ticker} prediction={prediction} />
            ))}
          </div>
        </section>
      )}

      {!predictions.length && !errors.length && (
        <section className="empty-state">
          <div className="empty-index">02</div>
          <div>
            <p className="eyebrow">AWAITING SYMBOLS</p>
            <h2>Your forecast panel is ready.</h2>
            <p>Run an analysis to populate the latest model predictions.</p>
          </div>
        </section>
      )}

      <footer>
        <span>VOLATILITY WATCH / NASDAQ RISK MONITOR</span>
        <span>FOR EDUCATIONAL USE ONLY · NOT INVESTMENT ADVICE</span>
      </footer>
    </main>
  );
}
