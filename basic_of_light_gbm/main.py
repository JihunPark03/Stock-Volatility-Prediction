import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.lower()
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )

    if "vol" in df.columns:
        df = df.rename(columns={"vol": "volume"})

    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.sort_values("date")

    return df

def make_features(df):
    """
    먼저 데이터에서 의미 있는 데이터로 변환
    데이터의 형태:
    date,open,high,low,close,volume
    2024-01-01,100,105,99,103,1000000
    2024-01-02,103,107,101,106,1200000
    """
    df = df.copy()

    df["return"] = df["close"].pct_change() # 이전 값 대비 얼마나 변했는지
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["open_close_return"] = (df["close"] - df["open"]) / df["open"]
    df["volume_change"] = df["volume"].pct_change()

    df["ma_5"] = df["close"].rolling(5).mean() # 가장 최근의 5개의 데이터로 계산
    df["ma_20"] = df["close"].rolling(20).mean()

    df["volatility_5"] = df["return"].rolling(5).std()
    df["volatility_20"] = df["return"].rolling(20).std()

    df["target"] = df["return"].rolling(5).std().shift(-5)
    """"
    volatility_5 = [NaN, NaN, NaN, NaN, 0.018, 0.021, 0.025, ...] 
    여기서 target = [0.021, 0.025, ..., NaN, NaN, NaN, NaN]
    이런식으로 이동시킨다
    현재 데이터로 미래를 예측하도록 학습 데이터 정렬
    """

    return df

def train_lightgbm(df):
    df = make_features(df)
    df = df.dropna()

    features = [
        "return",
        "high_low_range",
        "open_close_return",
        "volume_change",
        "ma_5",
        "ma_20",
        "volatility_5",
        "volatility_20",
    ]

    X = df[features]
    y = df["target"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    print("RMSE:", rmse)
    print("MAE:", mae)

    return model, y_test, predictions, features

def plot_actual_vs_predicted(y_test, predictions):
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label="Actual Volatility")
    plt.plot(predictions, label="Predicted Volatility")
    plt.title("Actual vs Predicted Volatility")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, features):
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(importance["feature"], importance["importance"])
    plt.title("LightGBM Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def predict_next_5_days(model, df, features):
    feature_df = make_features(df)
    feature_df = feature_df.dropna()

    latest_row = feature_df.iloc[-1]
    latest_features = feature_df[features].iloc[[-1]]

    predicted_volatility = model.predict(latest_features)[0]

    base_date = latest_row["date"]

    future_dates = pd.bdate_range(
        start=base_date + pd.Timedelta(days=1),
        periods=5
    )

    result = pd.DataFrame({
        "base_date": [base_date],
        "prediction_start_date": [future_dates[0]],
        "prediction_end_date": [future_dates[-1]],
        "predicted_5day_volatility": [predicted_volatility],
        "predicted_5day_volatility_percent": [predicted_volatility * 100],
    })

    print(result)

    return result

def plot_next_5_days_prediction(result):
    row = result.iloc[0]

    start_date = row["prediction_start_date"]
    end_date = row["prediction_end_date"]
    predicted_volatility = row["predicted_5day_volatility"]

    plt.figure(figsize=(10, 5))

    plt.hlines(
        y=predicted_volatility,
        xmin=start_date,
        xmax=end_date,
        linewidth=3,
        label="Predicted 5-Day Volatility"
    )

    plt.scatter(
        [start_date, end_date],
        [predicted_volatility, predicted_volatility]
    )

    plt.title("Predicted Volatility for Next 5 Business Days")
    plt.xlabel("Prediction Period")
    plt.ylabel("Predicted Volatility")

    plt.text(
        start_date,
        predicted_volatility,
        f"{predicted_volatility:.4f} ({predicted_volatility * 100:.2f}%)",
        verticalalignment="bottom"
    )

    plt.xticks([start_date, end_date], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

df = load_data("data/aapl.us.txt")

print(df.columns)

model, y_test, predictions, features = train_lightgbm(df)

plot_actual_vs_predicted(y_test, predictions)
plot_feature_importance(model, features)


next_5_days_result = predict_next_5_days(model, df, features)
plot_next_5_days_prediction(next_5_days_result)
