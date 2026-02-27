import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def prepare_features(merged_df: pd.DataFrame, lag_days: int = 1):

    df = merged_df.copy()
    df = df.sort_values(['account','date'])

    lower = df['daily_pnl'].quantile(0.01)
    upper = df['daily_pnl'].quantile(0.99)
    df = df[df['daily_pnl'].between(lower, upper)]

    df['sentiment_code'] = df['classification'].fillna("Unknown") \
        .astype(str) \
        .apply(lambda x: 1 if 'greed' in x.lower()
               else (-1 if 'fear' in x.lower() else 0))

    df['next_daily_pnl'] = df.groupby('account')['daily_pnl'].shift(-1)

    df['target_profit_next'] = (
        df['next_daily_pnl'] > 0
    ).astype(int)

    df['next_daily_pnl_log'] = (
        np.sign(df['next_daily_pnl']) *
        np.log1p(np.abs(df['next_daily_pnl']))
    )

    feature_cols = [
        'sentiment_code',
        'win_rate',
        'avg_trade_size',
        'trade_count',
        'pnl_lag1',
        'pnl_roll3',
        'winrate_lag1',
        'tradecount_roll3'
    ]

    X = df[feature_cols].fillna(0)
    y = df['target_profit_next']

    mask = ~df['next_daily_pnl_log'].isna()

    return X[mask], y[mask], df[mask]

def train_classifiers(X, y, test_size=0.3, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    models = {}
    lr = LogisticRegression(max_iter=1000,class_weight='balanced')
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, pred_lr)
    models['logistic'] = {'model': lr, 'acc': acc_lr, 'report': classification_report(y_test, pred_lr,zero_division=0)}
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state,class_weight='balanced')
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)
    models['random_forest'] = {'model': rf, 'acc': acc_rf, 'report': classification_report(y_test, pred_rf)}
    joblib.dump(lr, MODEL_DIR / "logistic.pkl")
    joblib.dump(rf, MODEL_DIR / "rf_classifier.pkl")
    return models

def train_regressor(X, y_reg, test_size=0.3, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=test_size, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    joblib.dump(rf, MODEL_DIR / "rf_regressor.pkl")
    return {'model': rf, 'mse': mse, 'r2': r2}