import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def train_predict_linear(
    d: pd.DataFrame,
    feature_cols: list[str],
    split_frac: float = 0.7,
):
    """
    Train on first split_frac of data, predict on the remainder (time-ordered).
    Returns: pred (np.ndarray), test_df (pd.DataFrame), metrics (dict)
    """
    split = int(len(d) * split_frac)

    X = d[feature_cols]
    y = d["y"]

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    test = d.iloc[split:].copy()

    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    corr = float(np.corrcoef(y_test.values, pred)[0, 1])

    metrics = {
        "split_frac": float(split_frac),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "mse": float(mse),
        "corr": float(corr),
    }

    return pred, test, metrics
