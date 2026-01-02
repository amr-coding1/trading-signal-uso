import pandas as pd

FEATURE_COLS = ["ret_1", "ret_5", "px_sma_5", "px_sma_10", "px_sma_20", "vol_10", "vol_20"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["ret_1"] = d["close"].pct_change(1)
    d["ret_5"] = d["close"].pct_change(5)

    for w in [5, 10, 20]:
        d[f"sma_{w}"] = d["close"].rolling(w).mean()
        d[f"px_sma_{w}"] = d["close"] / d[f"sma_{w}"] - 1.0

    d["vol_10"] = d["ret_1"].rolling(10).std()
    d["vol_20"] = d["ret_1"].rolling(20).std()

    # Predict next-day return (target)
    d["y"] = d["ret_1"].shift(-1)

    return d.dropna()
