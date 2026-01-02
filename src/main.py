import json
from pathlib import Path

import matplotlib.pyplot as plt

from .data import load_uso
from .features import make_features, FEATURE_COLS
from .model import train_predict_linear
from .backtests import backtest_long_flat

def main():
    # 1) Load + features
    df = load_uso(start="2010-01-01")
    d = make_features(df)

    # 2) Model
    pred, test, model_metrics = train_predict_linear(d, FEATURE_COLS, split_frac=0.7)

    # 3) Backtest
    bt_df, bt_metrics = backtest_long_flat(test, pred, cost_bps=10.0)

    # 4) Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    bt_df[["eq_buyhold", "eq_strat", "signal", "y", "pred_ret"]].to_csv(out_dir / "equity_curve.csv")

    metrics = {**model_metrics, **bt_metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 5) Save plot image (nice for README)
    plt.figure(figsize=(10, 5))
    plt.plot(bt_df.index, bt_df["eq_buyhold"], label="Buy & Hold")
    plt.plot(bt_df.index, bt_df["eq_strat"], label="Strategy (net)")
    plt.title("USO: Linear Regression Signal Strategy (Out-of-sample)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=200)
    plt.close()

    print("Saved:", out_dir / "metrics.json")
    print("Saved:", out_dir / "equity_curve.csv")
    print("Saved:", out_dir / "equity_curve.png")
    print("Metrics summary:", metrics)

if __name__ == "__main__":
    main()
