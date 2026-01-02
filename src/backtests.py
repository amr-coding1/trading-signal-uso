import pandas as pd
from .utils import sharpe_annualised, max_drawdown

def backtest_long_flat(
    test: pd.DataFrame,
    pred: pd.Series | pd.DataFrame | list | pd.Index,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """
    test must contain:
      - y : next-day return (actual)
    pred is predicted next-day return aligned to test rows.
    """
    out = test.copy()
    out["pred_ret"] = pred

    out["signal"] = (out["pred_ret"] > 0).astype(int)

    out["strat_ret_gross"] = out["signal"] * out["y"]

    cost = cost_bps / 10000.0
    out["turnover"] = out["signal"].diff().abs().fillna(0)
    out["strat_ret_net"] = out["strat_ret_gross"] - out["turnover"] * cost

    out["eq_buyhold"] = (1 + out["y"]).cumprod()
    out["eq_strat"] = (1 + out["strat_ret_net"]).cumprod()

    bt_metrics = {
        "cost_bps": float(cost_bps),
        "final_eq_buyhold": float(out["eq_buyhold"].iloc[-1]),
        "final_eq_strat": float(out["eq_strat"].iloc[-1]),
        "sharpe_strat": sharpe_annualised(out["strat_ret_net"]),
        "max_dd_strat": max_drawdown(out["eq_strat"]),
        "avg_turnover": float(out["turnover"].mean()),
    }

    return out, bt_metrics
