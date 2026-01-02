import numpy as np
import pandas as pd

def sharpe_annualised(daily_returns: pd.Series) -> float:
    if daily_returns.std() == 0:
        return float("nan")
    return float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())
