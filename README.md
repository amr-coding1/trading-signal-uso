# USO Signal Research (Linear Regression)

A minimal, reproducible “signal → model → backtest” research pipeline on the USO (oil) ETF.

## Goal
Test whether simple technical features contain predictive power for next-day returns, using:
- returns (1d, 5d)
- price vs moving averages (5/10/20)
- rolling volatility (10/20)

## Method
- Data: daily USO (Yahoo Finance)
- Model: linear regression (standardised features)
- Validation: time-ordered train/test split (out-of-sample)
- Strategy: long/flat if predicted return > 0
- Costs: 10 bps per trade

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
