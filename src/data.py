import pandas as pd
import yfinance as yf

def load_uso(start: str = "2010-01-01") -> pd.DataFrame:
    df = yf.download("USO", start=start, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present: e.g. ('Close','USO')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]
    return df.dropna()
