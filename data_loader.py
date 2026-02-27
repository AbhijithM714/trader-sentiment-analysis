import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_sentiment(csv_path: str = None):
    if csv_path is None:
        csv_path = DATA_DIR / "raw" / "fear_greed_index.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        raise ValueError("No date column found in sentiment CSV")
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    rename_map = {date_col: "date"}
    class_cols = [c for c in df.columns if "class" in c.lower() or "fear" in c.lower() or "greed" in c.lower()]
    if class_cols:
        rename_map[class_cols[0]] = "classification"
    df = df.rename(columns=rename_map)
    df = df[["date", "classification"]].dropna().reset_index(drop=True)
    return df

def load_trades(csv_path: str = None):
    if csv_path is None:
        csv_path = DATA_DIR / "raw" / "historical_data.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c: c for c in df.columns}
    ts_cols = [c for c in df.columns if c.lower() in ("time","timestamp","datetime","date_time","created_at")]
    if not ts_cols:
        ts_cols = [c for c in df.columns if "time" in c.lower()]
    if not ts_cols:
        raise ValueError("No timestamp/time column found in trades CSV")
    ts_col = ts_cols[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    rename_map = {}
    acct_cols = [c for c in df.columns if "account" in c.lower() or "wallet" in c.lower() or "client" in c.lower()]
    if acct_cols:
        rename_map[acct_cols[0]] = "account"
    pnl_cols = [c for c in df.columns if "closed" in c.lower() and "pnl" in c.lower() or c.lower() == "closedpnl" or c.lower()=="pnl"]
    if pnl_cols:
        rename_map[pnl_cols[0]] = "closed_pnl"
    size_cols = [c for c in df.columns if c.lower() in ("size","size_usd","trade_size","qty","quantity")]
    if size_cols:
        rename_map[size_cols[0]] = "size"
    side_cols = [c for c in df.columns if c.lower() in ("side","direction","position_side")]
    if side_cols:
        rename_map[side_cols[0]] = "side"
    lev_cols = [c for c in df.columns if "lever" in c.lower() or c.lower()=="leverage"]
    if lev_cols:
        rename_map[lev_cols[0]] = "leverage"
    price_cols = [c for c in df.columns if "price" in c.lower() or "execution_price" in c.lower()]
    if price_cols:
        rename_map[price_cols[0]] = "price"
    rename_map[ts_col] = "timestamp"
    df = df.rename(columns=rename_map)
    required = ["account", "timestamp", "closed_pnl"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Required column `{r}` not found in trades CSV after normalization. Columns found: {df.columns.tolist()}")
    return df