import pandas as pd
from pathlib import Path

# -----------------------------
# Output directory
# -----------------------------
PROCESSED_DIR = Path(r"D:\Abhiijith\trader-sentiment-analysis\data\processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# SENTIMENT CLEANING
# =============================
def clean_sentiment(sent_df: pd.DataFrame) -> pd.DataFrame:
    df = sent_df.copy()

    # standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # convert date
    df['date'] = pd.to_datetime(df['date'], errors="coerce").dt.normalize()

    # clean sentiment label
    df['classification'] = (
        df['classification']
        .astype(str)
        .str.strip()
    )

    # remove invalid rows
    df = df.dropna(subset=['date'])

    # remove duplicates
    df = df.drop_duplicates(subset=['date'])

    df = df.sort_values('date').reset_index(drop=True)

    return df


# =============================
# TRADES CLEANING
# =============================
def clean_trades(trades_df):

    df = trades_df.copy()

    # normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # --------------------------------
    # HANDLE TIMESTAMP SAFELY
    # --------------------------------
    if "timestamp ist" in df.columns:
        df["timestamp"] = df["timestamp ist"]
        df = df.drop(columns=["timestamp ist"])

    # remove duplicate timestamp column if exists
    df = df.loc[:, ~df.columns.duplicated()]

    # --------------------------------
    # rename important fields
    # --------------------------------
    rename_map = {
        "closed_pnl": "pnl",
        "size usd": "trade_size"
    }

    df = df.rename(columns=rename_map)

    # --------------------------------
    # basic cleaning
    # --------------------------------
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    # timestamp conversion
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])

    # --------------------------------
    # side normalization
    # --------------------------------
    if "side" in df.columns:
        df["side"] = (
            df["side"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({
                "buy": "long",
                "sell": "short"
            })
        )

    # --------------------------------
    # numeric conversion
    # --------------------------------
    for col in ["pnl", "trade_size", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["account"])

    # --------------------------------
    # create daily column
    # --------------------------------
    df["date"] = df["timestamp"].dt.floor("D")

    return df

# =============================
# SAVE FUNCTION
# =============================
def save_processed(df: pd.DataFrame, filename: str):
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    return output_path