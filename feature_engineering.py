import pandas as pd
import numpy as np

# ====================================
# DAILY TRADER METRICS
# ====================================
def compute_daily_metrics(trades_df):
    import numpy as np
    df = trades_df.copy()
    
    # 1. Clean the direction column to ensure string matching works
    if 'direction' in df.columns:
        df['direction'] = df['direction'].astype(str).str.lower()
    
    # 2. Define base aggregations
    agg_dict = {
        'daily_pnl': ('pnl', 'sum'),
        'trade_count': ('pnl', 'count'),
        'avg_trade_size': ('trade_size', 'mean'),
        'median_trade_size': ('trade_size', 'median'),
        'worst_trade_pnl': ('pnl', 'min') # Drawdown proxy
    }
    
    # 3. Only aggregate leverage if you created that column during data cleaning
    if 'leverage' in df.columns:
        agg_dict['avg_leverage'] = ('leverage', 'mean')
        
    group = df.groupby(['date', 'account'])
    agg = group.agg(**agg_dict).reset_index()

    # 4. Win / Loss calculations
    wins = df[df['pnl'] > 0].groupby(['date', 'account']).size().rename('win_count')
    losses = df[df['pnl'] <= 0].groupby(['date', 'account']).size().rename('loss_count')

    agg = agg.merge(wins, on=['date','account'], how='left')
    agg = agg.merge(losses, on=['date','account'], how='left')
    agg[['win_count','loss_count']] = agg[['win_count','loss_count']].fillna(0)

    agg['win_rate'] = (agg['win_count'] / (agg['win_count'] + agg['loss_count'])).fillna(0)

    # 5. Long / Short Ratio calculations
    if 'direction' in df.columns:
        longs = df[df['direction'] == 'long'].groupby(['date', 'account']).size().rename('long_count')
        shorts = df[df['direction'] == 'short'].groupby(['date', 'account']).size().rename('short_count')
        
        agg = agg.merge(longs, on=['date','account'], how='left')
        agg = agg.merge(shorts, on=['date','account'], how='left')
        agg[['long_count', 'short_count']] = agg[['long_count', 'short_count']].fillna(0)
        
        # Calculate ratio safely (if shorts == 0, default to the number of longs to avoid dividing by zero)
        agg['long_short_ratio'] = np.where(
            agg['short_count'] == 0, 
            agg['long_count'], 
            agg['long_count'] / agg['short_count']
        )

    return agg


# ====================================
# MERGE WITH SENTIMENT
# ====================================
def merge_with_sentiment(
    metrics_df: pd.DataFrame,
    sentiment_df: pd.DataFrame
) -> pd.DataFrame:

    # ===============================
    # MERGE METRICS + SENTIMENT
    # ===============================
    merged = pd.merge(
        metrics_df,
        sentiment_df,
        on='date',
        how='left'
    )

    # ===============================
    # SORT (VERY IMPORTANT)
    # ===============================
    merged = merged.sort_values(['account', 'date'])

    # ===============================
    # LAG FEATURES
    # ===============================

    # Previous day PnL
    merged['pnl_lag1'] = (
        merged.groupby('account')['daily_pnl']
        .shift(1)
    )

    # 3-day rolling average pnl
    merged['pnl_roll3'] = (
        merged.groupby('account')['daily_pnl']
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Previous win rate
    merged['winrate_lag1'] = (
        merged.groupby('account')['win_rate']
        .shift(1)
    )

    # Trading activity trend
    merged['tradecount_roll3'] = (
        merged.groupby('account')['trade_count']
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ===============================
    # CLEAN NULL VALUES
    # ===============================
    merged = merged.fillna(0)

    return merged