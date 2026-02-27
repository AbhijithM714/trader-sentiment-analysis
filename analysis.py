import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_dataset(trades_df: pd.DataFrame, sentiment_df: pd.DataFrame):
    print("Trades shape:", trades_df.shape)
    print("Sentiment shape:", sentiment_df.shape)
    print("\nTrades missing values:\n", trades_df.isnull().sum())
    print("\nSentiment missing values:\n", sentiment_df.isnull().sum())
    print("\nTrades duplicates:", trades_df.duplicated().sum())
    print("\nSentiment duplicates:", sentiment_df.duplicated().sum())

def compare_by_sentiment(merged_df: pd.DataFrame, save_fig: str = None):
    df = merged_df.copy()
    def coarse(c):
        if pd.isna(c): 
            return "Unknown"
        c = str(c).lower()
        if "fear" in c:
            return "Fear"
        if "greed" in c:
            return "Greed"
        return "Other"
    df['sentiment_coarse'] = df['classification'].apply(coarse)
    stat = df.groupby('sentiment_coarse').agg(
        mean_daily_pnl = ('daily_pnl','mean'),
        median_daily_pnl = ('daily_pnl','median'),
        mean_win_rate = ('win_rate','mean'),
        avg_trade_count = ('trade_count','mean'),
    ).reset_index()
    print(stat)
    plt.figure(figsize=(8,5))
    sns.boxplot(x='sentiment_coarse', y='daily_pnl', data=df)
    plt.title("Daily PnL by Sentiment (coarse)")
    if save_fig:
        plt.savefig(save_fig.replace(".png","_pnl_box.png"), bbox_inches="tight")
    plt.show()
    plt.figure(figsize=(6,4))
    sns.barplot(x='sentiment_coarse', y='win_rate', data=df, estimator=np.mean)
    plt.title("Average Win Rate by Sentiment (coarse)")
    if save_fig:
        plt.savefig(save_fig.replace(".png","_winrate_bar.png"), bbox_inches="tight")
    plt.show()
    return stat

def plot_trade_counts_time_series(merged_df: pd.DataFrame):
    ts = merged_df.groupby('date')['trade_count'].sum().reset_index()
    plt.figure(figsize=(12,4))
    plt.plot(ts['date'], ts['trade_count'])
    plt.title("Total trades per day")
    plt.xlabel("Date")
    plt.ylabel("Total trades")
    plt.show()