import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def define_simple_segments(daily_metrics_df,
                           pnl_threshold=None,
                           freq_threshold=None):

    df = daily_metrics_df.copy()

    # thresholds
    if pnl_threshold is None:
        pnl_threshold = df['daily_pnl'].median(skipna=True)

    if freq_threshold is None:
        freq_threshold = df['trade_count'].median(skipna=True)

    # simple rule-based segmentation
    def segment(row):
        if row['daily_pnl'] >= pnl_threshold and row['trade_count'] >= freq_threshold:
            return "High Performer"
        elif row['daily_pnl'] >= pnl_threshold:
            return "Profitable Low Activity"
        elif row['trade_count'] >= freq_threshold:
            return "Active Trader"
        else:
            return "Low Performer"

    df['segment'] = df.apply(segment, axis=1)

    return df


def cluster_traders(features, n_clusters=3):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    features = features.copy()
    features['cluster'] = labels

    sil_score = silhouette_score(features.drop(columns=['cluster']), labels)

    return features, sil_score