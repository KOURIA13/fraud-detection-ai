from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    features["sec_per_item"] = (
        features["transaction_duration_sec"] / features["nb_items"].clip(lower=1)
    )

    features["cancel_ratio"] = (
        features["nb_cancellations"] / features["nb_items"].clip(lower=1)
    )

    features["items_vs_store_avg"] = (
        features["nb_items"] / features["avg_basket_per_store"].clip(lower=1)
    )

    store_avg_duration = (
        features.groupby("store_id")["transaction_duration_sec"].mean().to_dict()
    )
    features["store_avg_duration"] = features["store_id"].map(store_avg_duration)

    features["duration_vs_store_avg"] = (
        features["transaction_duration_sec"] / features["store_avg_duration"].clip(lower=1)
    )

    checkout_freq = features["checkout_id"].value_counts().to_dict()
    features["checkout_tx_count"] = features["checkout_id"].map(checkout_freq)

    store_freq = features["store_id"].value_counts().to_dict()
    features["store_tx_count"] = features["store_id"].map(store_freq)

    store_format_map = {"city": 0, "mall": 1, "retail_park": 2, "hyper": 3}
    features["store_format_encoded"] = features["store_format"].map(store_format_map)

    return features