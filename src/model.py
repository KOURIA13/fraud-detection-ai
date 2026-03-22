from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def run_isolation_forest(df: pd.DataFrame, contamination: float, random_state: int) -> pd.DataFrame:
    numeric_cols = [
        "transaction_hour",
        "day_of_week",
        "is_weekend",
        "avg_basket_per_store",
        "nb_items",
        "transaction_duration_sec",
        "nb_cancellations",
        "sec_per_item",
        "cancel_ratio",
        "items_vs_store_avg",
        "duration_vs_store_avg",
        "checkout_tx_count",
        "store_tx_count",
        "store_format_encoded",
    ]

    X = df[numeric_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=350,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)

    result = df.copy()
    result["anomaly_score_raw"] = model.decision_function(X_scaled)

    min_s = result["anomaly_score_raw"].min()
    max_s = result["anomaly_score_raw"].max()

    if max_s - min_s == 0:
        result["ml_risk_score"] = 0.0
    else:
        result["ml_risk_score"] = (
            (max_s - result["anomaly_score_raw"]) / (max_s - min_s) * 100
        ).round(2)

    return result