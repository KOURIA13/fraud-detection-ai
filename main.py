from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================

@dataclass
class FraudConfig:
    n_samples: int = 7000
    random_state: int = 42
    contamination: float = 0.07
    output_dir: str = "outputs"
    output_file: str = "outputs/fraud_scores_v3.csv"


# =========================================================
# DATA GENERATION
# =========================================================

def generate_checkout_data(cfg: FraudConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)

    stores = [
        {"store_id": "STORE_01", "store_format": "city", "avg_basket": 6},
        {"store_id": "STORE_02", "store_format": "city", "avg_basket": 7},
        {"store_id": "STORE_03", "store_format": "mall", "avg_basket": 9},
        {"store_id": "STORE_04", "store_format": "mall", "avg_basket": 10},
        {"store_id": "STORE_05", "store_format": "retail_park", "avg_basket": 12},
        {"store_id": "STORE_06", "store_format": "retail_park", "avg_basket": 13},
        {"store_id": "STORE_07", "store_format": "hyper", "avg_basket": 15},
        {"store_id": "STORE_08", "store_format": "hyper", "avg_basket": 16},
    ]

    checkout_ids = [f"CHK_{i:03d}" for i in range(1, 61)]

    rows = []

    for i in range(cfg.n_samples):
        store = stores[rng.integers(0, len(stores))]
        store_id = store["store_id"]
        store_format = store["store_format"]
        avg_basket = store["avg_basket"]

        checkout_id = rng.choice(checkout_ids)
        transaction_hour = int(rng.integers(8, 22))
        day_of_week = int(rng.integers(0, 7))
        is_weekend = 1 if day_of_week >= 5 else 0

        base_items = max(1, int(rng.normal(avg_basket, 3)))

        hour_factor = 1.0
        if 12 <= transaction_hour <= 14:
            hour_factor = 1.1
        elif 18 <= transaction_hour <= 20:
            hour_factor = 1.15

        weekend_factor = 1.15 if is_weekend else 1.0
        nb_items = max(1, int(base_items * hour_factor * weekend_factor))

        base_duration = 20 + nb_items * rng.normal(8, 2)
        if is_weekend:
            base_duration *= 1.05

        transaction_duration_sec = max(10, int(base_duration))
        nb_cancellations = max(0, int(rng.poisson(0.4)))

        is_synthetic_fraud = 0
        fraud_pattern = "normal"

        if rng.random() < 0.05:
            is_synthetic_fraud = 1
            fraud_pattern = rng.choice([
                "many_cancellations",
                "too_fast_for_many_items",
                "too_slow_for_few_items",
                "night_abnormal",
                "basket_inconsistent_with_store",
            ])

            if fraud_pattern == "many_cancellations":
                nb_cancellations = int(rng.integers(4, 10))
                transaction_duration_sec = int(rng.integers(90, 220))

            elif fraud_pattern == "too_fast_for_many_items":
                nb_items = int(rng.integers(14, 32))
                transaction_duration_sec = int(rng.integers(12, 45))

            elif fraud_pattern == "too_slow_for_few_items":
                nb_items = int(rng.integers(1, 4))
                transaction_duration_sec = int(rng.integers(250, 700))

            elif fraud_pattern == "night_abnormal":
                transaction_hour = int(rng.choice([8, 21]))
                nb_items = int(rng.integers(10, 25))
                transaction_duration_sec = int(rng.integers(15, 50))
                nb_cancellations = int(rng.integers(2, 6))

            elif fraud_pattern == "basket_inconsistent_with_store":
                if store_format in ["city", "mall"]:
                    nb_items = int(rng.integers(25, 45))
                else:
                    nb_items = int(rng.integers(1, 3))

        rows.append({
            "transaction_id": f"TX_{i+1:06d}",
            "store_id": store_id,
            "store_format": store_format,
            "checkout_id": checkout_id,
            "transaction_hour": transaction_hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "avg_basket_per_store": avg_basket,
            "nb_items": nb_items,
            "transaction_duration_sec": transaction_duration_sec,
            "nb_cancellations": nb_cancellations,
            "fraud_pattern": fraud_pattern,
            "is_synthetic_fraud": is_synthetic_fraud,
        })

    return pd.DataFrame(rows)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

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
        features["transaction_duration_sec"]
        / features["store_avg_duration"].clip(lower=1)
    )

    checkout_freq = features["checkout_id"].value_counts().to_dict()
    features["checkout_tx_count"] = features["checkout_id"].map(checkout_freq)

    store_freq = features["store_id"].value_counts().to_dict()
    features["store_tx_count"] = features["store_id"].map(store_freq)

    store_format_map = {"city": 0, "mall": 1, "retail_park": 2, "hyper": 3}
    features["store_format_encoded"] = features["store_format"].map(store_format_map)

    return features


# =========================================================
# BUSINESS RULES V3
# =========================================================

def apply_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    rule_reasons = []
    rule_scores = []

    for _, row in result.iterrows():
        reasons = []
        score = 0

        if row["nb_cancellations"] >= 5:
            reasons.append("high_cancellations")
            score += 40
        elif row["nb_cancellations"] >= 3:
            reasons.append("moderate_cancellations")
            score += 20

        if row["sec_per_item"] < 2.5 and row["nb_items"] >= 10:
            reasons.append("too_fast")
            score += 30

        if row["sec_per_item"] > 120:
            reasons.append("extremely_slow")
            score += 35
        elif row["sec_per_item"] > 60:
            reasons.append("slow")
            score += 20

        if row["cancel_ratio"] > 0.4:
            reasons.append("very_high_cancel_ratio")
            score += 25
        elif row["cancel_ratio"] > 0.25:
            reasons.append("high_cancel_ratio")
            score += 15

        if row["items_vs_store_avg"] > 2.2:
            reasons.append("basket_too_large")
            score += 20
        elif row["items_vs_store_avg"] < 0.3:
            reasons.append("basket_too_small")
            score += 15

        if row["transaction_hour"] in [8, 21] and row["sec_per_item"] < 4:
            reasons.append("edge_hour_fast_pattern")
            score += 20

        rule_reasons.append(", ".join(reasons) if reasons else "none")
        rule_scores.append(min(score, 100))

    result["rule_reasons"] = rule_reasons
    result["rule_score"] = rule_scores

    return result


# =========================================================
# MODEL
# =========================================================

def run_isolation_forest(df: pd.DataFrame, cfg: FraudConfig) -> pd.DataFrame:
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

    X = df[numeric_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=350,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
    )

    model.fit(X_scaled)

    result = df.copy()
    result["anomaly_score_raw"] = model.decision_function(X_scaled)

    min_s = result["anomaly_score_raw"].min()
    max_s = result["anomaly_score_raw"].max()

    result["ml_risk_score"] = (
        (max_s - result["anomaly_score_raw"]) / (max_s - min_s) * 100
    ).round(2)

    return result


# =========================================================
# FINAL SCORING
# =========================================================

def combine_scores(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["final_risk_score"] = (
        0.6 * result["ml_risk_score"] + 0.4 * result["rule_score"]
    ).round(2)

    def risk_level(score):
        if score >= 75:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"

    result["risk_level"] = result["final_risk_score"].apply(risk_level)

    def action(score):
        if score >= 75:
            return "BLOCK"
        elif score >= 60:
            return "REVIEW"
        elif score >= 40:
            return "MONITOR"
        else:
            return "OK"

    result["action"] = result["final_risk_score"].apply(action)

    result["final_flag"] = np.where(result["final_risk_score"] >= 55, 1, 0)

    return result


# =========================================================
# REPORTING
# =========================================================

def evaluate(result: pd.DataFrame):
    total = len(result)
    flagged = result["final_flag"].sum()

    fraud_total = result["is_synthetic_fraud"].sum()
    fraud_detected = result[
        (result["is_synthetic_fraud"] == 1) & (result["final_flag"] == 1)
    ].shape[0]

    recall = fraud_detected / fraud_total * 100 if fraud_total > 0 else 0

    precision = result[result["final_flag"] == 1]["is_synthetic_fraud"].mean() * 100

    print("\n====== V3 RESULTS ======")
    print(f"Transactions: {total}")
    print(f"Alertes: {flagged} ({round(flagged/total*100,2)}%)")
    print(f"Recall: {round(recall,2)}%")
    print(f"Precision: {round(precision,2)}%")

    print("\nRépartition des niveaux de risque :")
    print(result["risk_level"].value_counts())


# =========================================================
# MAIN
# =========================================================

def main():
    cfg = FraudConfig()

    df = generate_checkout_data(cfg)
    df = build_features(df)
    df = apply_business_rules(df)
    df = run_isolation_forest(df, cfg)
    df = combine_scores(df)

    evaluate(df)

    os.makedirs(cfg.output_dir, exist_ok=True)
    df.to_csv(cfg.output_file, index=False)
    print(f"\nSaved to {cfg.output_file}")


if __name__ == "__main__":
    main()