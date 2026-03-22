from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FraudConfig:
    n_samples: int = 7000
    random_state: int = 42
    contamination: float = 0.07
    output_dir: str = "outputs"
    output_file: str = "outputs/fraud_scores_v3.csv"


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