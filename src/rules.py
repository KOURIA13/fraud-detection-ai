from __future__ import annotations

import pandas as pd


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