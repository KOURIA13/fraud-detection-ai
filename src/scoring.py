from __future__ import annotations

import os
import numpy as np
import pandas as pd


def combine_scores(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["final_risk_score"] = (
        0.6 * result["ml_risk_score"] + 0.4 * result["rule_score"]
    ).round(2)

    def risk_level(score: float) -> str:
        if score >= 75:
            return "critical"
        if score >= 60:
            return "high"
        if score >= 40:
            return "medium"
        return "low"

    def action(score: float) -> str:
        if score >= 75:
            return "BLOCK"
        if score >= 60:
            return "REVIEW"
        if score >= 40:
            return "MONITOR"
        return "OK"

    result["risk_level"] = result["final_risk_score"].apply(risk_level)
    result["action"] = result["final_risk_score"].apply(action)
    result["final_flag"] = np.where(result["final_risk_score"] >= 55, 1, 0)

    return result


def evaluate(result: pd.DataFrame) -> None:
    total = len(result)
    flagged = int(result["final_flag"].sum())

    fraud_total = int(result["is_synthetic_fraud"].sum())
    fraud_detected = int(
        result[
            (result["is_synthetic_fraud"] == 1) & (result["final_flag"] == 1)
        ].shape[0]
    )

    recall = (fraud_detected / fraud_total * 100) if fraud_total > 0 else 0.0

    flagged_df = result[result["final_flag"] == 1]
    precision = (
        flagged_df["is_synthetic_fraud"].mean() * 100 if len(flagged_df) > 0 else 0.0
    )

    print("\n====== V3 RESULTS ======")
    print(f"Transactions: {total}")
    print(f"Alertes: {flagged} ({round(flagged / total * 100, 2)}%)")
    print(f"Recall: {round(recall, 2)}%")
    print(f"Precision: {round(precision, 2)}%")

    print("\nRépartition des niveaux de risque :")
    print(result["risk_level"].value_counts())


def save_output(df: pd.DataFrame, output_dir: str, output_file: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")