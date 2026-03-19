import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🚨 Fraud Detection Dashboard")

df = pd.read_csv("outputs/fraud_scores_v3.csv")

store = st.selectbox("Store", ["All"] + sorted(df["store_id"].unique().tolist()))
checkout = st.selectbox("Checkout", ["All"] + sorted(df["checkout_id"].unique().tolist()))
risk = st.selectbox("Risk level", ["All"] + sorted(df["risk_level"].unique().tolist()))

filtered = df.copy()

if store != "All":
    filtered = filtered[filtered["store_id"] == store]

if checkout != "All":
    filtered = filtered[filtered["checkout_id"] == checkout]

if risk != "All":
    filtered = filtered[filtered["risk_level"] == risk]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", len(filtered))
col2.metric("Alertes", int(filtered["final_flag"].sum()))
col3.metric("Score moyen", round(filtered["final_risk_score"].mean(), 2))
col4.metric("Caisses uniques", filtered["checkout_id"].nunique())

st.subheader("📊 Répartition des niveaux de risque")
st.bar_chart(filtered["risk_level"].value_counts())

st.subheader("🔥 Top transactions à risque")
cols = [
    "transaction_id",
    "store_id",
    "checkout_id",
    "transaction_hour",
    "nb_items",
    "transaction_duration_sec",
    "nb_cancellations",
    "ml_risk_score",
    "rule_score",
    "final_risk_score",
    "risk_level",
    "action",
    "rule_reasons",
    "fraud_pattern",
]
st.dataframe(
    filtered.sort_values("final_risk_score", ascending=False)[cols].head(50),
    use_container_width=True
)

st.subheader("🏪 Caisses les plus souvent alertées")
checkout_alerts = (
    filtered.groupby("checkout_id", as_index=False)
    .agg(
        total_transactions=("transaction_id", "count"),
        total_alerts=("final_flag", "sum"),
        avg_risk_score=("final_risk_score", "mean"),
    )
    .sort_values(["total_alerts", "avg_risk_score"], ascending=[False, False])
)
st.dataframe(checkout_alerts.head(20), use_container_width=True)
