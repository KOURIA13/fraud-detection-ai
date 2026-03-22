# 🚨 AI Fraud Detection for Retail Self-Checkout

![Dashboard](assets/dashboard_overview.png)

---

## 📌 Overview

This project implements an **AI-powered fraud detection system** designed for retail self-checkout environments.

The goal is to detect suspicious transactions based on behavioral patterns such as:

- Number of items
- Transaction duration
- Cancellations
- Time of day
- Store type
- Checkout usage

The system combines **machine learning and business rules** to generate actionable decisions for store operations.

---

## ⚠️ Disclaimer

This project is based on **synthetic data** and does not use any real company or production data.

---

## 🧠 Approach

### 1. Machine Learning (Anomaly Detection)
- Model: Isolation Forest
- Detects abnormal transaction behavior without labeled fraud data

### 2. Feature Engineering
- Time per item
- Cancellation ratio
- Basket vs store average
- Transaction timing (hour, weekday, weekend)
- Store and checkout activity

### 3. Business Rules Engine
- High cancellations
- Extremely fast or slow transactions
- Abnormal basket size
- Suspicious edge-hour behavior

### 4. Hybrid Risk Scoring
Final score combines:
- 60% Machine Learning
- 40% Business rules

---

## 🚨 Risk Levels & Actions

| Risk Level | Action   |
|------------|----------|
| Low        | OK       |
| Medium     | MONITOR  |
| High       | REVIEW   |
| Critical   | BLOCK    |

Designed for real operational usage.

---

## 📊 Results

- Recall: ~60%
- Precision: ~92%
- Alert rate: ~3.3%

The system is optimized to balance fraud detection performance with operational workload.

---

## 📊 Dashboard Preview

### 🏠 Overview
![Dashboard Overview](assets/dashboard_overview.png)

### 📈 Risk Distribution
![Risk Distribution](assets/risk_distribution.png)

### 🔥 Top Risky Transactions
![Top Transactions](assets/top_transactions.png)

---

## 🏪 Use Case

Designed for:

- Retail stores with self-checkout systems
- Loss prevention teams
- Payment / transaction monitoring systems

---

## 🚀 Run the Project

```bash
python main.py
streamlit run app/streamlit_app.py