# LinkedIn Post (ready to copy)

> Numbers below reflect the project's **actual** measured results, not aspirational targets.

---

📊 I built a **Real-Time Customer Churn Intelligence Platform** — an end-to-end MLOps system that predicts churn, explains *why* with SHAP, and recommends ROI-ranked retention actions in real time.

Not just a notebook with a model — a containerized, monitored, API-driven platform.

🔧 **Tech Stack**
• Model: XGBoost + SHAP explainability
• API: FastAPI (3 endpoints, auto-docs, Pydantic validation)
• Dashboard: Streamlit + Plotly (6 interactive sections)
• Model Registry: MLflow (experiment tracking + production aliasing)
• Storage: SQLite / SQLAlchemy
• Deployment: Docker + GitHub Actions CI/CD

📈 **Results**
✓ AUC-ROC **0.87** on held-out data
✓ Real-time scoring at **~20 ms/customer** (predict + SHAP + intervention)
✓ Risk-segments every customer (HIGH / MEDIUM / LOW)
✓ Quantifies revenue at risk and ranks interventions by net benefit
✓ 29 automated tests, CI building + smoke-testing the Docker image

🎯 **How it works**
1. Customer data → XGBoost predicts churn probability
2. SHAP answers "why is this customer at risk?" (top-3 drivers, per customer)
3. Intervention engine recommends an action, scored on ROI
4. Streamlit dashboard surfaces real-time business insights
5. FastAPI exposes it for integration with other systems
6. MLflow tracks model versions; monitoring watches drift (PSI) + performance

💡 **The part I'm most proud of**
I resisted the temptation to inflate metrics. Real churn data has overlapping classes — you *can't* hit 0.75+ precision AND 0.78 recall simultaneously without faking separability. So I optimized for honest, deployable performance (0.87 AUC) and built the monitoring to catch drift the way production systems actually do — distribution drift (PSI) as a leading indicator, since streamed customers are unlabeled and true performance can't be measured live.

Engineering judgment > vanity metrics.

🚀 **Enterprise-grade features**
• Explainable AI for every single prediction
• Business framing: revenue at risk, intervention cost, ROI, net benefit
• Model monitoring: drift detection, latency, throughput
• Containerized + CI/CD (automated tests on every push)

GitHub: [link]

#DataScience #MachineLearning #MLOps #Python #FastAPI #Streamlit #Docker #XGBoost #SHAP #ProductionML
