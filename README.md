# Telecom Churn Advisor

AI-powered Telecom Customer Churn Prediction and Retention Advisor.

This project provides an internal CRM dashboard to help retention teams identify customers with high churn risk and receive AI-assisted retention recommendations.

## Features

- Telecom customer churn prediction
- Customer profile and churn-risk simulation
- Retention action recommendations
- AI Retention Advisor
- Selective RAG retrieval
- Visible retrieved knowledge chunks
- LLM model and token usage monitoring
- MLflow experiment tracking
- Best model registration
- Automated data pipeline
- Docker containerization
- Prompt injection defense
- Public Streamlit deployment

---

## System Architecture

```text
Raw Telco Customer Data
        |
        v
+-----------------------+
|   Data Pipeline       |
|                       |
| Ingestion             |
| Cleaning              |
| Validation            |
| Storage               |
+-----------+-----------+
            |
            v
Processed Customer Data
            |
            v
+-----------------------+
| ML Training Pipeline  |
|                       |
| Logistic Regression   |
| Random Forest 100     |
| Random Forest 200     |
+-----------+-----------+
            |
            v
 Evaluation Metrics
 Accuracy / Precision
 Recall / F1 / ROC-AUC
            |
            v
      Best Model
            |
            v
    churn_model.pkl
            |
            v
+-----------------------+
| Streamlit Application |
+-----------------------+
       |           |
       |           |
       v           v
 Churn Model    AI Advisor
                    |
                    v
              RAG Router
                    |
          +---------+---------+
          |                   |
       Retrieve            No Retrieve
          |                   |
          v                   |
 Knowledge Base              |
          |                   |
          +---------+---------+
                    |
                    v
              OpenRouter LLM
                    |
                    v
             AI Recommendation