# ⚙️ End-to-End Wine Quality Prediction MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-orange)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready, fully containerized MLOps pipeline for automated model training, version control, deployment, and real-time monitoring — built on the Wine Quality dataset.

---

## 🚀 Overview

This project demonstrates a complete MLOps lifecycle, including:

- 🔁 **Automated Model Training** (scheduled retraining)
- 📦 **Model Versioning** with MLflow Registry
- 🐳 **Containerized Deployment** via Docker Compose
- 📊 **Live Monitoring** using Prometheus & Grafana
- 🔍 **Data Drift Detection** with Evidently AI
- 🚨 **Alerting System** integrated with Slack
- 🔧 **CI/CD Pipelines** (via GitHub Actions) for deployment & monitoring

---

## 🛠️ Tech Stack

| Category        | Tools & Frameworks                       |
|----------------|------------------------------------------|
| **ML Framework**  | MLflow, Scikit-learn                   |
| **API Service**   | Flask, Gunicorn                        |
| **Monitoring**    | Prometheus, Grafana, Evidently         |
| **Storage**       | MinIO (S3-compatible)                  |
| **Containerization** | Docker, Docker Compose             |
| **CI/CD**         | GitHub Actions                         |
| **Notifications** | Slack Webhooks                         |

---

## 📊 Monitoring Interfaces

| Tool         | URL                             |
|--------------|----------------------------------|
| MLflow UI    | `http://localhost:5000`         |
| Grafana      | `http://localhost:3000`         |
| MinIO Console| `http://localhost:9001`         |
| Prometheus   | `http://localhost:9090`         |

---

## 📦 Project Structure (Simplified)

```bash
📦 mlops-pipeline/
├── app.py                      # Flask app entrypoint
├── Dockerfile                  # Service container config
├── docker-compose.yml          # Compose setup
├── requirements.txt            # Python dependencies
├── src/                        # Source code (model, preprocessing, utils)
├── notebooks/                  # Optional exploratory notebooks
├── models/                     # Saved model artifacts
├── data/                       # Raw / processed datasets
├── mlflow_tracking.py          # Custom MLflow logic
├── infrastructure/             # Grafana/Prometheus configs
├── setup.sh                    # Setup script
├── tests/                      # Unit tests
└── README.md
```
💡 Note: CI/CD workflows (.github/workflows) and full Grafana/Prometheus configs are not included in this version but are planned for future integration.

## 📝 License

This project was developed by İrfan Eren Cömert as part of his personal MLOps portfolio.
⚠️ Unauthorized use, replication, or misrepresentation is strictly prohibited under the MIT License.
