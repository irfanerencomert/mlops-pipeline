#!/bin/bash

echo "🚀 MLOps Pipeline Kurulum Başlıyor..."

# Eksik dizinleri oluştur
echo "📁 Dizin yapısı oluşturuluyor..."
mkdir -p models data mlruns infrastructure logs

# .env dosyası kontrolü
if [ ! -f ".env" ]; then
    echo "⚙️ .env dosyası oluşturuluyor..."
    cat > .env << 'EOF'
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123

# Model Configuration
MODEL_NAME=WineQualityModel
PRODUCTION_STAGE=Production

# Monitoring Configuration
DRIFT_THRESHOLD=0.15
ALERT_INTERVAL=3600
ENV=development

# Slack Configuration (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/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

# API Configuration
PORT=5000
EOF
fi

# Infrastructure dosyalarını taşı
echo "🏗️ Infrastructure dosyaları düzenleniyor..."
if [ -f "prometheus.yml" ]; then
    mv prometheus.yml infrastructure/
fi

if [ -f "grafana-dashboard.json" ]; then
    mv grafana-dashboard.json infrastructure/
fi

# Docker Compose dosyasını güncelle
echo "🐳 Docker Compose konfigürasyonu güncelleniyor..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    environment:
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=minio
      - AWS_SECRET_ACCESS_KEY=minio123
      - MLFLOW_S3_IGNORE_TLS=true
    command: >
      mlflow server --host 0.0.0.0
      --backend-store-uri sqlite:///mlruns/mlflow.db
      --default-artifact-root s3://mlflow/
      --registry-store-uri sqlite:///mlruns/registry.db
    volumes:
      - ./mlruns:/mlruns
      - mlflow_db:/mlflow
    ports:
      - "5000:5000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/')"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    depends_on:
      - minio

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
      MINIO_DEFAULT_BUCKETS: "mlflow"
    volumes:
      - minio-data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  minio-setup:
    image: minio/mc
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      until (/usr/bin/mc alias set minio http://minio:9000 minio minio123) do echo '...waiting...' && sleep 1; done;
      /usr/bin/mc mb minio/mlflow;
      /usr/bin/mc anonymous set public minio/mlflow;
      echo 'MinIO setup complete';
      "

  app:
    build: .
    ports:
      - "5001:5000"
    environment:
      - PORT=5000
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_NAME=WineQualityModel
    depends_on:
      mlflow:
        condition: service_healthy
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 10
    volumes:
    - ./models:/app/models

  monitor:
    build: .
    command: python src/monitor.py
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_NAME=WineQualityModel
      - DRIFT_THRESHOLD=0.15
      - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=minio
      - AWS_SECRET_ACCESS_KEY=minio123
    depends_on:
      mlflow:
        condition: service_healthy
      app:
        condition: service_healthy
    ports:
      - "8000:8000"

    prometheus:
    image: prom/prometheus:v2.40.0
    ports:
      - "9090:9090"
    volumes:
      - ./infrastructure/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    depends_on:
      - app
      - monitor

  grafana:
    image: grafana/grafana:9.5.2
    ports:
      - "3000:3000"
    volumes:
      - ./infrastructure/grafana-dashboard.json:/etc/grafana/provisioning/dashboards/dashboard.json
      - grafana-data:/var/lib/grafana
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    depends_on:
      - prometheus

volumes:
  mlflow_db:
  minio-data:
  grafana-data:
  prometheus-data:
EOF

echo "✅ Kurulum tamamlandı!"
echo ""
echo "🚀 Başlangıç komutları:"
echo "1. Servisleri başlat: docker-compose up -d"
echo "2. MLflow'u kontrol et: http://localhost:5000"
echo "3. Model eğit: python mlflow_tracking.py"
echo "4. API'yi test et: python tests/test_api.py"
echo "5. Grafana: http://localhost:3000"
echo "6. Prometheus: http://localhost:9090"
echo ""
echo "📋 Önemli Notlar:"
echo "- .env dosyasını ihtiyaçlarınıza göre düzenleyin"
echo "- Slack webhook URL'sini eklemek isterseniz .env dosyasını güncelleyin"
echo "- İlk çalıştırmada servislerin başlaması 2-3 dakika sürebilir"