import time
import pandas as pd
import numpy as np
from prometheus_client1 import start_http_server, Gauge
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_production_model, log_data_drift_report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import requests
from dotenv import load_dotenv
import mlflow

load_dotenv()

# Prometheus Metrics
DRIFT_SCORE = Gauge('data_drift_score', 'Data drift score')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
PREDICTION_DISTRIBUTION = Gauge('prediction_distribution', 'Prediction distribution', ['class'])
LAST_ALERT_TIME = Gauge('last_alert_time', 'Timestamp of last alert sent')

# Global state
last_alert_sent = 0


def load_reference_data():
    from sklearn.datasets import load_wine
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df


def calculate_drift_score(reference, current):
    try:
        ref_features = reference.drop('target', axis=1, errors='ignore')
        curr_features = current.drop('target', axis=1, errors='ignore')

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=curr_features)
        result = report.as_dict()

        # Evidently 0.5.0+ için güncel format
        if 'metrics' in result:
            for metric in result['metrics']:
                if metric['metric'] == 'DataDriftTable':
                    drift_score = metric['result']['dataset_drift_score']
                    return drift_score, report

        # Eski format (fallback)
        drift_metrics = result['metrics'][0]['result']
        drifted_features = sum(
            1 for feature in drift_metrics['drift_by_columns'].values()
            if feature['drift_detected']
        )
        drift_score = drifted_features / len(drift_metrics['drift_by_columns'])

        return drift_score, report
    except Exception as e:
        logger.error(f"Drift calculation error: {str(e)}")
        return 0.0, None


def check_model_performance():
    """Calculate current model accuracy"""
    try:
        model = load_production_model()
        if model is None:
            print("Model not available for performance check")
            return 0.0

        df = load_reference_data()
        X = df.drop('target', axis=1)
        y = df['target']

        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        MODEL_ACCURACY.set(accuracy)

        # Track prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        for cls, count in zip(unique, counts):
            PREDICTION_DISTRIBUTION.labels(str(cls)).set(count)

        return accuracy
    except Exception as e:
        print(f"Performance check failed: {e}")
        return 0.0


def send_slack_alert(message):
    """Send alert to Slack with detailed information"""
    global last_alert_sent
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print(f"Alert would be sent: {message}")
        return

    # Prevent alert flooding
    current_time = time.time()
    if current_time - last_alert_sent < int(os.getenv("ALERT_INTERVAL", 3600)):
        print("Alert suppressed due to rate limiting")
        return

    payload = {
        "text": f":warning: *ML System Alert* :warning:\n{message}",
        "attachments": [
            {
                "color": "#ff0000",
                "fields": [
                    {
                        "title": "Environment",
                        "value": os.getenv("ENV", "development"),
                        "short": True
                    },
                    {
                        "title": "Service",
                        "value": "Wine Quality Model",
                        "short": True
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            last_alert_sent = current_time
            LAST_ALERT_TIME.set(current_time)
            print("Alert sent successfully")
        else:
            print(f"Failed to send alert: {response.status_code}")
    except Exception as e:
        print(f"Failed to send alert: {e}")


def monitor():
    """Main monitoring loop"""
    print("Starting monitoring service...")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    try:
        reference = load_reference_data()
        print(f"Reference data loaded: {len(reference)} samples")
    except Exception as e:
        print(f"Failed to load reference data: {e}")
        return

    iteration = 0
    while True:
        try:
            iteration += 1
            print(f"\n--- Monitoring Iteration {iteration} ---")

            # Simulate current production data (replace with actual data source)
            current = reference.sample(frac=0.1, random_state=iteration).reset_index(drop=True)

            # Add some noise to simulate drift
            if iteration % 10 == 0:  # Every 10th iteration, add drift
                noise_cols = current.select_dtypes(include=[np.number]).columns[:3]
                for col in noise_cols:
                    current[col] *= np.random.normal(1.0, 0.1, len(current))
                print("Added artificial drift for testing")

            # Calculate drift
            drift_score, report = calculate_drift_score(reference, current)
            if drift_score is not None:
                DRIFT_SCORE.set(drift_score)
                print(f"Drift score: {drift_score:.3f}")

            # Check performance
            accuracy = check_model_performance()
            print(f"Model accuracy: {accuracy:.3f}")

            # Check for anomalies
            drift_threshold = float(os.getenv("DRIFT_THRESHOLD", 0.1))
            accuracy_threshold = 0.85

            if drift_score and drift_score > drift_threshold:
                report_path = log_data_drift_report(reference, current)
                send_slack_alert(
                    f"Data drift detected! Score: {drift_score:.2f} "
                    f"(Threshold: {drift_threshold})\n"
                    f"Drift report saved to: {report_path}"
                )

            if accuracy < accuracy_threshold:
                send_slack_alert(
                    f"Model accuracy dropped to {accuracy:.4f} "
                    f"(Threshold: {accuracy_threshold})"
                )

        except Exception as e:
            print(f"Monitoring error: {e}")
            import traceback
            traceback.print_exc()

        # Wait until next check
        print("Waiting 5 minutes until next check...")
        time.sleep(300)  # 5 minutes


if __name__ == '__main__':
    print("Starting ML monitoring service...")
    print(f"MLflow URI: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
    print(f"Drift threshold: {os.getenv('DRIFT_THRESHOLD', 0.1)}")

    # Start Prometheus metrics server
    try:
        start_http_server(8000)
        print("Prometheus metrics server started on port 8000")
    except Exception as e:
        print(f"Failed to start Prometheus server: {e}")

    monitor()