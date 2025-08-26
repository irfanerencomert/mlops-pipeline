import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from datetime import datetime

# Set environment variables from .env
from dotenv import load_dotenv

load_dotenv()

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# MLflow configuration
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("wine-quality-prediction")


def load_data():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df


def train_and_log_model(n_estimators=100, max_depth=None):
    with mlflow.start_run(run_name=f"rf_{n_estimators}t_{max_depth}d"):
        # Load and split data
        df = load_data()
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Log dataset info
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("test_size", len(X_test))

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log metrics
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1
        })

        # Save signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log model with signature and input example
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train.iloc[:1]
        )

        return model, accuracy, f1, mlflow.active_run().info.run_id


if __name__ == "__main__":
    best_run_id = None
    best_accuracy = 0

    print("Starting hyperparameter tuning...")

    for n_est in [50, 100, 200]:
        for depth in [None, 5, 10]:
            print(f"Training with n_estimators={n_est}, max_depth={depth}")
            model, accuracy, f1, run_id = train_and_log_model(n_estimators=n_est, max_depth=depth)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = run_id

            print(f"Run {run_id}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

    print(f"\nBest run: {best_run_id} with accuracy: {best_accuracy:.4f}")

    # Register best model
    try:
        client = mlflow.MlflowClient()
        model_uri = f"runs:/{best_run_id}/model"
        model_details = mlflow.register_model(model_uri, "WineQualityModel")

        print(f"Model registered: {model_details.name} version {model_details.version}")

        # Transition to Production
        client.transition_model_version_stage(
            name="WineQualityModel",
            version=model_details.version,
            stage="Production"
        )
        print(f"Model version {model_details.version} moved to Production stage")

    except Exception as e:
        print(f"Error registering model: {e}")
        print("Make sure MLflow server is running and accessible")