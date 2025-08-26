import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_model_client, register_new_model_version

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class WineQualityTrainer:
    def __init__(self, experiment_name="wine-quality-training"):
        self.experiment_name = experiment_name
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {self.mlflow_uri}")
        logger.info(f"Experiment: {experiment_name}")

    def load_and_prepare_data(self):
        """Load and prepare wine dataset"""
        logger.info("Loading wine dataset...")
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        # Log dataset info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Classes: {np.unique(data.target)}")
        logger.info(f"Class distribution: {pd.Series(data.target).value_counts().to_dict()}")

        return df

    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, hyperparams=None):
        """Train RandomForest model"""
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }

        logger.info(f"Training model with hyperparameters: {hyperparams}")
        model = RandomForestClassifier(**hyperparams)
        model.fit(X_train, y_train)

        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }

        logger.info(f"Model metrics: {metrics}")
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))

        return metrics, y_pred

    def hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        """Perform hyperparameter tuning"""
        logger.info("Starting hyperparameter tuning...")

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        # Evaluate best model
        best_model = grid_search.best_estimator_
        metrics, y_pred = self.evaluate_model(best_model, X_test, y_test)

        return best_model, grid_search.best_params_, metrics

    def train_with_mlflow(self, perform_tuning=True):
        """Complete training pipeline with MLflow logging"""
        run_name = f"wine_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")

            # Load and prepare data
            df = self.load_and_prepare_data()
            X_train, X_test, y_train, y_test = self.split_data(df)

            # Log data info
            mlflow.log_params({
                "dataset_size": len(df),
                "n_features": X_train.shape[1],
                "n_classes": len(np.unique(y_train)),
                "test_size": len(X_test),
                "train_size": len(X_train)
            })

            if perform_tuning:
                # Hyperparameter tuning
                model, best_params, metrics = self.hyperparameter_tuning(
                    X_train, y_train, X_test, y_test
                )
                mlflow.log_params(best_params)
            else:
                # Simple training
                model = self.train_model(X_train, y_train)
                metrics, _ = self.evaluate_model(model, X_test, y_test)
                mlflow.log_params({
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                })

            # Log metrics
            mlflow.log_metrics(metrics)

            # Create model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, model.predict(X_train))

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_train.iloc[:1],
                registered_model_name=os.getenv("MODEL_NAME", "WineQualityModel")
            )

            # Save local backup
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, f"models/model_{run.info.run_id}.pkl")
            joblib.dump(model, "models/best_model.pkl")  # Latest model

            logger.info(f"Model saved locally: models/model_{run.info.run_id}.pkl")
            logger.info(f"MLflow run completed: {run.info.run_id}")

            return model, metrics, run.info.run_id


def main():
    """Main training function"""
    trainer = WineQualityTrainer()

    try:
        model, metrics, run_id = trainer.train_with_mlflow(perform_tuning=True)

        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Final F1 Score: {metrics['f1_score']:.4f}")
        logger.info("=" * 50)

        # Try to promote to production if accuracy is good
        if metrics['accuracy'] > 0.85:
            try:
                client = get_model_client()
                model_name = os.getenv("MODEL_NAME", "WineQualityModel")

                # Get latest version
                latest_versions = client.get_latest_versions(model_name, stages=["None"])
                if latest_versions:
                    latest_version = latest_versions[0].version

                    # Transition to Production
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version,
                        stage="Production"
                    )
                    logger.info(f"Model version {latest_version} promoted to Production")

            except Exception as e:
                logger.warning(f"Failed to promote model to production: {e}")

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)