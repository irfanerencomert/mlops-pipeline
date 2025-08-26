import os
import sys
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from dotenv import load_dotenv
import json
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_production_model, get_model_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.model_name = os.getenv("MODEL_NAME", "WineQualityModel")
        self.class_names = ['Class_0', 'Class_1', 'Class_2']
        self.feature_names = None

    def load_model_for_evaluation(self, model_version="Production"):
        """Load model for evaluation"""
        try:
            if model_version == "Production":
                self.model = load_production_model()
                logger.info("Loaded production model")
            else:
                # Load specific version
                model_uri = f"models:/{self.model_name}/{model_version}"
                self.model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Loaded model version: {model_version}")

            return self.model is not None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def load_test_data(self, test_size=0.2, random_state=42):
        """Load and split wine dataset for evaluation"""
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        self.feature_names = data.feature_names

        X = df.drop('target', axis=1)
        y = df['target']

        # Use same split as training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Test dataset loaded: {len(X_test)} samples")
        return X_test, y_test

    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        for i in range(len(self.class_names)):
            metrics[f'precision_class_{i}'] = precision_per_class[i]
            metrics[f'recall_class_{i}'] = recall_per_class[i]
            metrics[f'f1_class_{i}'] = f1_per_class[i]

        # ROC AUC if probabilities available
        if y_prob is not None:
            try:
                # Binarize labels for multiclass ROC
                y_bin = label_binarize(y_true, classes=[0, 1, 2])

                # Calculate ROC AUC for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()

                for i in range(len(self.class_names)):
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    metrics[f'roc_auc_class_{i}'] = roc_auc[i]

                # Macro average ROC AUC
                metrics['roc_auc_macro'] = np.mean(list(roc_auc.values()))

            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")

        return metrics

    def create_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Create and optionally save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()
        return cm

    def create_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # Pretty print
        report_str = classification_report(
            y_true, y_pred,
            target_names=self.class_names
        )

        return report, report_str

    def evaluate_model_performance(self, model_version="Production", save_plots=True):
        """Complete model evaluation pipeline"""
        if not self.load_model_for_evaluation(model_version):
            raise RuntimeError("Failed to load model for evaluation")

        # Load test data
        X_test, y_test = self.load_test_data()

        # Make predictions
        try:
            if hasattr(self.model, 'predict'):
                if 'mlflow' in str(type(self.model)):
                    # MLflow model expects DataFrame
                    predictions = self.model.predict(X_test)
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(X_test)
                    else:
                        probabilities = None
                else:
                    # Sklearn model
                    predictions = self.model.predict(X_test)
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(X_test)
                    else:
                        probabilities = None
            else:
                raise RuntimeError("Model doesn't have predict method")

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, predictions, probabilities)

        # Generate reports
        classification_dict, classification_str = self.create_classification_report(y_test, predictions)

        # Create confusion matrix
        if save_plots:
            os.makedirs("evaluation_results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cm_path = f"evaluation_results/confusion_matrix_{timestamp}.png"
            cm = self.create_confusion_matrix(y_test, predictions, cm_path)
        else:
            cm = self.create_confusion_matrix(y_test, predictions)

        # Compile evaluation results
        evaluation_results = {
            'model_version': model_version,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': len(X_test),
            'metrics': metrics,
            'classification_report': classification_dict,
            'confusion_matrix': cm.tolist()
        }

        # Save results
        if save_plots:
            results_path = f"evaluation_results/evaluation_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            logger.info(f"Evaluation results saved to {results_path}")

        # Print summary
        self.print_evaluation_summary(evaluation_results, classification_str)

        return evaluation_results

    def print_evaluation_summary(self, results, classification_str):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model Version: {results['model_version']}")
        print(f"Test Samples: {results['test_samples']}")
        print(f"Evaluation Time: {results['evaluation_timestamp']}")
        print("\n" + "-" * 40)
        print("KEY METRICS:")
        print("-" * 40)
        metrics = results['metrics']
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")

        if 'roc_auc_macro' in metrics:
            print(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")

        print("\n" + "-" * 40)
        print("DETAILED CLASSIFICATION REPORT:")
        print("-" * 40)
        print(classification_str)
        print("=" * 60)

    def compare_models(self, model_versions):
        """Compare multiple model versions"""
        comparison_results = {}

        for version in model_versions:
            logger.info(f"Evaluating model version: {version}")
            try:
                results = self.evaluate_model_performance(version, save_plots=False)
                comparison_results[version] = results['metrics']
            except Exception as e:
                logger.error(f"Failed to evaluate version {version}: {e}")
                comparison_results[version] = {"error": str(e)}

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results).T

        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(comparison_df.round(4))

        return comparison_df

    def continuous_evaluation(self, threshold_accuracy=0.85, threshold_f1=0.85):
        """Continuous evaluation for monitoring"""
        results = self.evaluate_model_performance("Production", save_plots=False)
        metrics = results['metrics']

        alerts = []

        # Check thresholds
        if metrics['accuracy'] < threshold_accuracy:
            alerts.append(f"Accuracy below threshold: {metrics['accuracy']:.4f} < {threshold_accuracy}")