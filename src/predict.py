import os
import sys
import numpy as np
import pandas as pd
import mlflow
import joblib
from dotenv import load_dotenv
import logging
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_production_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class WineQualityPredictor:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.feature_names = [
            'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue',
            'od280/od315_of_diluted_wines', 'proline'
        ]
        self.class_names = {0: 'Class_0', 1: 'Class_1', 2: 'Class_2'}
        self.load_model()

    def load_model(self):
        """Load the production model"""
        try:
            logger.info("Loading production model...")
            self.model = load_production_model()

            if self.model is None:
                logger.warning("Production model not found, loading local backup...")
                self.model = joblib.load('models/best_model.pkl')
                self.model_version = "local_backup"
            else:
                self.model_version = "production"

            logger.info(f"Model loaded successfully: {self.model_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def validate_input(self, features):
        """Validate input features"""
        if not isinstance(features, (list, np.ndarray)):
            raise ValueError("Features must be a list or numpy array")

        if len(features) != 13:
            raise ValueError(f"Expected 13 features, got {len(features)}")

        # Convert to float and check for valid numbers
        try:
            features = [float(f) for f in features]
        except (ValueError, TypeError):
            raise ValueError("All features must be numeric")

        # Basic range validation
        feature_ranges = {
            'alcohol': (11.0, 15.0),
            'malic_acid': (0.7, 5.8),
            'ash': (1.4, 3.2),
            'alcalinity_of_ash': (10.0, 30.0),
            'magnesium': (70, 162),
            'total_phenols': (0.98, 3.9),
            'flavanoids': (0.34, 5.1),
            'nonflavanoid_phenols': (0.13, 0.66),
            'proanthocyanins': (0.41, 3.6),
            'color_intensity': (1.3, 13.0),
            'hue': (0.48, 1.71),
            'od280/od315_of_diluted_wines': (1.3, 4.0),
            'proline': (278, 1680)
        }

        warnings = []
        for i, (feature_name, (min_val, max_val)) in enumerate(feature_ranges.items()):
            if not (min_val <= features[i] <= max_val):
                warnings.append(f"{feature_name}: {features[i]} (expected: {min_val}-{max_val})")

        return features, warnings

    def predict_single(self, features, return_probabilities=False):
        """Make prediction for single sample"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Validate input
        features, warnings = self.validate_input(features)

        # Convert to DataFrame for MLflow models or array for sklearn
        if hasattr(self.model, 'predict'):
            if 'mlflow' in str(type(self.model)):
                # MLflow model expects DataFrame
                df = pd.DataFrame([features], columns=self.feature_names)
                prediction = self.model.predict(df)
                if return_probabilities and hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(df)
                else:
                    probabilities = None
            else:
                # Sklearn model expects array
                prediction = self.model.predict([features])
                if return_probabilities and hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba([features])
                else:
                    probabilities = None
        else:
            raise RuntimeError("Model doesn't have predict method")

        # Format result
        predicted_class = int(prediction[0])
        predicted_label = self.class_names.get(predicted_class, f"Unknown_{predicted_class}")

        result = {
            'prediction': predicted_class,
            'predicted_label': predicted_label,
            'model_version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'warnings': warnings
        }

        if probabilities is not None:
            result['probabilities'] = {
                self.class_names.get(i, f"Class_{i}"): float(prob)
                for i, prob in enumerate(probabilities[0])
            }
            result['confidence'] = float(max(probabilities[0]))

        return result

    def predict_batch(self, features_list, return_probabilities=False):
        """Make predictions for multiple samples"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        results = []
        for i, features in enumerate(features_list):
            try:
                result = self.predict_single(features, return_probabilities)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        return results

    def get_feature_importance(self):
        """Get feature importance if available"""
        if self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = {
                self.feature_names[i]: float(importance[i])
                for i in range(len(importance))
            }
            # Sort by importance
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        return None

    def get_model_info(self):
        """Get model information"""
        info = {
            'model_loaded': self.model is not None,
            'model_version': self.model_version,
            'model_type': str(type(self.model)) if self.model else None,
            'expected_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }

        # Add model parameters if available
        if self.model and hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()

        return info


def main():
    """Test the predictor"""
    predictor = WineQualityPredictor()

    # Test with sample data
    sample_features = [13.20, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]

    try:
        # Single prediction
        result = predictor.predict_single(sample_features, return_probabilities=True)
        print("\nSingle Prediction:")
        print(json.dumps(result, indent=2))

        # Batch prediction
        batch_results = predictor.predict_batch([sample_features, sample_features], return_probabilities=True)
        print("\nBatch Prediction:")
        print(json.dumps(batch_results, indent=2))

        # Model info
        model_info = predictor.get_model_info()
        print("\nModel Info:")
        print(json.dumps(model_info, indent=2))

        # Feature importance
        importance = predictor.get_feature_importance()
        if importance:
            print("\nFeature Importance:")
            for feature, imp in list(importance.items())[:5]:  # Top 5
                print(f"{feature}: {imp:.4f}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()