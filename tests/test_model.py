import pytest
import numpy as np
import pandas as pd
import os
import sys
import joblib
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import WineQualityPredictor
from train import WineQualityTrainer
from evaluate import ModelEvaluator


class TestWineQualityPredictor:
    """Test suite for WineQualityPredictor"""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = Mock()
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.2, 0.7, 0.1]])
        model.feature_importances_ = np.random.random(13)
        return model

    @pytest.fixture
    def predictor(self, mock_model):
        """Create predictor with mocked model"""
        predictor = WineQualityPredictor()
        predictor.model = mock_model
        predictor.model_version = "test"
        return predictor

    def test_validate_input_valid(self, predictor):
        """Test input validation with valid input"""
        features = [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]
        validated, warnings = predictor.validate_input(features)

        assert len(validated) == 13
        assert isinstance(validated, list)
        assert all(isinstance(f, float) for f in validated)

    def test_validate_input_wrong_length(self, predictor):
        """Test input validation with wrong number of features"""
        features = [1, 2, 3]  # Too short

        with pytest.raises(ValueError, match="Expected 13 features"):
            predictor.validate_input(features)

    def test_validate_input_non_numeric(self, predictor):
        """Test input validation with non-numeric values"""
        features = ['a'] * 13  # Non-numeric

        with pytest.raises(ValueError, match="All features must be numeric"):
            predictor.validate_input(features)

    def test_predict_single_success(self, predictor):
        """Test successful single prediction"""
        features = [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]

        result = predictor.predict_single(features, return_probabilities=True)

        assert 'prediction' in result
        assert 'predicted_label' in result
        assert 'model_version' in result
        assert 'timestamp' in result
        assert 'probabilities' in result
        assert 'confidence' in result
        assert result['prediction'] == 1
        assert result['model_version'] == "test"

    def test_predict_batch(self, predictor):
        """Test batch prediction"""
        features_list = [
            [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050],
            [12.5, 1.5, 2.0, 10.0, 90, 2.4, 2.5, 0.3, 1.2, 4.0, 1.0, 3.2, 900]
        ]

        results = predictor.predict_batch(features_list, return_probabilities=True)

        assert len(results) == 2
        assert all('sample_id' in r for r in results)
        assert all('prediction' in r for r in results)

    def test_get_feature_importance(self, predictor):
        """Test feature importance extraction"""
        importance = predictor.get_feature_importance()

        assert importance is not None
        assert len(importance) == 13
        assert all(isinstance(v, float) for v in importance.values())

    def test_get_model_info(self, predictor):
        """Test model info extraction"""
        info = predictor.get_model_info()

        assert info['model_loaded'] is True
        assert info['model_version'] == "test"
        assert info['expected_features'] == 13
        assert len(info['feature_names']) == 13


class TestWineQualityTrainer:
    """Test suite for WineQualityTrainer"""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance"""
        with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):
            return WineQualityTrainer("test_experiment")

    def test_load_and_prepare_data(self, trainer):
        """Test data loading"""
        df = trainer.load_and_prepare_data()

        assert isinstance(df, pd.DataFrame)
        assert 'target' in df.columns
        assert len(df) > 0
        assert len(df.columns) == 14  # 13 features + target

    def test_split_data(self, trainer):
        """Test data splitting"""
        df = trainer.load_and_prepare_data()
        X_train, X_test, y_train, y_test = trainer.split_data(df)

        assert len(X_train) > len(X_test)  # 80/20 split
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == 13  # 13 features

    def test_train_model(self, trainer):
        """Test model training"""
        df = trainer.load_and_prepare_data()
        X_train, X_test, y_train, y_test = trainer.split_data(df)

        model = trainer.train_model(X_train, y_train)

        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'predict')
        assert model.n_estimators == 100

    def test_evaluate_model(self, trainer):
        """Test model evaluation"""
        df = trainer.load_and_prepare_data()
        X_train, X_test, y_train, y_test = trainer.split_data(df)
        model = trainer.train_model(X_train, y_train)

        metrics, y_pred = trainer.evaluate_model(model, X_test, y_test)

        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert len(y_pred) == len(y_test)


class TestModelEvaluator:
    """Test suite for ModelEvaluator"""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return ModelEvaluator()

    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        X = df.drop(columns=[], axis=1)  # No target column in features
        y = data.target

        # Small test set
        X_test, _, y_test, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_test[:20], y_test[:20]  # Even smaller for fast tests

    def test_load_test_data(self, evaluator):
        """Test test data loading"""
        X_test, y_test = evaluator.load_test_data()

        assert len(X_test) > 0
        assert len(y_test) > 0
        assert len(X_test) == len(y_test)
        assert X_test.shape[1] == 13  # 13 features

    def test_calculate_metrics(self, evaluator, sample_data):
        """Test metrics calculation"""
        X_test, y_test = sample_data

        # Create dummy predictions
        y_pred = np.random.choice([0, 1, 2], size=len(y_test))
        y_prob = np.random.random((len(y_test), 3))
        # Normalize probabilities
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        metrics = evaluator.calculate_metrics(y_test, y_pred, y_prob)

        assert 'accuracy' in metrics
        assert 'f1_weighted' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'roc_auc_macro' in metrics

        # Check bounds
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_weighted'] <= 1

    def test_create_classification_report(self, evaluator, sample_data):
        """Test classification report generation"""
        X_test, y_test = sample_data
        y_pred = np.random.choice([0, 1, 2], size=len(y_test))

        report_dict, report_str = evaluator.create_classification_report(y_test, y_pred)

        assert isinstance(report_dict, dict)
        assert isinstance(report_str, str)
        assert 'accuracy' in report_dict
        assert len(report_str) > 0


class TestIntegration:
    """Integration tests"""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from training to prediction"""
        # Create a simple model for testing
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        X = df
        y = data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Save model temporarily
        os.makedirs("temp_models", exist_ok=True)
        joblib.dump(model, "temp_models/test_model.pkl")

        # Test prediction
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1, 2] for p in predictions)

        # Evaluate
        accuracy = (predictions == y_test).mean()
        assert 0 <= accuracy <= 1

        # Cleanup
        if os.path.exists("temp_models/test_model.pkl"):
            os.remove("temp_models/test_model.pkl")
        if os.path.exists("temp_models"):
            os.rmdir("temp_models")

    def test_model_serialization(self):
        """Test model saving and loading"""
        # Train simple model
        data = load_wine()
        X = data.data[:50]  # Small subset for speed
        y = data.target[:50]

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Save and load
        os.makedirs("temp_models", exist_ok=True)
        joblib.dump(model, "temp_models/serialization_test.pkl")
        loaded_model = joblib.load("temp_models/serialization_test.pkl")

        # Test predictions are same
        original_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])

        assert np.array_equal(original_pred, loaded_pred)

        # Cleanup
        os.remove("temp_models/serialization_test.pkl")
        os.rmdir("temp_models")


# Pytest configuration
def pytest_configure():
    """Configure pytest"""
    pytest.test_data_loaded = False


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)

    yield

    # Cleanup if needed
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])