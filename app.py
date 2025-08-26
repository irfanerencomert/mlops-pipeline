from flask import Flask, request, jsonify
import mlflow
import os
import logging
import sys
from sklearn.datasets import load_wine
import joblib
import numpy as np
import pandas as pd
from prometheus_client1 import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime
import traceback

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.utils import load_production_model, get_model_metadata
except ImportError:
    # Fallback for direct imports
    from src.utils import load_production_model, get_model_metadata

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_request_count', 'API request count', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
PREDICTION_COUNT = Counter('model_prediction_count', 'Model prediction count', ['predicted_class'])
MODEL_ERRORS = Counter('model_error_count', 'Model error count', ['error_type'])

# MLflow connection setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Global model cache
model_cache = {
    'model': None,
    'metadata': None,
    'last_loaded': None
}


def load_model():
    """Load model with caching and fallback logic"""
    try:
        # Check if we need to reload (every 5 minutes)
        if (model_cache['last_loaded'] is None or
                (datetime.now() - model_cache['last_loaded']).seconds > 300):

            logger.info("Loading/reloading model...")

            # Try to load from MLflow registry first
            model = load_production_model()

            if model is not None:
                model_cache['model'] = model
                model_cache['metadata'] = get_model_metadata()
                model_cache['last_loaded'] = datetime.now()
                logger.info("Model loaded successfully from registry")
            else:
                # Fallback to local model
                try:
                    local_path = 'models/best_model.pkl'
                    if os.path.exists(local_path):
                        model_cache['model'] = joblib.load(local_path)
                        model_cache['metadata'] = {'source': 'local_fallback', 'version': 'unknown'}
                        model_cache['last_loaded'] = datetime.now()
                        logger.warning("Loaded local fallback model")
                    else:
                        logger.error("No model available - neither registry nor local")
                        return None
                except Exception as e:
                    logger.error(f"Failed to load local model: {e}")
                    return None

        return model_cache['model']

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        MODEL_ERRORS.labels(error_type='model_loading').inc()
        return None


def validate_features(features):
    """Validate input features"""
    if not isinstance(features, list):
        raise ValueError("Features must be a list")

    if len(features) != 13:
        raise ValueError(f"Expected 13 features, got {len(features)}")

    # Convert to float and validate
    try:
        features = [float(f) for f in features]
    except (ValueError, TypeError):
        raise ValueError("All features must be numeric")

    return features


@app.before_request
def before_request():
    """Log request start time"""
    request.start_time = datetime.now()


@app.after_request
def after_request(response):
    """Log request metrics"""
    if hasattr(request, 'start_time'):
        duration = (datetime.now() - request.start_time).total_seconds()
        REQUEST_DURATION.observe(duration)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown',
        status=response.status_code
    ).inc()

    return response


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Load model
        model = load_model()
        if model is None:
            MODEL_ERRORS.labels(error_type='model_unavailable').inc()
            return jsonify({"error": "Model not available"}), 503

        # Get request data
        if not request.json or 'features' not in request.json:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = request.json['features']

        # Validate features
        try:
            features = validate_features(features)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Make prediction
        try:
            # Handle both MLflow and sklearn models
            if hasattr(model, 'predict'):
                if 'mlflow' in str(type(model)).lower():
                    # MLflow model expects DataFrame
                    feature_names = [
                        'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                        'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                        'proanthocyanins', 'color_intensity', 'hue',
                        'od280/od315_of_diluted_wines', 'proline'
                    ]
                    df = pd.DataFrame([features], columns=feature_names)
                    prediction = model.predict(df)

                    # DEBUG: Tahmin sonucunu logla
                    print(f"Raw prediction output: {prediction}")
                    print(f"Prediction type: {type(prediction)}")
                    print(f"First prediction value: {prediction[0]}")

                    # Format response
                    predicted_class = int(prediction[0])
                    print(f"Predicted class (int): {predicted_class}")

                    # Try to get probabilities if available
                    probabilities = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            probabilities = model.predict_proba(df)
                        except:
                            pass
                else:
                    # Sklearn model expects array
                    prediction = model.predict([features])

                    # Try to get probabilities
                    probabilities = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            probabilities = model.predict_proba([features])
                        except:
                            pass
            else:
                MODEL_ERRORS.labels(error_type='model_interface').inc()
                return jsonify({"error": "Invalid model interface"}), 500

            # Format response
            predicted_class = int(prediction[0])
            class_names = {0: 'Class_0', 1: 'Class_1', 2: 'Class_2'}

            result = {
                "prediction": predicted_class,
                "predicted_label": class_names.get(predicted_class, f"Unknown_{predicted_class}"),
                "model_version": model_cache['metadata'].get('version', 'unknown') if model_cache[
                    'metadata'] else 'unknown',
                "model_source": model_cache['metadata'].get('source', 'unknown') if model_cache[
                    'metadata'] else 'unknown',
                "timestamp": datetime.now().isoformat()
            }

            # Add probabilities if available
            if probabilities is not None:
                prob_dict = {}
                for i, prob in enumerate(probabilities[0]):
                    prob_dict[class_names.get(i, f"Class_{i}")] = float(prob)
                result["probabilities"] = prob_dict
                result["confidence"] = float(max(probabilities[0]))

            # Update metrics
            PREDICTION_COUNT.labels(predicted_class=str(predicted_class)).inc()

            return jsonify(result)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.error(traceback.format_exc())
            MODEL_ERRORS.labels(error_type='prediction_error').inc()
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Request processing error: {e}")
        MODEL_ERRORS.labels(error_type='request_processing').inc()
        return jsonify({"error": "Internal server error"}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        model = load_model()
        if model is None:
            return jsonify({"error": "Model not available"}), 503

        if not request.json or 'features_list' not in request.json:
            return jsonify({"error": "Missing 'features_list' in request"}), 400

        features_list = request.json['features_list']

        if not isinstance(features_list, list):
            return jsonify({"error": "'features_list' must be a list"}), 400

        results = []
        for i, features in enumerate(features_list):
            try:
                features = validate_features(features)

                # Make prediction (similar logic as single predict)
                if 'mlflow' in str(type(model)).lower():
                    feature_names = [
                        'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                        'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                        'proanthocyanins', 'color_intensity', 'hue',
                        'od280/od315_of_diluted_wines', 'proline'
                    ]
                    df = pd.DataFrame([features], columns=feature_names)
                    prediction = model.predict(df)
                else:
                    prediction = model.predict([features])

                predicted_class = int(prediction[0])
                class_names = {0: 'Class_0', 1: 'Class_1', 2: 'Class_2'}

                results.append({
                    "sample_id": i,
                    "prediction": predicted_class,
                    "predicted_label": class_names.get(predicted_class, f"Unknown_{predicted_class}")
                })

                PREDICTION_COUNT.labels(predicted_class=str(predicted_class)).inc()

            except Exception as e:
                results.append({
                    "sample_id": i,
                    "error": str(e)
                })
                MODEL_ERRORS.labels(error_type='batch_prediction_error').inc()

        return jsonify({
            "results": results,
            "processed_count": len([r for r in results if 'error' not in r]),
            "error_count": len([r for r in results if 'error' in r]),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": "Batch prediction failed"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        model = load_model()
        model_status = "healthy" if model is not None else "unhealthy"

        health_info = {
            "status": model_status,
            "model_available": model is not None,
            "timestamp": datetime.now().isoformat(),
            "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        }

        # Add model metadata if available
        if model_cache['metadata']:
            health_info["model_info"] = model_cache['metadata']

        status_code = 200 if model is not None else 503
        return jsonify(health_info), status_code

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint"""
    try:
        model = load_model()

        info_data = {
            "model_loaded": model is not None,
            "model_type": str(type(model)) if model else None,
            "expected_features": 13,
            "feature_names": [
                'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                'proanthocyanins', 'color_intensity', 'hue',
                'od280/od315_of_diluted_wines', 'proline'
            ],
            "class_names": {0: 'Class_0', 1: 'Class_1', 2: 'Class_2'},
            "timestamp": datetime.now().isoformat()
        }

        # Add model metadata
        if model_cache['metadata']:
            info_data["model_metadata"] = model_cache['metadata']

        # Add model parameters if available
        if model and hasattr(model, 'get_params'):
            try:
                info_data["model_parameters"] = model.get_params()
            except:
                pass

        return jsonify(info_data)

    except Exception as e:
        logger.error(f"Info endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Force model reload"""
    try:
        # Clear cache to force reload
        model_cache['last_loaded'] = None
        model_cache['model'] = None
        model_cache['metadata'] = None

        # Load fresh model
        model = load_model()

        if model is not None:
            return jsonify({
                "status": "success",
                "message": "Model reloaded successfully",
                "timestamp": datetime.now().isoformat(),
                "model_info": model_cache['metadata']
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to reload model",
                "timestamp": datetime.now().isoformat()
            }), 503

    except Exception as e:
        logger.error(f"Model reload error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Wine Quality Prediction API...")
    logger.info(f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")

    # Try to load model on startup
    initial_model = load_model()
    if initial_model is not None:
        logger.info("Model loaded successfully on startup")
    else:
        logger.warning("Model not available on startup - will try to load on first request")

    # Start the Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true"
    )