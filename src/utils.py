import os
import mlflow
import joblib
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import logging

load_dotenv()

logger = logging.getLogger(__name__)


def get_model_client():
    """Get MLflow client instance"""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    return MlflowClient(tracking_uri=tracking_uri)


def load_production_model():
    """Load production model from MLflow registry or local fallback"""
    model_name = os.getenv("MODEL_NAME", "WineQualityModel")
    stage = os.getenv("PRODUCTION_STAGE", "Production")

    try:
        # Try to load from MLflow registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded production model from registry: {model_uri}")
        return model

    except Exception as e:
        logger.warning(f"Failed to load from registry: {e}")

        try:
            # Fallback: Try to get latest version
            client = get_model_client()
            latest_versions = client.get_latest_versions(model_name, stages=[stage])
            if latest_versions:
                model_uri = f"models:/{model_name}/{latest_versions[0].version}"
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Loaded latest model version: {model_uri}")
                return model
        except Exception as e:
            logger.warning(f"Failed to load latest version: {e}")

        try:
            # Final fallback: Load local model
            local_model_path = "models/best_model.pkl"
            if os.path.exists(local_model_path):
                model = joblib.load(local_model_path)
                logger.info(f"Loaded local model: {local_model_path}")
                return model
            else:
                logger.error(f"Local model not found: {local_model_path}")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")

        return None


def log_data_drift_report(reference, current, path="drift_report.html"):
    """Generate and save data drift report"""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        # Remove target column if exists for drift calculation
        ref_features = reference.drop('target', axis=1, errors='ignore')
        curr_features = current.drop('target', axis=1, errors='ignore')

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=curr_features)

        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        report.save_html(path)
        logger.info(f"Data drift report saved to: {path}")
        return path

    except Exception as e:
        logger.error(f"Failed to generate drift report: {e}")
        return None


def register_new_model_version(run_id, model_name=None, description=None):
    """Register new model version from MLflow run"""
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "WineQualityModel")

    try:
        client = get_model_client()

        # Register model from run
        model_uri = f"runs:/{run_id}/model"
        model_details = mlflow.register_model(model_uri, model_name)

        # Add description if provided
        if description:
            client.update_model_version(
                name=model_name,
                version=model_details.version,
                description=description
            )

        # Transition to staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_details.version,
            stage="Staging"
        )

        logger.info(f"Model registered: {model_name} v{model_details.version}")
        logger.info(f"Model transitioned to Staging")

        return model_details

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return None


def promote_model_to_production(model_name=None, version=None):
    """Promote model version to production"""
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "WineQualityModel")

    try:
        client = get_model_client()

        if version is None:
            # Get latest staging version
            staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
            if not staging_versions:
                logger.error("No staging version found to promote")
                return False
            version = staging_versions[0].version

        # First, demote current production model to archived
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            for prod_version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=prod_version.version,
                    stage="Archived"
                )
                logger.info(f"Previous production model v{prod_version.version} archived")
        except Exception as e:
            logger.warning(f"Failed to archive previous production model: {e}")

        # Promote staging to production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        logger.info(f"Model v{version} promoted to Production")
        return True

    except Exception as e:
        logger.error(f"Failed to promote model to production: {e}")
        return False


def get_model_metadata(model_name=None, stage="Production"):
    """Get model metadata from registry"""
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "WineQualityModel")

    try:
        client = get_model_client()
        versions = client.get_latest_versions(model_name, stages=[stage])

        if not versions:
            logger.warning(f"No {stage} version found for {model_name}")
            return None

        version = versions[0]

        metadata = {
            'name': model_name,
            'version': version.version,
            'stage': version.current_stage,
            'description': version.description,
            'creation_timestamp': version.creation_timestamp,
            'last_updated_timestamp': version.last_updated_timestamp,
            'run_id': version.run_id,
            'source': version.source,
            'status': version.status
        }

        # Get run details
        try:
            run = client.get_run(version.run_id)
            metadata['run_metrics'] = run.data.metrics
            metadata['run_params'] = run.data.params
            metadata['run_tags'] = run.data.tags
        except Exception as e:
            logger.warning(f"Failed to get run details: {e}")

        return metadata

    except Exception as e:
        logger.error(f"Failed to get model metadata: {e}")
        return None


def list_model_versions(model_name=None, max_results=10):
    """List all versions of a model"""
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "WineQualityModel")

    try:
        client = get_model_client()

        # Get all versions
        model_versions = client.search_model_versions(f"name='{model_name}'")

        # Sort by version number (descending)
        model_versions = sorted(
            model_versions,
            key=lambda x: int(x.version),
            reverse=True
        )[:max_results]

        versions_info = []
        for version in model_versions:
            version_info = {
                'version': version.version,
                'stage': version.current_stage,
                'description': version.description,
                'creation_timestamp': version.creation_timestamp,
                'run_id': version.run_id,
                'status': version.status
            }
            versions_info.append(version_info)

        return versions_info

    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        return []


def cleanup_old_models(model_name=None, keep_latest_n=5):
    """Archive old model versions, keeping only the latest N"""
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "WineQualityModel")

    try:
        client = get_model_client()

        # Get all versions except Production and Staging
        all_versions = client.search_model_versions(f"name='{model_name}'")

        # Filter out Production and Staging
        archivable_versions = [
            v for v in all_versions
            if v.current_stage not in ["Production", "Staging"]
        ]

        # Sort by version number (descending)
        archivable_versions = sorted(
            archivable_versions,
            key=lambda x: int(x.version),
            reverse=True
        )

        # Keep latest N, archive the rest
        to_archive = archivable_versions[keep_latest_n:]

        archived_count = 0
        for version in to_archive:
            if version.current_stage != "Archived":
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
                archived_count += 1

        logger.info(f"Archived {archived_count} old model versions")
        return archived_count

    except Exception as e:
        logger.error(f"Failed to cleanup old models: {e}")
        return 0


def health_check():
    """Check health of MLflow connection and model availability"""
    health_status = {
        'mlflow_connection': False,
        'model_available': False,
        'model_metadata': None,
        'timestamp': None
    }

    try:
        # Test MLflow connection
        client = get_model_client()
        client.list_experiments(max_results=1)
        health_status['mlflow_connection'] = True
        logger.info("MLflow connection: OK")

        # Test model availability
        model = load_production_model()
        if model is not None:
            health_status['model_available'] = True
            health_status['model_metadata'] = get_model_metadata()
            logger.info("Production model: OK")
        else:
            logger.warning("Production model: NOT AVAILABLE")

    except Exception as e:
        logger.error(f"Health check failed: {e}")

    from datetime import datetime
    health_status['timestamp'] = datetime.now().isoformat()

    return health_status


if __name__ == "__main__":
    # CLI interface for utility functions
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utils.py <command> [args...]")
        print("Commands:")
        print("  health-check          - Check system health")
        print("  list-versions         - List model versions")
        print("  promote <version>     - Promote version to production")
        print("  cleanup               - Cleanup old models")
        print("  model-info            - Get production model info")
        sys.exit(1)

    command = sys.argv[1]

    if command == "health-check":
        status = health_check()
        print("System Health Check:")
        print(f"MLflow Connection: {'✓' if status['mlflow_connection'] else '✗'}")
        print(f"Model Available: {'✓' if status['model_available'] else '✗'}")

    elif command == "list-versions":
        versions = list_model_versions()
        print("Model Versions:")
        for v in versions:
            print(f"  v{v['version']} - {v['stage']} - {v['description'] or 'No description'}")

    elif command == "promote":
        if len(sys.argv) < 3:
            print("Usage: python utils.py promote <version>")
            sys.exit(1)
        version = sys.argv[2]
        success = promote_model_to_production(version=version)
        print(f"Promotion {'successful' if success else 'failed'}")

    elif command == "cleanup":
        count = cleanup_old_models()
        print(f"Archived {count} old model versions")

    elif command == "model-info":
        metadata = get_model_metadata()
        if metadata:
            print("Production Model Info:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("No production model found")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)