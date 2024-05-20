from mlflow.tracking import MlflowClient
from utils import get_model_version_from_run_id, get_latest_run

    
def test_against_baseline(test_accuracy: float, baseline_accuracy: float = 0.8):
    '''Check if model is better than the baseline'''
    if test_accuracy is None or test_accuracy <= baseline_accuracy:
        raise ValueError("Model validation failed: Accuracy is not better than the baseline of 50%")


def validate_model(client: object, experiment_name: str, model_name: str):
    '''set alias of model to challenger and validation_status to passed if tests are passed. 
    Otherwise set validation status to failed. If tests fail and challenger alias has already been assigned, alias is deleted'''

    # TODO instead of validating only model from most recent run, validate all unvalidated models
    latest_run = get_latest_run(client, experiment_name)
    latest_run_id = latest_run.info.run_id
    test_accuracy = latest_run.data.metrics.get("test_accuracy")
    model_version = get_model_version_from_run_id(client, model_name, latest_run_id)

    try: 
        test_against_baseline(test_accuracy)
        client.set_model_version_tag(model_name, model_version, "validation_status", "passed")
        client.set_registered_model_alias(model_name, "Challenger", model_version)
    except ValueError:
        client.set_model_version_tag(model_name, model_version, "validation_status", "failed")
        client.delete_registered_model_alias(model_name, "Challenger")


if __name__ == "__main__":

    mlflow_client = MlflowClient(tracking_uri="http://localhost:5000")
    experiment_name = "/ml-ops-test-experiment"
    model_name = "churn-prediction-model"

    validate_model(mlflow_client, experiment_name, model_name)