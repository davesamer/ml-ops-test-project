from mlflow.tracking import MlflowClient


def test_against_baseline(test_accuracy: float, baseline_accuracy: float = 0.5):
    '''Check if model is better than the baseline'''
    if test_accuracy is None or test_accuracy <= baseline_accuracy:
        raise ValueError("Model validation failed: Accuracy is not better than the baseline of 50%")


def test_against_registered_models(client: object, model_name: str, test_accuracy: float, run_id: str):
    '''Check if model is better than other registered models (if any)'''

    versions = client.get_latest_versions(model_name)
    if versions:
        best_version = max(versions, key=lambda v: float(v.metadata.get("test_accuracy", 0)))
        best_accuracy = float(best_version.metadata.get("test_accuracy", 0))
        
        if test_accuracy > best_accuracy:
            client.set_tag(run_id, "Champion", "true")
            print("Model is better than existing registered models.")
            return True
        else:
            print("Model is not better than the existing champion model.")
            return False
    else:
        print("No existing registered models.")
        return True


def validate_model(client: object, experiment_name: str):

    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    latest_run = runs[0]
    run_id = latest_run.info.run_id

    test_accuracy = latest_run.data.metrics.get("test_accuracy")
    model_name = "churn_prediction_model"

    client.set_tag(run_id, "environment", "dev")

    if test_against_baseline(test_accuracy, 0.5):
        client.set_tag(run_id, "Validated", "true")

    if test_against_registered_models(client, model_name, test_accuracy, run_id):
        client.set_tag(run_id, "Champion", "true")


def validate_model():
    # set alias of model to challenger 
    # set tag model_validation to pending
    # sleep a few seconds
    # set tag model_validation to pass
    pass
    


if __name__ == "__main__":

    mlflow_client = MlflowClient(tracking_uri="http://localhost:5000")
    experiment_name = "/ml-ops-test-experiment"
    validate_model(mlflow_client, experiment_name)
