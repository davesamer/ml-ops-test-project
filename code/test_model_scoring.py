import joblib
import mlflow
from mlflow.tracking import MlflowClient
from utils import get_run_id_from_model_alias


if __name__ == "__main__":

    remote_server_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(remote_server_uri)

    mlflow_client = MlflowClient(tracking_uri=remote_server_uri)

    model_name = "churn-prediction-model"
    model_alias = "Challenger"

    run_id = get_run_id_from_model_alias(mlflow_client, model_name, model_alias)
    print(run_id)
    # TODO somehow the correct run ID is not found
    run_id = "aa54887ec8164c07a3b8a7d9e8d52054"

    if run_id:
        path_to_preprocessor = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessor.pkl", dst_path="pipeline")
        print("Artifact downloaded successfully.")
        preprocessor = joblib.load(path_to_preprocessor)
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")
    else:
        print("No run ID found.")




