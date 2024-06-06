import os
from dotenv import load_dotenv
import mlflow

load_dotenv("config.env")
MLFLOW_ADDRESS = os.environ.get("MLFLOW_ADDRESS")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
MLFLOW_MODEL_DOCKER_IMAGE_NAME = os.environ.get("MLFLOW_MODEL_DOCKER_IMAGE_NAME")

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_ADDRESS)
    model_alias = "Challenger"
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{model_alias}"

    mlflow.models.build_docker(
        model_uri=model_uri,
        name=f"davesamer/{MLFLOW_MODEL_DOCKER_IMAGE_NAME}",
        enable_mlserver=False,
    )
