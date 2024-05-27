import os
from dotenv import load_dotenv
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
from mlflow.tracking import MlflowClient
from utils import get_run_id_from_model_alias

load_dotenv("config.env")
MLFLOW_ADDRESS = os.environ.get("MLFLOW_ADDRESS")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")


if __name__ == "__main__":

    mlflow.set_tracking_uri(MLFLOW_ADDRESS)
    mlflow_client = MlflowClient(tracking_uri=MLFLOW_ADDRESS)

    # TODO update to Champion
    model_alias = "Challenger"
    run_id = get_run_id_from_model_alias(mlflow_client, MLFLOW_MODEL_NAME, model_alias)

    if run_id:
        path_to_preprocessor = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessor.pkl", dst_path="pipeline")
        print("Artifact downloaded successfully.")
        preprocessor = joblib.load(path_to_preprocessor)
        model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}@{model_alias}")
    else:
        print("No run ID found.")


    data = pd.read_csv("data/X_test.csv")
    labels = pd.read_csv("data/y_test.csv")

    data_trans = preprocessor.transform(data)
    y_pred = model.predict(data_trans)

    print(accuracy_score(labels.churn, y_pred))
    print(precision_score(labels.churn, y_pred))
    print(recall_score(labels.churn, y_pred))
