import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run

def get_model_version_from_run_id(client: MlflowClient, model_name: str, run_id: str) -> str:
    '''get model version from run id'''
    mv = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")[0]
    return mv.version

def get_latest_run(client: MlflowClient, experiment_name: str) -> Run:
    '''get id of latest mlflow run of experiment'''
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    latest_run = runs[0]
    return latest_run

def get_run_id_from_model_alias(client: MlflowClient, model_name: str, model_alias: str) -> str:
    '''Get the run ID associated with a specific model alias'''
    model_versions = client.search_model_versions(f"name='{model_name}'")
    for version in model_versions:
        detailed_version = client.get_model_version(name=model_name, version=version.version)        
        if model_alias in detailed_version.aliases:
            return detailed_version.run_id

def load_data():
    '''load train and eval data'''
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").churn.values
    y_test = pd.read_csv("data/y_test.csv").churn.values
    return X_train, X_test, y_train, y_test