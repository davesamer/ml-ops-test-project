import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib


load_dotenv("config.env")
MLFLOW_ADDRESS = os.environ.get("MLFLOW_ADDRESS")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")


if __name__ == "__main__":

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").churn.values
    y_test = pd.read_csv("data/y_test.csv").churn.values

    categorical_features = ['country', 'gender']
    numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']

    categorical_transformer = Pipeline(steps=[
        ('ohe_encoder', OneHotEncoder())
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])

    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    n_estimators = 100
    random_state = 42
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
    rf.fit(X_train_trans, y_train)

    y_train_pred = rf.predict(X_train_trans)
    y_test_pred = rf.predict(X_test_trans)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(test_accuracy)

    mlflow.set_tracking_uri(MLFLOW_ADDRESS)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:

        mlflow.log_param("random_state_rf", n_estimators)
        mlflow.log_param("n_estimators", random_state)
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        with tempfile.TemporaryDirectory() as tmpdir:
            joblib_file = os.path.join(tmpdir, "preprocessor.pkl")
            joblib.dump(preprocessor, joblib_file)
            mlflow.log_artifact(joblib_file)
        
        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path=MLFLOW_MODEL_NAME,
            registered_model_name=MLFLOW_MODEL_NAME)


