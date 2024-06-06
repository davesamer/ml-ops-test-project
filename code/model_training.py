import os
import tempfile
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.pyfunc
from utils import load_data

load_dotenv("config.env")
MLFLOW_ADDRESS = os.environ.get("MLFLOW_ADDRESS")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")


class ChurnPredictionModel(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, context, model_input):
        processed_data = self.preprocessor.transform(model_input)
        predictions = self.model.predict(processed_data)
        return predictions


def train_model(n_estimators: int, random_state: int):
    
    X_train, X_test, y_train, y_test = load_data()

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
    
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
    rf.fit(X_train_trans, y_train)

    y_train_pred = rf.predict(X_train_trans)
    y_test_pred = rf.predict(X_test_trans)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Test Accuracy:", test_accuracy)
    print("Precision", precision_score(y_test, y_test_pred))
    print("Recall", recall_score(y_test, y_test_pred))

    mlflow.set_tracking_uri(MLFLOW_ADDRESS)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("random_state_rf", random_state)
        mlflow.log_param("n_estimators", n_estimators)
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Save the custom model
        custom_model = ChurnPredictionModel(preprocessor, rf)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "custom_churn_model")
            mlflow.pyfunc.save_model(
                path=model_path,
                python_model=custom_model
            )
            mlflow.pyfunc.log_model(
                artifact_path=MLFLOW_MODEL_NAME,
                python_model=custom_model,
                registered_model_name=MLFLOW_MODEL_NAME
            )

    print("Custom model logged successfully.")



if __name__ == "__main__":
    n_estimators = 100
    random_state = 42
    train_model(n_estimators, random_state)



