import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow
import mlflow.sklearn


if __name__ == "__main__":

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")

    categorical_features = ['country', 'gender']
    numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']

    categorical_transformer = Pipeline(steps=[
        ('label_encoder', OneHotEncoder())
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing steps into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])

    # Model training pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate model
    train_accuracy = model_pipeline.score(X_train, y_train)
    test_accuracy = model_pipeline.score(X_test, y_test)

    # Log parameters and metrics with MLflow
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model_pipeline, "random_forest_model")
        
        # Add tag to the run
        mlflow.set_tag("validation_status", "not validated")

