import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


DATA_FILE = "Telco-Customer-Churn.csv"
MODEL_FILE = "churn_model.pkl"

EXPERIMENT_NAME = "Telecom Churn Model Selection"
REGISTERED_MODEL_NAME = "TelecomChurnAdvisorBestModel"


def load_and_preprocess_data(filepath=DATA_FILE):
    print("Loading data...")

    df = pd.read_csv(filepath)

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(" ", np.nan),
        errors="coerce",
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert target
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0,
    })

    return df


def build_preprocessor(X):
    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    cat_cols = [
        col for col in X.columns
        if col not in num_cols
    ]

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                num_cols,
            ),
            (
                "cat",
                categorical_transformer,
                cat_cols,
            ),
        ]
    )

    return preprocessor


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(
            y_test,
            y_prob,
        )
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def train_model():

    df = load_and_preprocess_data()

    X = df.drop(
        columns=["customerID", "Churn"]
    )

    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest 100": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        ),

        "Random Forest 200": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }

    # MLflow configuration
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    best_model = None
    best_metrics = None
    best_model_name = None

    print("\nStarting MLflow experiments...\n")

    for model_name, model in models.items():

        print(
            f"Training {model_name}..."
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    model,
                ),
            ]
        )

        with mlflow.start_run(
            run_name=model_name
        ):

            pipeline.fit(
                X_train,
                y_train,
            )

            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test,
            )

            # Log common parameters
            mlflow.log_param(
                "model",
                model_name,
            )

            mlflow.log_param(
                "test_size",
                0.2,
            )

            mlflow.log_param(
                "random_state",
                42,
            )

            mlflow.log_param(
                "class_weight",
                "balanced",
            )

            # Log model-specific parameters
            if model_name.startswith(
                "Random Forest"
            ):
                mlflow.log_param(
                    "n_estimators",
                    model.n_estimators,
                )

            if model_name == "Logistic Regression":
                mlflow.log_param(
                    "max_iter",
                    model.max_iter,
                )

            # Log metrics
            mlflow.log_metric(
                "accuracy",
                metrics["accuracy"],
            )

            mlflow.log_metric(
                "precision",
                metrics["precision"],
            )

            mlflow.log_metric(
                "recall",
                metrics["recall"],
            )

            mlflow.log_metric(
                "f1_score",
                metrics["f1_score"],
            )

            mlflow.log_metric(
                "roc_auc",
                metrics["roc_auc"],
            )

            # Log model artifact
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
            )

            print(
                f"Accuracy : {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {metrics['precision']:.4f}"
            )

            print(
                f"Recall   : {metrics['recall']:.4f}"
            )

            print(
                f"F1 Score : {metrics['f1_score']:.4f}"
            )

            print(
                f"ROC-AUC  : {metrics['roc_auc']:.4f}"
            )

            print()

            # Select best model based on F1
            if (
                best_metrics is None
                or metrics["f1_score"]
                > best_metrics["f1_score"]
            ):
                best_model = pipeline
                best_metrics = metrics
                best_model_name = model_name

    # Save best model locally
    print(
        f"Best model: {best_model_name}"
    )

    with open(
        MODEL_FILE,
        "wb",
    ) as f:
        pickle.dump(
            best_model,
            f,
        )

    print(
        f"Best model saved to {MODEL_FILE}"
    )

    # Register best model in MLflow
    with mlflow.start_run(
        run_name="Best Model Registration"
    ) as run:

        mlflow.log_param(
            "selected_model",
            best_model_name,
        )

        mlflow.log_param(
            "selection_metric",
            "f1_score",
        )

        mlflow.log_metric(
            "accuracy",
            best_metrics["accuracy"],
        )

        mlflow.log_metric(
            "precision",
            best_metrics["precision"],
        )

        mlflow.log_metric(
            "recall",
            best_metrics["recall"],
        )

        mlflow.log_metric(
            "f1_score",
            best_metrics["f1_score"],
        )

        mlflow.log_metric(
            "roc_auc",
            best_metrics["roc_auc"],
        )

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    print(
        "\n=================================="
    )

    print(
        "MLFLOW TRAINING COMPLETED"
    )

    print(
        f"Best Model : {best_model_name}"
    )

    print(
        f"F1 Score   : {best_metrics['f1_score']:.4f}"
    )

    print(
        f"Registered : {REGISTERED_MODEL_NAME}"
    )

    print(
        "=================================="
    )


if __name__ == "__main__":
    train_model()