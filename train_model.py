import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "Telco-Customer-Churn.csv"
MODEL_FILE = "churn_model.pkl"
MLFLOW_DB = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Telecom Churn Advisor"


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_and_preprocess_data(filepath=DATA_FILE):
    df = pd.read_csv(filepath)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(" ", np.nan),
        errors="coerce",
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert target
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0,
    })

    return df


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(model):
    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    df = load_and_preprocess_data()
    X = df.drop(columns=["customerID", "Churn"])

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

    clf = Pipeline(
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

    return clf


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
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
        "roc_auc": roc_auc_score(
            y_test,
            y_prob,
        ),
    }

    return metrics


# ============================================================
# TRAINING
# ============================================================

def train_model():

    print("=" * 60)
    print("TELECOM CHURN ADVISOR - MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\n[1/5] Loading data...")

    df = load_and_preprocess_data()

    X = df.drop(
        columns=["customerID", "Churn"]
    )

    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Total data : {len(df)}")
    print(f"Train data : {len(X_train)}")
    print(f"Test data  : {len(X_test)}")

    # --------------------------------------------------------
    # MLflow setup
    # --------------------------------------------------------

    print("\n[2/5] Starting MLflow...")

    mlflow.set_tracking_uri(MLFLOW_DB)

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # Define experiments
    # --------------------------------------------------------

    experiments = [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            "params": {
                "model_type": "LogisticRegression",
                "max_iter": 1000,
                "class_weight": "balanced",
            },
        },
        {
            "name": "Random Forest 100",
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                max_depth=None,
            ),
            "params": {
                "model_type": "RandomForest",
                "n_estimators": 100,
                "max_depth": "None",
                "class_weight": "balanced",
            },
        },
        {
            "name": "Random Forest 200",
            "model": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                max_depth=None,
            ),
            "params": {
                "model_type": "RandomForest",
                "n_estimators": 200,
                "max_depth": "None",
                "class_weight": "balanced",
            },
        },
    ]

    results = []

    # --------------------------------------------------------
    # Run multiple experiments
    # --------------------------------------------------------

    print("\n[3/5] Running experiments...")

    for experiment in experiments:

        print(
            f"\nTraining: {experiment['name']}"
        )

        with mlflow.start_run(
            run_name=experiment["name"]
        ) as run:

            clf = build_pipeline(
                experiment["model"]
            )

            clf.fit(
                X_train,
                y_train,
            )

            metrics = evaluate_model(
                clf,
                X_test,
                y_test,
            )

            # Log parameters
            mlflow.log_params(
                experiment["params"]
            )

            mlflow.log_param(
                "test_size",
                0.20,
            )

            mlflow.log_param(
                "random_state",
                42,
            )

            # Log metrics
            mlflow.log_metrics(
                metrics
            )

            # Log model artifact
            mlflow.sklearn.log_model(
                clf,
                "model",
            )

            results.append(
                {
                    "run_id": run.info.run_id,
                    "name": experiment["name"],
                    "model": clf,
                    **metrics,
                }
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
                f"ROC AUC  : {metrics['roc_auc']:.4f}"
            )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    print("\n[4/5] Selecting best model...")

    best_result = max(
        results,
        key=lambda x: x["f1_score"],
    )

    print(
        f"\nBEST MODEL: {best_result['name']}"
    )

    print(
        f"Best F1 Score: "
        f"{best_result['f1_score']:.4f}"
    )

    # --------------------------------------------------------
    # Register best model
    # --------------------------------------------------------

    print("\n[5/5] Registering best model...")

    model_name = "TelecomChurnAdvisorBestModel"

    model_uri = (
        f"runs:/{best_result['run_id']}/model"
    )

    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        print(
            f"Registered model: {model_name}"
        )

        print(
            f"Version: "
            f"{registered_model.version}"
        )

    except Exception as e:

        print(
            "Model registration warning:"
        )

        print(e)

    # --------------------------------------------------------
    # Save best model for Streamlit
    # --------------------------------------------------------

    with open(
        MODEL_FILE,
        "wb",
    ) as f:

        pickle.dump(
            best_result["model"],
            f,
        )

    print(
        f"\nBest model saved to "
        f"{MODEL_FILE}"
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    train_model()