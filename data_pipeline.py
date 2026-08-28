from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "telco_clean.csv"


def ingest_data():
    """Load raw customer data."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset tidak ditemukan: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    print(f"[INGESTION] Loaded {len(df)} raw records")

    return df


def clean_data(df):
    """Clean and validate the raw dataset."""

    df = df.copy()

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(" ", pd.NA),
        errors="coerce",
    )

    # Fill missing TotalCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Clean target
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0,
    })

    # Remove rows with invalid target
    df = df.dropna(subset=["Churn"])

    # Convert target to integer
    df["Churn"] = df["Churn"].astype(int)

    # Remove duplicate customers
    if "customerID" in df.columns:
        df = df.drop_duplicates(
            subset=["customerID"]
        )

    print(
        f"[CLEANING] Clean records: {len(df)}"
    )

    return df


def validate_data(df):
    """Validate the cleaned dataset."""

    required_columns = {
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "Clean dataset is empty."
        )

    if df["Churn"].isna().any():
        raise ValueError(
            "Churn contains missing values."
        )

    print("[VALIDATION] Dataset validation passed")


def save_data(df):
    """Save cleaned data to usable storage."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_FILE,
        index=False,
    )

    print(
        f"[STORAGE] Saved processed dataset to: "
        f"{PROCESSED_FILE}"
    )


def run_pipeline():
    print("=" * 60)
    print("TELECOM CHURN DATA PIPELINE")
    print("=" * 60)

    df = ingest_data()
    df = clean_data(df)
    validate_data(df)
    save_data(df)

    print("=" * 60)
    print("DATA PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()