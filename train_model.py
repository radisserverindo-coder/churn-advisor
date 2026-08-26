import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def load_and_preprocess_data(filepath="Telco-Customer-Churn.csv"):
    df = pd.read_csv(filepath)
    
    # Handle TotalCharges missing values (spaces to NaN to numeric)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    # Fill NaN with 0 for TotalCharges (assume 0 charges if just joined)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert Target to numeric (0 or 1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def train_model():
    print("Loading data...")
    df = load_and_preprocess_data()
    
    # Features and Target
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Identify numerical and categorical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in X.columns if col not in num_cols]
    
    print("Building pipeline...")
    # Preprocessing for numerical data
    numeric_transformer = StandardScaler()
    
    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    # Define the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Create and evaluate the pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    print("Saving model and preprocessor pipeline...")
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    print("Done! Model saved as churn_model.pkl")

if __name__ == "__main__":
    train_model()
