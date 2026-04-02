#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "quality"
QUALITY_THRESHOLD = 7
SELECTED_FEATURE_COLUMNS: List[str] = [
    "alcohol",
    "density",
    "volatile acidity",
    "chlorides",
    "residual sugar",
]
SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a plain RandomForestClassifier and save it as model.pkl")
    parser.add_argument("--data_path", default=None, help="Optional path to the dataset file. If omitted, a file picker will open.")
    parser.add_argument("--model_output", default="model.pkl", help="Output path for the plain saved model. Default: model.pkl")
    parser.add_argument("--test_size", type=float, default=0.20, help="Proportion of the dataset used for testing. Default: 0.20")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility. Default: 42")
    return parser.parse_args()


def choose_dataset_file() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select the wine dataset file",
            filetypes=[
                ("Dataset files", "*.csv *.xlsx *.xls"),
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()

        if not file_path:
            raise ValueError("No file was selected.")
        return file_path

    except Exception as exc:
        raise RuntimeError(
            "Could not open the file picker. Run this script on your local machine, or pass the path manually with --data_path."
        ) from exc


def resolve_dataset_path(data_path: str | None) -> str:
    return data_path if data_path else choose_dataset_file()


def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{extension}'. Supported formats are: {sorted(SUPPORTED_EXTENSIONS)}")

    if extension == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if df.empty:
        raise ValueError("The dataset file was loaded, but it is empty.")
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")

    missing_feature_columns = [col for col in SELECTED_FEATURE_COLUMNS if col not in df.columns]
    if missing_feature_columns:
        raise ValueError(f"Missing required feature columns: {missing_feature_columns}")


def train_and_save_model(
    data_path: str | None = None,
    model_output: str = "model.pkl",
    test_size: float = 0.20,
    random_state: int = 42,
):
    resolved_path = resolve_dataset_path(data_path)

    print("Loading dataset...")
    df = load_dataset(resolved_path)
    validate_dataset(df)

    print(f"Dataset loaded successfully: {resolved_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    X = df[SELECTED_FEATURE_COLUMNS].copy()
    y = (df[TARGET_COLUMN] >= QUALITY_THRESHOLD).astype(int)

    print("\nSelected input features:")
    print(SELECTED_FEATURE_COLUMNS)
    print(f"Target conversion: 1 if quality >= {QUALITY_THRESHOLD}, else 0")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\nTrain/test split completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=300,
        criterion="gini",
        max_depth=10,
        random_state=random_state,
    )

    print("\nTraining RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("Model training completed.")

    y_pred = model.predict(X_test)

    print("\nEvaluation results:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, model_output)
    print(f"\nSaved plain model object to: {model_output}")

    print("\nIMPORTANT FOR PREDICTION")
    print("Use these exact 5 features in this exact order:")
    print(SELECTED_FEATURE_COLUMNS)

    return model


def main() -> None:
    args = parse_args()
    try:
        train_and_save_model(
            data_path=args.data_path,
            model_output=args.model_output,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
