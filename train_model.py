#!/usr/bin/env python3
"""
train_model.py

Decision Tree training script for the winequalityN dataset.

What is different in this version:
- If you do not pass --data_path, a file picker opens so you can choose the dataset from your device.
- The same code can be used inside Jupyter Notebook by importing train_and_save_model().
- Supports CSV and Excel files (.csv, .xls, .xlsx).

Command line example:
    python train_model.py
    python train_model.py --data_path winequalityN.csv

Jupyter Notebook example:
    from train_model import train_and_save_model
    result = train_and_save_model()   # opens file picker

Or:
    from train_model import train_and_save_model
    result = train_and_save_model(data_path="winequalityN.csv")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "quality"
EXPECTED_FEATURE_COLUMNS = [
    "type",
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Decision Tree model for wine quality prediction.")
    parser.add_argument(
        "--data_path",
        default=None,
        help="Optional path to the dataset file. If omitted, a file picker will open.",
    )
    parser.add_argument(
        "--model_output",
        default="wine_quality_pipeline.pkl",
        help="Path to save the trained pipeline bundle (.pkl). Default: wine_quality_pipeline.pkl",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Proportion of the dataset used for testing. Default: 0.20",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Optional Decision Tree max_depth. Leave empty for no depth limit.",
    )
    return parser.parse_args()


def choose_dataset_file() -> str:
    """Open a file picker so the user can choose a CSV or Excel file."""
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
            "Could not open the file picker. This can happen if tkinter is not available or the environment does not support a desktop window. "
            "In that case, pass the path directly using --data_path or train_and_save_model(data_path='...')."
        ) from exc


def resolve_dataset_path(data_path: str | None) -> str:
    if data_path:
        return data_path
    return choose_dataset_file()


def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{extension}'. Supported formats are: {sorted(SUPPORTED_EXTENSIONS)}"
        )

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

    missing_feature_columns = [col for col in EXPECTED_FEATURE_COLUMNS if col not in df.columns]
    if missing_feature_columns:
        raise ValueError(
            "The dataset does not match the expected winequalityN structure.\n"
            f"Missing feature columns: {missing_feature_columns}"
        )


def get_feature_matrix_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[EXPECTED_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    if y.isna().any():
        raise ValueError("The target column contains missing values. Please clean the target before training.")

    return X, y


def build_pipeline(feature_frame: pd.DataFrame, random_state: int, max_depth: int | None) -> Pipeline:
    categorical_columns = feature_frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [col for col in feature_frame.columns if col not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    model = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def print_missing_value_summary(df: pd.DataFrame) -> None:
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    print("\nMissing value summary:")
    if missing_counts.empty:
        print("No missing values found.")
    else:
        print(missing_counts.to_string())


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_and_save_model(
    data_path: str | None = None,
    model_output: str = "wine_quality_pipeline.pkl",
    test_size: float = 0.20,
    random_state: int = 42,
    max_depth: int | None = None,
) -> Dict[str, Any]:
    """
    Train the Decision Tree pipeline and save it as a single .pkl file.

    This function is notebook-friendly and can be imported directly into Jupyter.
    If data_path is None, a file picker opens.
    """
    resolved_path = resolve_dataset_path(data_path)

    print("Loading dataset...")
    df = load_dataset(resolved_path)

    print(f"Dataset loaded successfully: {resolved_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    validate_dataset(df)
    print_missing_value_summary(df)

    X, y = get_feature_matrix_and_target(df)

    print("\nTarget class distribution:")
    print(y.value_counts().sort_index().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\nTrain/test split completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    pipeline = build_pipeline(X_train, random_state=random_state, max_depth=max_depth)

    print("\nTraining Decision Tree model...")
    pipeline.fit(X_train, y_train)

    print("Model training completed.")

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    print("\nEvaluation results:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    bundle = {
        "pipeline": pipeline,
        "feature_columns": EXPECTED_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "task_type": "multiclass_classification",
        "model_name": "DecisionTreeClassifier",
        "metrics": metrics,
        "source_file": resolved_path,
    }

    joblib.dump(bundle, model_output)
    print(f"\nSaved trained pipeline bundle to: {model_output}")

    return {
        "bundle": bundle,
        "metrics": metrics,
        "model_output": model_output,
        "data_path": resolved_path,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def main() -> None:
    args = parse_args()

    try:
        train_and_save_model(
            data_path=args.data_path,
            model_output=args.model_output,
            test_size=args.test_size,
            random_state=args.random_state,
            max_depth=args.max_depth,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
