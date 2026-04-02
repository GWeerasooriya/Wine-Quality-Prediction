#!/usr/bin/env python3
"""
predict_quality.py

Loads the saved .pkl pipeline bundle and predicts wine quality for one sample.
This version can be used both from the command line and inside Jupyter Notebook.

Command line example:
    python predict_quality.py --model_path wine_quality_pipeline.pkl

Jupyter Notebook example:
    from predict_quality import predict_quality
    sample = {
  "type": "white",
  "fixed acidity": 6.6,
  "volatile acidity": 0.16,
  "citric acid": 0.4,
  "residual sugar": 1.5,
  "chlorides": 0.044,
  "free sulfur dioxide": 48.0,
  "total sulfur dioxide": 143.0,
  "density": 0.9912,
  "pH": 3.54,
  "sulphates": 0.52,
  "alcohol": 12.4
}
    result = predict_quality(sample=sample, model_path="wine_quality_pipeline.pkl")
    print(result)

Expected JSON input structure:
{
  "type": "white",
  "fixed acidity": 7.0,
  "volatile acidity": 0.27,
  "citric acid": 0.36,
  "residual sugar": 20.7,
  "chlorides": 0.045,
  "free sulfur dioxide": 45.0,
  "total sulfur dioxide": 170.0,
  "density": 1.0010,
  "pH": 3.00,
  "sulphates": 0.45,
  "alcohol": 8.8
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

import joblib
import pandas as pd


SAMPLE_INPUT = {
  "type": "white",
  "fixed acidity": 6.6,
  "volatile acidity": 0.16,
  "citric acid": 0.4,
  "residual sugar": 1.5,
  "chlorides": 0.044,
  "free sulfur dioxide": 48.0,
  "total sulfur dioxide": 143.0,
  "density": 0.9912,
  "pH": 3.54,
  "sulphates": 0.52,
  "alcohol": 12.4
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict wine quality using a saved Decision Tree pipeline.")
    parser.add_argument(
        "--model_path",
        default="wine_quality_pipeline.pkl",
        help="Path to the saved .pkl pipeline bundle. Default: wine_quality_pipeline.pkl",
    )
    parser.add_argument(
        "--input_json",
        default=None,
        help="A JSON string containing one sample to predict.",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="Path to a JSON file containing one sample to predict.",
    )
    return parser.parse_args()


def load_model_bundle(model_path: str) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("The loaded model file is not in the expected bundle format.")

    required_keys = {"pipeline", "feature_columns", "task_type"}
    missing_keys = [key for key in required_keys if key not in bundle]
    if missing_keys:
        raise ValueError(f"The model bundle is missing required keys: {missing_keys}")

    return bundle


def load_input_data(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input_json:
        try:
            payload = json.loads(args.input_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON passed in --input_json: {exc}") from exc
    elif args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input JSON file not found: {args.input_file}")

        with open(args.input_file, "r", encoding="utf-8") as file:
            try:
                payload = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in input file: {exc}") from exc
    else:
        print("No input JSON was provided. Using SAMPLE_INPUT for manual testing.\n")
        payload = SAMPLE_INPUT

    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("This script expects exactly one sample. Provide a single JSON object or a list with one object.")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError("Input data must be a JSON object with feature names as keys.")

    return payload


def validate_and_prepare_input(sample: Dict[str, Any], expected_features: list[str]) -> Tuple[pd.DataFrame, list[str]]:
    missing_features = [feature for feature in expected_features if feature not in sample]
    if missing_features:
        raise ValueError(f"The input sample is missing required fields: {missing_features}")

    extra_fields = [field for field in sample if field not in expected_features]
    ordered_sample = {feature: sample[feature] for feature in expected_features}
    input_df = pd.DataFrame([ordered_sample])

    return input_df, extra_fields


def predict_quality(
    sample: Dict[str, Any] | None = None,
    model_path: str = "wine_quality_pipeline.pkl",
    model_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Notebook-friendly prediction function.
    Pass a sample dictionary directly from Jupyter or any Python code.
    """
    bundle = model_bundle if model_bundle is not None else load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    expected_features = bundle["feature_columns"]

    if sample is None:
        sample = SAMPLE_INPUT

    input_df, extra_fields = validate_and_prepare_input(sample, expected_features)
    prediction = pipeline.predict(input_df)[0]

    result: Dict[str, Any] = {
        "predicted_quality": int(prediction) if str(prediction).isdigit() else prediction,
        "extra_ignored_fields": extra_fields,
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]
        class_labels = pipeline.named_steps["model"].classes_
        result["class_probabilities"] = {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, probabilities)
        }

    return result


def main() -> None:
    args = parse_args()

    try:
        bundle = load_model_bundle(args.model_path)
        sample = load_input_data(args)
        result = predict_quality(sample=sample, model_bundle=bundle)

        print("Prediction completed successfully.")
        print(f"Predicted quality: {result['predicted_quality']}")

        if "class_probabilities" in result:
            print("Class probabilities:")
            print(json.dumps(result["class_probabilities"], indent=2))

        if result["extra_ignored_fields"]:
            print("\nWarning: Extra input fields were ignored:")
            print(result["extra_ignored_fields"])

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
