#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

EXPECTED_FEATURES: List[str] = [
    "alcohol",
    "density",
    "volatile acidity",
    "chlorides",
    "residual sugar",
]

SAMPLE_INPUT: Dict[str, float] = {
    "alcohol": 12.4,
    "density": 0.9912,
    "volatile acidity": 0.16,
    "chlorides": 0.044,
    "residual sugar": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict wine quality class using a plain saved model.pkl file."
    )
    parser.add_argument(
        "--model_path",
        default="model.pkl",
        help="Path to the saved plain .pkl model. Default: model.pkl",
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


def load_model(model_path: str) -> Any:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def load_input_data(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input_json:
        try:
            payload = json.loads(args.input_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON passed in --input_json: {exc}") from exc
    elif args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input JSON file not found: {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in input file: {exc}") from exc
    else:
        print("No input JSON provided. Using SAMPLE_INPUT for testing.\n")
        payload = SAMPLE_INPUT

    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("This script expects exactly one sample.")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object with feature names as keys.")

    return payload


def validate_and_prepare_input(sample: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    missing_features = [feature for feature in EXPECTED_FEATURES if feature not in sample]
    if missing_features:
        raise ValueError(f"The input sample is missing required fields: {missing_features}")

    extra_fields = [field for field in sample if field not in EXPECTED_FEATURES]

    ordered_sample = {feature: sample[feature] for feature in EXPECTED_FEATURES}
    input_df = pd.DataFrame([ordered_sample])

    for col in EXPECTED_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors="raise")

    return input_df, extra_fields


def predict_quality(sample: Dict[str, Any] | None = None, model_path: str = "model.pkl") -> Dict[str, Any]:
    model = load_model(model_path)

    if sample is None:
        sample = SAMPLE_INPUT

    input_df, extra_fields = validate_and_prepare_input(sample)
    prediction = model.predict(input_df)[0]

    result: Dict[str, Any] = {
        "predicted_class": int(prediction),
        "predicted_label": "Good wine" if int(prediction) == 1 else "Bad wine",
        "extra_ignored_fields": extra_fields,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_labels = model.classes_
        result["class_probabilities"] = {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, probabilities)
        }

    return result


def main() -> None:
    args = parse_args()

    try:
        sample = load_input_data(args)
        result = predict_quality(sample=sample, model_path=args.model_path)

        print("Prediction completed successfully.")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Predicted label: {result['predicted_label']}")

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
