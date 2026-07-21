"""Load a sklearn model from MLflow and save it locally as a .pkl file."""

from __future__ import annotations

import argparse

import joblib
import mlflow


def get_model_from_mlflow(
    model_uri: str,
    output_path: str,
    mlflow_tracking_uri: str | None = None,
) -> str:
    """Load a sklearn model from MLflow and persist it as a local pickle file."""
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(model, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull a sklearn model from MLflow and save it as a local .pkl file."
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, e.g. models:/smart_goals/1 or runs:/<run_id>/model",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Local path for the output pickle file, e.g. model.pkl",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default="https://mlflow.analytics-dev.tapestry.com",
        help="Optional MLflow tracking URI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = get_model_from_mlflow(
        model_uri=args.model_uri,
        output_path=args.output_path,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
