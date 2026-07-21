"""Export an MLflow model to ONNX via local pickle serialization."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from mlflow.types.schema import DataType
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from smart_goals.utils.export_model import export_model_to_onnx
from smart_goals.utils.get_model import get_model_from_mlflow

DEFAULT_MLFLOW_URI = "https://mlflow.analytics-dev.tapestry.com"
MODEL_NAME = "smart_goals"


def _to_onnx_dtype(col_type: DataType) -> Any:
    if col_type in {DataType.float, DataType.double}:
        return FloatTensorType([None, 1])
    if col_type in {DataType.integer, DataType.long}:
        return Int64TensorType([None, 1])
    if col_type == DataType.boolean:
        return BooleanTensorType([None, 1])
    if col_type == DataType.string:
        return StringTensorType([None, 1])
    if col_type == DataType.datetime:
        # ONNX has no native datetime tensor in this conversion path.
        return StringTensorType([None, 1])

    raise ValueError(f"Unsupported MLflow input type for ONNX export: {col_type}")


def get_initial_types_from_signature(
    model_uri: str,
    mlflow_tracking_uri: str | None = None,
) -> list[tuple[str, Any]]:
    """Build skl2onnx initial_types from MLflow model signature inputs."""
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    model_info = mlflow.models.get_model_info(model_uri)
    signature = model_info.signature
    if signature is None or signature.inputs is None:
        raise ValueError("Model has no input signature.")

    initial_types: list[tuple[str, Any]] = []
    for idx, spec in enumerate(signature.inputs):
        name = getattr(spec, "name", None) or f"input_{idx}"
        col_type = getattr(spec, "type", None)
        if col_type is None:
            raise ValueError(
                f"Input '{name}' has no scalar type in signature. "
                "Tensor-only signatures are not supported by this exporter."
            )
        initial_types.append((name, _to_onnx_dtype(col_type)))

    if not initial_types:
        raise ValueError("Model signature inputs are empty.")

    return initial_types


def export_model(
    model_version: str,
    output_path: str | Path,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_URI,
    local_model_path: str | Path | None = None,
) -> Path:
    """Download an MLflow model and export it to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_uri = f"models:/{MODEL_NAME}/{model_version}"

    initial_types = get_initial_types_from_signature(
        model_uri=model_uri,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    if local_model_path:
        model_path = Path(local_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        get_model_from_mlflow(
            model_uri=model_uri,
            output_path=str(model_path),
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        return export_model_to_onnx(
            model_path=model_path,
            output_path=output_path,
            initial_types=initial_types,
        )

    with TemporaryDirectory() as tmp_dir:
        temp_model_path = Path(tmp_dir) / "model.pkl"
        get_model_from_mlflow(
            model_uri=model_uri,
            output_path=str(temp_model_path),
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        return export_model_to_onnx(
            model_path=temp_model_path,
            output_path=output_path,
            initial_types=initial_types,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an MLflow sklearn model to ONNX."
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="MLflow registered model version to export, e.g. 1.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to write the ONNX model, e.g. models/smart_goals.onnx.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_MLFLOW_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_MLFLOW_URI}).",
    )
    parser.add_argument(
        "--local-model-path",
        default=None,
        help="Optional path to persist the downloaded local .pkl model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported_path = export_model(
        model_version=args.model_version,
        output_path=args.output_path,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        local_model_path=args.local_model_path,
    )
    print(f"Exported ONNX model to {exported_path}")


if __name__ == "__main__":
    main()
