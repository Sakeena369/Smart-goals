"""Export a trained scikit-learn model to ONNX using skl2onnx.

Example:
    python src/export_model.py \
        --model-path artifacts/model.pkl \
        --output-path artifacts/model.onnx \
        --n-features 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
from skl2onnx import convert_sklearn
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import FloatTensorType

DEFAULT_ONNX_OPSET = 17
DEFAULT_ONNX_ML_OPSET = 3
TARGET_OPSET = {"": DEFAULT_ONNX_OPSET, "ai.onnx.ml": DEFAULT_ONNX_ML_OPSET}


def _register_lightgbm_converter() -> None:
    """Register LightGBM converters required by skl2onnx."""
    try:
        from lightgbm import LGBMRegressor
        from onnxmltools.convert.common.shape_calculator import (
            calculate_linear_regressor_output_shapes,
        )
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,
        )
    except ImportError as exc:
        raise ImportError(
            "LightGBM ONNX export requires 'onnxmltools'. "
            "Install it with: uv add --dev onnxmltools"
        ) from exc

    update_registered_converter(
        LGBMRegressor,
        "LightGbmLGBMRegressor",
        calculate_linear_regressor_output_shapes,
        convert_lightgbm,
    )


def export_model_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    n_features: int | None = None,
    input_name: str = "input",
    target_opset: int | dict[str, int] = TARGET_OPSET,
    initial_types: list[tuple[str, Any]] | None = None,
) -> Path:
    """Load a sklearn model from disk and export it to ONNX format."""
    model_path = Path(model_path)
    output_path = Path(output_path)

    if initial_types is None:
        if n_features is None or n_features <= 0:
            raise ValueError(
                "Provide either initial_types or a positive n_features value."
            )
        initial_types = [(input_name, FloatTensorType([None, n_features]))]

    model = joblib.load(model_path)
    _register_lightgbm_converter()

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=target_opset,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained scikit-learn model to ONNX using skl2onnx."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a serialized sklearn model (e.g., .pkl/.joblib).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write the ONNX file (e.g., model.onnx).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        required=True,
        help="Number of input features expected by the model.",
    )
    parser.add_argument(
        "--input-name",
        default="input",
        help="ONNX input tensor name (default: input).",
    )
    parser.add_argument(
        "--target-opset",
        type=int,
        default=DEFAULT_ONNX_ML_OPSET,
        help=f"ONNX-ML domain opset version to target (default: {DEFAULT_ONNX_ML_OPSET}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported_path = export_model_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        n_features=args.n_features,
        input_name=args.input_name,
        target_opset={"": DEFAULT_ONNX_OPSET, "ai.onnx.ml": args.target_opset},
    )
    print(f"Exported ONNX model to {exported_path}")


if __name__ == "__main__":
    main()
