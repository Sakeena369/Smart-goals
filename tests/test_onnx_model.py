from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


TESTS_DIR = Path(__file__).resolve().parent
MODEL_PATH = TESTS_DIR.parent / "models" / "smart_goals.onnx"
FIXTURE_DATA_PATH = TESTS_DIR / "fixtures" / "filtered_data_20250302_fixture.csv"


def _shape_with_batch_one(shape: list[int | str | None]) -> tuple[int, ...]:
    normalized: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            normalized.append(dim)
        else:
            normalized.append(1)
    return tuple(normalized)


def _value_for_input(onnx_type: str, shape: tuple[int, ...]) -> np.ndarray:
    if onnx_type == "tensor(float)":
        return np.full(shape, 1.0, dtype=np.float32)
    if onnx_type == "tensor(double)":
        return np.full(shape, 1.0, dtype=np.float64)
    if onnx_type == "tensor(int64)":
        return np.full(shape, 1, dtype=np.int64)
    if onnx_type == "tensor(int32)":
        return np.full(shape, 1, dtype=np.int32)
    if onnx_type == "tensor(bool)":
        return np.full(shape, True, dtype=bool)
    if onnx_type == "tensor(string)":
        return np.full(shape, "test", dtype=object)
    raise ValueError(f"Unsupported ONNX input tensor type: {onnx_type}")


def _value_from_row(value: object, onnx_type: str, shape: tuple[int, ...]) -> np.ndarray:
    if onnx_type == "tensor(float)":
        return np.full(shape, float(value), dtype=np.float32)
    if onnx_type == "tensor(double)":
        return np.full(shape, float(value), dtype=np.float64)
    if onnx_type == "tensor(int64)":
        return np.full(shape, int(value), dtype=np.int64)
    if onnx_type == "tensor(int32)":
        return np.full(shape, int(value), dtype=np.int32)
    if onnx_type == "tensor(bool)":
        return np.full(shape, bool(value), dtype=bool)
    if onnx_type == "tensor(string)":
        return np.full(shape, "" if pd.isna(value) else str(value), dtype=object)
    raise ValueError(f"Unsupported ONNX input tensor type: {onnx_type}")


@pytest.fixture(scope="module")
def onnxruntime_module():
    return pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def model_path() -> Path:
    if not MODEL_PATH.exists():
        pytest.skip(f"Missing ONNX file at {MODEL_PATH}")
    return MODEL_PATH


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    if not FIXTURE_DATA_PATH.exists():
        pytest.skip(f"Missing sample fixture at {FIXTURE_DATA_PATH}")
    df = pd.read_csv(FIXTURE_DATA_PATH)
    if df.empty:
        pytest.skip("Sample fixture is empty.")
    return df


@pytest.fixture(scope="module")
def onnx_session(onnxruntime_module, model_path: Path):
    session = onnxruntime_module.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    if not session.get_inputs():
        pytest.skip("Model has no inputs.")
    return session


def test_onnx_model_exists(model_path: Path) -> None:
    assert model_path.exists(), f"Missing ONNX file at {model_path}"


def test_onnx_model_inference_smoke(onnx_session) -> None:
    feed = {}
    for inp in onnx_session.get_inputs():
        shape = _shape_with_batch_one(inp.shape)
        feed[inp.name] = _value_for_input(inp.type, shape)

    outputs = onnx_session.run(None, feed)
    assert outputs, "Model produced no outputs."

    first_output = np.asarray(outputs[0])
    assert first_output.size > 0, "First output tensor is empty."
    assert np.isfinite(first_output.astype(np.float64)).all(), "Output has non-finite values."


def test_onnx_model_inference_with_real_sample_row(onnx_session, sample_df: pd.DataFrame) -> None:
    row = sample_df.iloc[0]
    feed = {}
    missing_columns: list[str] = []

    for inp in onnx_session.get_inputs():
        if inp.name not in row.index:
            missing_columns.append(inp.name)
            continue
        shape = _shape_with_batch_one(inp.shape)
        feed[inp.name] = _value_from_row(row[inp.name], inp.type, shape)

    assert not missing_columns, f"Sample data is missing model inputs: {missing_columns}"

    outputs = onnx_session.run(None, feed)
    assert outputs, "Model produced no outputs."

    first_output = np.asarray(outputs[0])
    assert first_output.size > 0, "First output tensor is empty."
    assert np.isfinite(first_output.astype(np.float64)).all(), "Output has non-finite values."
