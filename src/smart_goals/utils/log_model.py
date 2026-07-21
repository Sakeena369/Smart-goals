"""Load a sklearn model from MLflow and register it in Snowflake ML Registry.

Example:
    python src/log_model.py \
        --model-uri runs:/<run_id>/model \
        --registry-database MY_DB \
        --registry-schema ML_MODELS \
        --model-name SMART_GOALS \
        --version-name V1 \
        --sample-input-path data/sample_input.csv
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_sample_input(path: str | Path) -> pd.DataFrame:
    """Load sample input rows used to infer model signatures in Snowflake."""
    sample_path = Path(path)
    suffix = sample_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(sample_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(sample_path)

    raise ValueError(
        f"Unsupported sample input format '{suffix}'. Use CSV or Parquet."
    )


def create_session(connection_name: str | None = None) -> Session:
    """Create a Snowpark session from default config or named connection."""
    if connection_name:
        logger.info("Creating Snowpark session using connection_name='%s'", connection_name)
        return Session.builder.config("connection_name", connection_name).create()

    logger.info("Creating Snowpark session using default Snowflake settings")
    return Session.builder.getOrCreate()


def _build_log_model_kwargs(
    log_model_fn: Any,
    model: Any,
    model_name: str,
    version_name: str | None,
    sample_input_data: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build kwargs that match the installed snowflake-ml-python signature."""
    params = set(inspect.signature(log_model_fn).parameters)
    kwargs: dict[str, Any] = {"model": model}

    if "model_name" in params:
        kwargs["model_name"] = model_name
    elif "name" in params:
        kwargs["name"] = model_name

    if version_name:
        if "version_name" in params:
            kwargs["version_name"] = version_name
        elif "version" in params:
            kwargs["version"] = version_name

    sample_param_candidates = [
        "sample_input_data",
        "sample_input",
        "sample_data",
        "input_data",
        "X",
    ]
    for candidate in sample_param_candidates:
        if candidate in params and sample_input_data is not None:
            kwargs[candidate] = sample_input_data
            break

    return kwargs


def register_mlflow_model_to_snowflake(
    model_uri: str,
    registry_database: str,
    registry_schema: str,
    model_name: str,
    version_name: str | None = None,
    mlflow_tracking_uri: str | None = None,
    sample_input_data: pd.DataFrame | None = None,
    connection_name: str | None = None,
) -> Any:
    """Load sklearn model from MLflow and log it to Snowflake ML Registry."""
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info("Using MLflow tracking URI: %s", mlflow_tracking_uri)

    logger.info("Loading sklearn model from MLflow URI: %s", model_uri)
    sklearn_model = mlflow.sklearn.load_model(model_uri)

    try:
        from snowflake.ml.registry import Registry
    except ImportError as exc:
        raise ImportError(
            "snowflake-ml-python is required to use Snowflake ML Registry. "
            "Install it with: pip install snowflake-ml-python"
        ) from exc

    session = create_session(connection_name=connection_name)
    try:
        registry = Registry(
            session=session,
            database_name=registry_database,
            schema_name=registry_schema,
        )

        kwargs = _build_log_model_kwargs(
            log_model_fn=registry.log_model,
            model=sklearn_model,
            model_name=model_name,
            version_name=version_name,
            sample_input_data=sample_input_data,
        )

        logger.info(
            "Registering model '%s' in Snowflake registry %s.%s",
            model_name,
            registry_database,
            registry_schema,
        )
        model_version = registry.log_model(**kwargs)
        logger.info("Model registration complete")
        return model_version
    finally:
        session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load sklearn model from MLflow and register it in Snowflake ML Registry."
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, e.g. runs:/<run_id>/model or models:/name/version.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking server URI (optional).",
    )
    parser.add_argument(
        "--registry-database",
        required=True,
        help="Snowflake database containing the ML Registry.",
    )
    parser.add_argument(
        "--registry-schema",
        required=True,
        help="Snowflake schema containing the ML Registry.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Target model name in Snowflake ML Registry.",
    )
    parser.add_argument(
        "--version-name",
        default=None,
        help="Optional model version name in Snowflake ML Registry.",
    )
    parser.add_argument(
        "--sample-input-path",
        default=None,
        help="Optional path to CSV/Parquet sample input data for model signature inference.",
    )
    parser.add_argument(
        "--connection-name",
        default=None,
        help="Optional Snowflake connection name from ~/.snowflake/connections.toml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_input = (
        load_sample_input(args.sample_input_path) if args.sample_input_path else None
    )

    model_version = register_mlflow_model_to_snowflake(
        model_uri=args.model_uri,
        registry_database=args.registry_database,
        registry_schema=args.registry_schema,
        model_name=args.model_name,
        version_name=args.version_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        sample_input_data=sample_input,
        connection_name=args.connection_name,
    )
    print(f"Registered model version: {model_version}")


if __name__ == "__main__":
    main()
