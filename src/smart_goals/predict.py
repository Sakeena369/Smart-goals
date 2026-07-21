"""
Inference pipeline for smart_goals model using MLflow.

This script:
1. Loads a trained model from MLflow by stage (for example, Production)
2. Fetches data from Snowflake
3. Prepares the data for inference
4. Generates predictions

Usage:
    from smart_goals.predict import predict_sales_per_goal_hour
    from snowflake.snowpark import Session

    session = Session.builder.getOrCreate()
    predictions_df = predict_sales_per_goal_hour(
        session=session,
        model_stage="Production",
        mlflow_tracking_uri="https://mlflow.analytics-dev.tapestry.com"
    )
"""

import argparse
import logging
from typing import Any, Optional

import mlflow
import pandas as pd
from sklearn import set_config
from snowflake.snowpark import Session

from smart_goals.config import DATA_CONFIG, FEATURE_CONFIG, MLFLOW_CONFIG
from smart_goals.data import load_data_from_snowflake

set_config(transform_output="pandas")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COL = DATA_CONFIG.target_col
DATE_COL = DATA_CONFIG.date_col
MLFLOW_MODEL = MLFLOW_CONFIG.model_name
MLFLOW_URI = MLFLOW_CONFIG.uri
ALLOCATION_ALPHA_DEFAULT = 0.5


def get_signature_input_columns(model_uri: str) -> Optional[list[str]]:
    """
    Get named input columns from a logged MLflow model signature.
    """
    model_info = mlflow.models.get_model_info(model_uri)
    signature = model_info.signature

    if signature is None or signature.inputs is None:
        return None

    input_columns: list[str] = []
    for col in signature.inputs:
        name = getattr(col, "name", None)
        if isinstance(name, str) and name:
            input_columns.append(name)

    return input_columns or None


def load_model(
    model_stage: str = "Production",
    mlflow_tracking_uri: str = MLFLOW_URI,
    return_signature_columns: bool = False,
) -> Any:
    """
    Load a trained model from MLflow.

    Args:
        model_stage: Registered model stage (e.g., Production, Staging)
        mlflow_tracking_uri: MLflow tracking server URI

    Returns:
        Loaded MLflow model (sklearn pipeline)
        If return_signature_columns=True, returns tuple of
        (model, signature_input_columns)
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
    model_uri = f"models:/{MLFLOW_MODEL}/{model_stage}"
    logger.info(f"Loading model stage {model_stage} from registry: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)
    signature_input_columns = get_signature_input_columns(model_uri)
    if signature_input_columns:
        logger.info(
            "Loaded model signature with %d input columns",
            len(signature_input_columns),
        )
    else:
        logger.warning(
            "Model signature is missing or unnamed; using incoming dataframe columns."
        )

    if return_signature_columns:
        return model, signature_input_columns
    return model


def load_data(
    session: Session,
) -> pd.DataFrame:
    """
    Load data from Snowflake using Snowpark.

    Args:
        session: Snowflake Snowpark session
    ) -> pd.DataFrame:
    """
    logger.info("Loading data from Snowflake using SNOWFLAKE_CONFIG defaults")
    df = load_data_from_snowflake(session)
    logger.info(f"Loaded {len(df)} rows from Snowflake")

    return df


def apply_blended_goal_weighting(data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    group_cols = [DATE_COL, "STORE_NUMBER", "HOURS_SINCE_OPEN_CORRECTED"]
    daily_group_cols = [DATE_COL, "STORE_NUMBER"]
    out = data.copy()

    out["STORE_HOURLY_SALES"] = out.groupby(group_cols)[
        "GROSS_SALES_AMOUNT_DLABS"
    ].transform("sum")
    out["STORE_DAILY_SALES"] = out.groupby([DATE_COL, "STORE_NUMBER"])[
        "GROSS_SALES_AMOUNT_DLABS"
    ].transform("sum")
    out["PCT_DAILY_SALES"] = out["STORE_HOURLY_SALES"] / out["STORE_DAILY_SALES"]
    out["HISTORICAL_HOURLY_PCT_OF_DAILY_SALES"] = out.groupby(
        ["STORE_NUMBER", "HOURS_SINCE_OPEN_CORRECTED"]
    )["PCT_DAILY_SALES"].transform("mean")

    hourly_pct_sum = (
        out[
            [
                DATE_COL,
                "STORE_NUMBER",
                "HOURS_SINCE_OPEN_CORRECTED",
                "HISTORICAL_HOURLY_PCT_OF_DAILY_SALES",
            ]
        ]
        .drop_duplicates([DATE_COL, "STORE_NUMBER", "HOURS_SINCE_OPEN_CORRECTED"])
        .groupby(daily_group_cols)["HISTORICAL_HOURLY_PCT_OF_DAILY_SALES"]
        .sum()
        .rename("HISTORICAL_HOURLY_PCT_SUM")
        .reset_index()
    )
    out = out.merge(hourly_pct_sum, on=daily_group_cols, how="left")
    out["HISTORICAL_HOURLY_PCT_OF_DAILY_SALES"] = (
        out["HISTORICAL_HOURLY_PCT_OF_DAILY_SALES"] / out["HISTORICAL_HOURLY_PCT_SUM"]
    )

    out["ASSOCIATE_PCT_OF_PREDICTED"] = out.groupby(group_cols)[
        "PREDICTED_SALES_PER_GOAL_HOUR"
    ].transform(lambda x: x / x.sum())
    out["ASSOCIATE_PCT_OF_HOURS"] = out.groupby(daily_group_cols)[
        "GOALS_SCHEDULED_HOURS_CORRECTED"
    ].transform(lambda x: x / x.sum())
    out["ASSOCIATE_PCT_BLENDED"] = (
        alpha * out["ASSOCIATE_PCT_OF_PREDICTED"]
        + (1 - alpha) * out["ASSOCIATE_PCT_OF_HOURS"]
    )
    out["ASSOCIATE_PCT_BLENDED"] = out["ASSOCIATE_PCT_BLENDED"] / out.groupby(
        group_cols
    )["ASSOCIATE_PCT_BLENDED"].transform("sum")

    out["HOURLY_STORE_GOAL"] = (
        out["STORE_MANAGER_GOAL"] * out["HISTORICAL_HOURLY_PCT_OF_DAILY_SALES"]
    )
    out["SMART_GOAL"] = out["HOURLY_STORE_GOAL"] * out["ASSOCIATE_PCT_BLENDED"]
    return out


def predict(
    model,
    data,
    return_predictions_only: bool = False,
    model_input_columns: Optional[list[str]] = None,
    allocation_alpha: float = ALLOCATION_ALPHA_DEFAULT,
) -> pd.DataFrame:
    """
    Complete inference pipeline for sales per goal hour prediction.

    Args:
        model: Trained MLflow model (sklearn pipeline)
        data: DataFrame with features (target column will be excluded if present)
        return_predictions_only: If True, return only predictions array; if False, return full DataFrame

    Returns:
        DataFrame with predictions (and original data if return_predictions_only=False)
    """
    X = data.drop(columns=[TARGET_COL], errors="ignore")

    if model_input_columns:
        missing_columns = [c for c in model_input_columns if c not in X.columns]
        if missing_columns:
            raise ValueError(
                "Missing columns required by model signature: "
                + ", ".join(sorted(missing_columns))
            )
        X = X.loc[:, model_input_columns]

    X = X.copy()
    for col, dtype in FEATURE_CONFIG.cast_dtypes.items():
        if col in X.columns:
            X[col] = X[col].astype(dtype)

    logger.info("Loading model from MLflow")

    logger.info("Generating predictions")
    predictions = model.predict(X)
    logger.info(f"Generated {len(predictions)} predictions")

    if return_predictions_only:
        return predictions

    data["PREDICTED_SALES_PER_GOAL_HOUR"] = predictions

    if TARGET_COL in data.columns:
        data["RESIDUAL"] = data[TARGET_COL] - data["PREDICTED_SALES_PER_GOAL_HOUR"]
        data["ABSOLUTE_ERROR"] = data["RESIDUAL"].abs()
        data["PERCENTAGE_ERROR"] = (data["RESIDUAL"] / data[TARGET_COL]) * 100
        logger.info(f"Mean Absolute Error: {data['ABSOLUTE_ERROR'].mean():.4f}")
        logger.info(
            f"Root Mean Squared Error: {(data['RESIDUAL'] ** 2).mean() ** 0.5:.4f}"
        )

    data = apply_blended_goal_weighting(data, alpha=allocation_alpha)
    return data


def main():
    """
    Main function for command-line execution.
    """

    parser = argparse.ArgumentParser(
        description="Run inference using smart_goals MLflow model"
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default="Production",
        help="Model stage to load from MLflow registry (default: Production)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=MLFLOW_URI,
        help=f"MLflow tracking server URI (default: {MLFLOW_URI})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save predictions CSV (optional)",
    )
    parser.add_argument(
        "--allocation-alpha",
        type=float,
        default=ALLOCATION_ALPHA_DEFAULT,
        help=(
            "Blend weight for model share vs scheduled-hours share "
            f"(default: {ALLOCATION_ALPHA_DEFAULT})"
        ),
    )

    args = parser.parse_args()

    logger.info("Creating Snowpark session")
    session = Session.builder.getOrCreate()

    model, model_input_columns = load_model(
        model_stage=args.model_stage,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        return_signature_columns=True,
    )

    data = load_data(session=session)

    sample = True
    if sample:
        data = data[
            (data["STORE_NUMBER"] == "4002") & (data["BUSINESS_DATE"] == "2026-01-10")
        ]

    predictions_df = predict(
        model=model,
        data=data,
        return_predictions_only=False,
        model_input_columns=model_input_columns,
        allocation_alpha=args.allocation_alpha,
    )

    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Total predictions: {len(predictions_df)}")
    print("\nPrediction statistics:")
    print(predictions_df["PREDICTED_SALES_PER_GOAL_HOUR"].describe())

    if TARGET_COL in predictions_df.columns:
        print("\nActual vs Predicted:")
        print(f"Mean Actual: {predictions_df[TARGET_COL].mean():.4f}")
        print(
            f"Mean Predicted: {predictions_df['PREDICTED_SALES_PER_GOAL_HOUR'].mean():.4f}"
        )
        print(f"MAE: {predictions_df['ABSOLUTE_ERROR'].mean():.4f}")
        print(f"RMSE: {(predictions_df['RESIDUAL'] ** 2).mean() ** 0.5:.4f}")

    if args.output_path:
        logger.info(f"Saving predictions to: {args.output_path}")
        predictions_df.to_csv(args.output_path, index=False)
        print(f"\nPredictions saved to: {args.output_path}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
