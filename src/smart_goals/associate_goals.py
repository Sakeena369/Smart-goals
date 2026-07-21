"""
Generate associate-level goal comparison output:
- Actual goal assigned (`GOAL`)
- Smart Goal allocation from model predictions (`SMART_GOAL`)
- Actual sales (`ACTUAL_SALES`)
- Key driver features for analysis
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from snowflake.snowpark import Session

from smart_goals.config import DATA_CONFIG, FEATURE_CONFIG, MLFLOW_CONFIG
from smart_goals.predict import ALLOCATION_ALPHA_DEFAULT, load_data, load_model, predict

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DATE_COL = DATA_CONFIG.date_col
TARGET_COL = DATA_CONFIG.target_col

MAJOR_FEATURE_COLUMNS = [
    "CHNL_DESC",
    "REGION_DESC",
    "TENURE_BUCKET",
    "TENURE_MONTHS_CAPPED_2YR",
    "LOG_TENURE_MONTHS_CAPPED_2YR",
    "LONG_TENURE_X_LOG_TENURE",
    "HOURS_SINCE_OPEN_CORRECTED",
    "STORE_MANAGER_GOAL",
    "IS_HOLIDAY_SEASON",
    "IS_WEEKEND",
    "IS_LONG_TENURE",
    "SPGH_2WKS_SMOOTH",
    "SPGH_8WKS_SMOOTH",
    "SPGH_32WKS_SMOOTH",
]


def _prepare_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same gating logic as training preparation while retaining extra columns
    needed for comparison output.
    """
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])

    mask = pd.Series(True, index=out.index)
    if "GOALS_SCHEDULED_HOURS_CORRECTED" in out.columns:
        mask &= (out["GOALS_SCHEDULED_HOURS_CORRECTED"] >= 4) & (
            out["GOALS_SCHEDULED_HOURS_CORRECTED"] <= 9
        )
    if "STORE_MANAGER_GOAL" in out.columns:
        mask &= out["STORE_MANAGER_GOAL"] > 0
    if "HOURS_SINCE_OPEN" in out.columns:
        mask &= out["HOURS_SINCE_OPEN"].notna()
    if "CHNL_DESC" in out.columns:
        mask &= out["CHNL_DESC"].notna()

    out = out.loc[mask].copy()

    required_model_cols = [DATE_COL] + FEATURE_CONFIG.feature_cols
    missing_model_cols = [c for c in required_model_cols if c not in out.columns]
    if missing_model_cols:
        raise ValueError(f"Missing model input columns: {missing_model_cols}")

    for col, dtype in FEATURE_CONFIG.cast_dtypes.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)

    out = out.sort_values([DATE_COL, "STORE_NUMBER", "SALES_ASSOCIATE"]).reset_index(
        drop=True
    )
    return out


def _apply_optional_filters(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    store_numbers: list[str] | None,
    associates: list[str] | None,
) -> pd.DataFrame:
    out = df.copy()
    if start_date:
        out = out[out[DATE_COL] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out[DATE_COL] <= pd.to_datetime(end_date)]
    if store_numbers:
        out = out[out["STORE_NUMBER"].astype(str).isin(store_numbers)]
    if associates:
        out = out[out["SALES_ASSOCIATE"].astype(str).isin(associates)]
    return out


def _ensure_smart_goal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = [DATE_COL, "STORE_NUMBER", "HOURS_SINCE_OPEN_CORRECTED"]
    daily_group_cols = [DATE_COL, "STORE_NUMBER"]
    if "ASSOCIATE_PCT_OF_PREDICTED" not in out.columns:
        out["ASSOCIATE_PCT_OF_PREDICTED"] = out.groupby(group_cols)[
            "PREDICTED_SALES_PER_GOAL_HOUR"
        ].transform(lambda s: s / s.sum())
    if "ASSOCIATE_PCT_OF_HOURS" not in out.columns:
        out["ASSOCIATE_PCT_OF_HOURS"] = out.groupby(daily_group_cols)[
            "GOALS_SCHEDULED_HOURS_CORRECTED"
        ].transform(lambda s: s / s.sum())
    if "ASSOCIATE_PCT_BLENDED" not in out.columns:
        out["ASSOCIATE_PCT_BLENDED"] = (
            ALLOCATION_ALPHA_DEFAULT * out["ASSOCIATE_PCT_OF_PREDICTED"]
            + (1 - ALLOCATION_ALPHA_DEFAULT) * out["ASSOCIATE_PCT_OF_HOURS"]
        )

    if "SMART_GOAL" not in out.columns and "STORE_MANAGER_GOAL" in out.columns:
        out["SMART_GOAL"] = out["STORE_MANAGER_GOAL"] * out["ASSOCIATE_PCT_BLENDED"]

    return out


def _add_actual_sales_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "GROSS_SALES_AMOUNT_DLABS" in out.columns:
        out["ACTUAL_SALES"] = out["GROSS_SALES_AMOUNT_DLABS"]
        return out
    if "NET_SALES_AMOUNT_DLABS" in out.columns:
        out["ACTUAL_SALES"] = out["NET_SALES_AMOUNT_DLABS"]
        return out
    if TARGET_COL in out.columns and "GOALS_SCHEDULED_HOURS_CORRECTED" in out.columns:
        out["ACTUAL_SALES"] = out[TARGET_COL] * out["GOALS_SCHEDULED_HOURS_CORRECTED"]
        return out

    out["ACTUAL_SALES"] = np.nan
    return out


def _build_comparison_table(
    df: pd.DataFrame,
    include_all_model_features: bool,
) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_smart_goal_columns(out)
    out = _add_actual_sales_column(out)

    if "GOAL" in out.columns:
        out["SMART_GOAL_MINUS_GOAL"] = out["SMART_GOAL"] - out["GOAL"]
        out["ACTUAL_SALES_MINUS_GOAL"] = out["ACTUAL_SALES"] - out["GOAL"]
    if "SMART_GOAL" in out.columns:
        out["ACTUAL_SALES_MINUS_SMART_GOAL"] = out["ACTUAL_SALES"] - out["SMART_GOAL"]

    base_cols = [
        DATE_COL,
        "STORE_NUMBER",
        "SALES_ASSOCIATE",
        "GOAL",
        "SMART_GOAL",
        "ACTUAL_SALES",
        "STORE_MANAGER_GOAL",
        "HISTORICAL_HOURLY_PCT_OF_DAILY_SALES",
        "HOURLY_STORE_GOAL",
        "ASSOCIATE_PCT_OF_PREDICTED",
        "ASSOCIATE_PCT_OF_HOURS",
        "ASSOCIATE_PCT_BLENDED",
        "PREDICTED_SALES_PER_GOAL_HOUR",
        TARGET_COL,
        "SMART_GOAL_MINUS_GOAL",
        "ACTUAL_SALES_MINUS_GOAL",
        "ACTUAL_SALES_MINUS_SMART_GOAL",
    ]
    feature_cols = (
        FEATURE_CONFIG.feature_cols
        if include_all_model_features
        else MAJOR_FEATURE_COLUMNS
    )
    selected_cols = base_cols + feature_cols

    # keep only available columns and preserve order without duplicates
    deduped = []
    seen = set()
    for col in selected_cols:
        if col in out.columns and col not in seen:
            deduped.append(col)
            seen.add(col)

    return out.loc[:, deduped].sort_values(
        [DATE_COL, "STORE_NUMBER", "SALES_ASSOCIATE"]
    )


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    vals = [v.strip() for v in value.split(",") if v.strip()]
    return vals or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build associate goal comparison output from Smart Goals model predictions."
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
        default=MLFLOW_CONFIG.uri,
    )
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--store-numbers",
        type=str,
        default=None,
        help="Comma-separated list, e.g. 4002,1202",
    )
    parser.add_argument(
        "--associates",
        type=str,
        default=None,
        help="Comma-separated list of SALES_ASSOCIATE values",
    )
    parser.add_argument(
        "--include-all-model-features",
        action="store_true",
        help="Include all FEATURE_CONFIG model features instead of only major feature columns.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/associate_goals_comparison.csv",
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
    parser.add_argument("--preview-rows", type=int, default=20)
    args = parser.parse_args()

    logger.info("Creating Snowpark session")
    session = Session.builder.getOrCreate()

    logger.info("Loading model")
    model, model_input_columns = load_model(
        model_stage=args.model_stage,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        return_signature_columns=True,
    )

    logger.info("Loading source data")
    raw_df = load_data(session=session)

    prepared_df = _prepare_inference_data(raw_df)
    prepared_df = _apply_optional_filters(
        prepared_df,
        start_date=args.start_date,
        end_date=args.end_date,
        store_numbers=_parse_csv_arg(args.store_numbers),
        associates=_parse_csv_arg(args.associates),
    )

    logger.info("Running predictions on %d rows", len(prepared_df))
    scored_df = predict(
        model=model,
        data=prepared_df,
        return_predictions_only=False,
        model_input_columns=model_input_columns,
        allocation_alpha=args.allocation_alpha,
    )

    comparison_df = _build_comparison_table(
        scored_df,
        include_all_model_features=args.include_all_model_features,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)

    logger.info("Wrote comparison output: %s", output_path)
    logger.info("Rows: %d, Columns: %d", len(comparison_df), len(comparison_df.columns))
    print(comparison_df.head(args.preview_rows).to_string(index=False))


if __name__ == "__main__":
    main()
