"""
Fetch and process associate goals data from Snowflake.

This script:
1. Connects to Snowflake and loads data from various sources
2. Joins transactions, goals, schedules, and store information
3. Calculates features like tenure, shift duration, and time of day
4. Saves the final dataset to CSV

Usage:
    from smart_goals.fetch_data import fetch_associate_goals_data

    df = fetch_associate_goals_data(
        analysis_start_date='2023-01-01',
        output_path='/home/sagemaker-user/smart-goals/data/associate_goals_df.csv'
    )
"""

import logging

import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark import Window
from snowflake.snowpark.types import DecimalType
import decimal

from smart_goals.config import FEATURE_CONFIG, DATA_CONFIG, SNOWFLAKE_CONFIG
from smart_goals.utils.data_loaders import (
    load_associate_daily_sales_snowpark,
    load_associate_daily_schedule_snowpark,
    load_associate_goals_snowpark,
    load_store_data,
    load_store_hours_snowpark,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATE_COL = DATA_CONFIG.date_col
TARGET_COL = DATA_CONFIG.target_col
ANALYSIS_START_DATE = DATA_CONFIG.analysis_start_date
SNOWFLAKE_DATABASE = SNOWFLAKE_CONFIG.database
SNOWFLAKE_SCHEMA = SNOWFLAKE_CONFIG.schema
SNOWFLAKE_TABLE = SNOWFLAKE_CONFIG.table
SNOWFLAKE_WAREHOUSE = SNOWFLAKE_CONFIG.warehouse

ROLLING_WEEK_WINDOWS = [32, 16, 8, 4, 2]
BLEND_PRIOR_HOURS = 40.0

def _week_window(weeks: int, partition_by: list[str]):
    """Create a time-range window over BUSINESS_DATE for any partition key(s)."""
    return (
        Window.partition_by(*partition_by)
        .order_by("BUSINESS_DATE")
        .range_between(-F.make_interval(weeks=weeks), -F.make_interval(days=1))
    )


def load_data_from_file(file_path: str) -> pd.DataFrame:
    """
    Load training data from a CSV file.

    Args:
        file_path: Path to the CSV file
    Returns:
        DataFrame with training data
    """
    if file_path.endswith(".feather"):
        df = pd.read_feather(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("File must be .feather or .csv")
    return df


def load_data_from_snowflake(
    session: "Session"
) -> pd.DataFrame:
    """
    Load training data from Snowflake using a Snowpark session.

    Args:
        session: Snowpark session object
    Returns:
        DataFrame with training data
    """
    session.use_database(SNOWFLAKE_DATABASE)
    session.use_schema(SNOWFLAKE_SCHEMA)
    session.use_warehouse(SNOWFLAKE_WAREHOUSE)
    df = session.table(SNOWFLAKE_TABLE).to_pandas()
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for training - just basic sorting.
    Feature engineering is handled by the sklearn pipeline.

    Args:
        df: Raw dataframe
    Returns:
        Prepared dataframe
    """
    mask = (
        (df["GOALS_SCHEDULED_HOURS_CORRECTED"] >= 4)
        & (df["GOALS_SCHEDULED_HOURS_CORRECTED"] <= 9)
        & (df["STORE_MANAGER_GOAL"] > 0)
        & (df["HOURS_SINCE_OPEN"].notna())
        & (df["CHNL_DESC"].notna()) # Store 3824 has some NULL 
    )
    df = df.loc[mask].copy()
    df.loc[:, DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    required_cols = [DATE_COL, TARGET_COL] + FEATURE_CONFIG.feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required model input columns: {missing_cols}")

    df = df[required_cols]

    for col, dtype in FEATURE_CONFIG.cast_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    return df


def fetch_associate_goals_data(
    analysis_start_date: str = ANALYSIS_START_DATE,
    database=SNOWFLAKE_DATABASE,
    schema=SNOWFLAKE_SCHEMA,
):
    """
    Fetch and process associate goals data from Snowflake.

    Parameters
    ----------
    analysis_start_date : str
        Start date for transaction data analysis (YYYY-MM-DD)
    goal_start_date : str
        Start date for goal and schedule data (YYYY-MM-DD)
    output_path : str, optional
        Path to save the output file. If None, saves to data/associate_goals_df.{format}
    database : str
        Snowflake database name
    schema : str
        Snowflake schema name
    save_format : str
        Output format: 'csv', 'feather', or 'parquet'

    Returns
    -------
    snowflake.snowpark.Table
        Processed associate goals dataframe
    """
    logger.info("Starting data fetch process")
    logger.info(f"Analysis start date: {analysis_start_date}")

    # Create Snowflake session
    logger.info("Creating Snowflake session")
    session = Session.builder.getOrCreate()
    session.use_database(database)
    session.use_schema(schema)

    # Load transaction data for associate sales
    logger.info("Loading transaction data from COH_NA_TXN table")
    coh_na_txn_tbl = session.table("prd_crm.dlab.dlab_coh_na_txn")

    # Calculate associate daily sales from transaction data
    logger.info("Calculating associate daily sales from DLABS")
    associate_daily_sales_hourly_dlabs_tbl = (
        coh_na_txn_tbl.filter(
            (F.nvl(F.col("SALES_ASSOCIATE"), F.lit("999999999")) < F.lit("999900000"))
            & (F.col("TXN_DATETIME") >= F.lit(analysis_start_date))
        )
        .with_column(
            "TXN_DATEHOUR",
            F.trunc(F.col("TXN_DATETIME").cast("timestamp"), F.lit("HOUR")),
        )
        .with_column(
            "USD_ITEM_GROSS_AMT",
            F.when(
                F.col("SALE_CREDIT_CODE") == "1", F.col("USD_ITEM_NET_AMT")
            ).otherwise(0),
        )
        .group_by("TXN_DATE", "TXN_DATEHOUR", "STORE_NBR", "SALES_ASSOCIATE")
        .agg(
            F.sum("USD_ITEM_NET_AMT").alias("NET_SALES_AMOUNT_DLABS"),
            F.sum("USD_ITEM_GROSS_AMT").alias("GROSS_SALES_AMOUNT_DLABS"),
        )
        .rename({"TXN_DATE": "BUSINESS_DATE", "STORE_NBR": "STORE_NUMBER"})
    )

    associate_daily_sales_dlabs_tbl = associate_daily_sales_hourly_dlabs_tbl.group_by(
        "BUSINESS_DATE", "STORE_NUMBER", "SALES_ASSOCIATE"
    ).agg(
        F.sum("NET_SALES_AMOUNT_DLABS").alias("NET_SALES_AMOUNT_DLABS"),
        F.sum("GROSS_SALES_AMOUNT_DLABS").alias("GROSS_SALES_AMOUNT_DLABS"),
    )

    # Load associate goals and hours
    logger.info("Loading associate goals and hours data")
    associate_goal_hours_tbl = load_associate_goals_snowpark(session)
    associate_goal_hours_tbl = associate_goal_hours_tbl.filter(
        (F.col("BUSINESS_DATE") >= F.lit("2025-11-01")) & 
        (F.nvl(F.col('GOALS_SCHEDULED_HOURS'), F.lit(0)) > 0)  # Focus on days when goals were scheduled
    )

    # Join sales with goals/hours
    # An Outer join keeps all associates even if they have no sales or no goals, which is important for training the model to understand the impact of having no goals or no sales.
    logger.info("Joining sales data with goals and hours")
    associate_daily_sales_dlab_with_hours_tbl = associate_goal_hours_tbl.join(
        associate_daily_sales_dlabs_tbl,
        using_columns=["BUSINESS_DATE", "STORE_NUMBER", "SALES_ASSOCIATE"],
        how="left",
    )

    # Load Zipline sales data for validation
    logger.info("Loading Zipline sales data")
    associate_daily_sales_zipline_tbl = load_associate_daily_sales_snowpark(session)
    associate_daily_sales_zipline_tbl = associate_daily_sales_zipline_tbl.rename(
        {"NET_SALES_AMOUNT": "NET_SALES_AMOUNT_ZIPLINE"}
    )

    # Join with Zipline data
    logger.info("Joining with Zipline sales data")
    associate_sales_hours_all_tbl = associate_daily_sales_dlab_with_hours_tbl.join(
        associate_daily_sales_zipline_tbl,
        using_columns=["BUSINESS_DATE", "STORE_NUMBER", "SALES_ASSOCIATE"],
        how="left",
    ).fillna({
        "NET_SALES_AMOUNT_DLABS": decimal.Decimal(0),   
        "GROSS_SALES_AMOUNT_DLABS": decimal.Decimal(0)
    })

    # Calculate associate tenure
    logger.info("Calculating associate tenure")
    associate_tenure_days = coh_na_txn_tbl.group_by("SALES_ASSOCIATE").agg(
        F.min("TXN_DATE").alias("FIRST_SELLING_DATE")
    ).with_column("IS_LONG_TENURE", F.when(F.col("FIRST_SELLING_DATE") < 
    F.lit("2020-01-01"), F.lit(1)).otherwise(F.lit(0))
    )

    # complete_tenure_days_tbl = associate_tenure_days.filter(
    #     F.col("FIRST_SELLING_DATE").is_not_null()
    #     & (F.col("FIRST_SELLING_DATE") >= F.lit(associate_tenure_start_date))
    # )

    # Load associate schedules
    logger.info("Loading associate daily schedules")
    associate_daily_schedule_tbl = load_associate_daily_schedule_snowpark(
        session, deduplicate=True
    )
    associate_daily_schedule_tbl = associate_daily_schedule_tbl.select(
        "STORE_NUMBER",
        "SALES_ASSOCIATE",
        "BUSINESS_DATE",
        "SHIFT_TYPE",
        F.hour(F.col("SHIFT_START").cast("timestamp")).alias("SHIFT_START_HOUR"),
        F.hour(F.col("SHIFT_END").cast("timestamp")).alias("SHIFT_END_HOUR"),
    ).filter(F.col("BUSINESS_DATE") >= F.lit("2025-11-01"))

    logger.info("Loading store operating hours")
    store_hours_tbl = load_store_hours_snowpark(session)
    store_hours_tbl = (
        store_hours_tbl.select(
            "STORE_NUMBER",
            "BUSINESS_DATE",
            F.hour(F.col("OPEN_TIME").cast("timestamp")).alias("OPEN_HOUR"),
            F.hour(F.col("CLOSE_TIME").cast("timestamp")).alias("CLOSE_HOUR"),
        )
        .group_by(F.col("STORE_NUMBER"))
        .agg(
            F.mode("OPEN_HOUR").alias("OPEN_HOUR"),
            F.mode("CLOSE_HOUR").alias("CLOSE_HOUR"),
        )
        .with_column(
            "OPEN_HOUR",
            F.when(F.col("OPEN_HOUR") == 0, F.lit(10)).otherwise(F.col("OPEN_HOUR")),
        )
        .with_column(
            "CLOSE_HOUR",
            F.when(F.col("CLOSE_HOUR") == 0, F.lit(20)).otherwise(F.col("CLOSE_HOUR")),
        )
    )

    associate_daily_schedule_with_store_hours = (
        associate_daily_schedule_tbl.join(
            store_hours_tbl, using_columns=["STORE_NUMBER"], how="left"
        )
        .with_column(
            "HOURS_SINCE_OPEN",
            F.when(
                F.col("SHIFT_START_HOUR") >= F.col("OPEN_HOUR"),
                F.col("SHIFT_START_HOUR") - F.col("OPEN_HOUR"),
            ).otherwise(0),
        )
        .with_column(
            "HOURS_SINCE_OPEN_CORRECTED",
            F.when(F.col("HOURS_SINCE_OPEN") >= 0, F.col("HOURS_SINCE_OPEN")).otherwise(
                0
            ),
        )
    )

    # Load store master data
    logger.info("Loading store master data")
    store_mstr = load_store_data(session)

    # Join all data together
    logger.info("Creating final associate goals dataset")

    associate_rolling_windows = {
        weeks: _week_window(weeks, ["SALES_ASSOCIATE"])
        for weeks in ROLLING_WEEK_WINDOWS
    }
    store_rolling_windows = {
        weeks: _week_window(weeks, ["STORE_NUMBER"])
        for weeks in ROLLING_WEEK_WINDOWS
    }
    

    # Add the rolling average columns with INTERVAL filtering
    final_associate_goals_tbl = (
        associate_sales_hours_all_tbl.join(
            associate_daily_schedule_with_store_hours,
            using_columns=["BUSINESS_DATE", "STORE_NUMBER", "SALES_ASSOCIATE"],
            how="left",
        )
        # .join(complete_tenure_days_tbl, using_columns=["SALES_ASSOCIATE"], how="inner")
        .join(associate_tenure_days, using_columns=["SALES_ASSOCIATE"], how="inner")  # Associate must have made at least one sale to be included in the dataset, so inner join is appropriate here
        .filter(F.col('BUSINESS_DATE') >= F.col('FIRST_SELLING_DATE')) #TODO: VALIDATE WITH BUSINESS
        .join(store_mstr, using_columns=["STORE_NUMBER"], how="left")
        .order_by(["STORE_NUMBER", "BUSINESS_DATE", "SALES_ASSOCIATE"])
        .with_column("MONTH", F.month("BUSINESS_DATE"))
        .with_column("YEAR", F.year("BUSINESS_DATE"))
        .with_column("DAY_OF_WEEK", F.dayofweek("BUSINESS_DATE"))
        .with_column("WEEK_OF_YEAR", F.weekofyear("BUSINESS_DATE"))
        .with_column(
            "TENURE_MONTHS",
            F.datediff("MONTH", F.col("FIRST_SELLING_DATE"), F.col("BUSINESS_DATE")),
        )
        .with_column("LOG_TENURE", F.log(10, F.col("TENURE_MONTHS") + F.lit(1)))
        # Observed tenure since 2020-01-01 to avoid over-trusting pre-ID-switch history.
        .with_column(
            "TENURE_MONTHS_CAPPED_2YR",
            F.least(F.col("TENURE_MONTHS"), F.lit(24)),
        )
        .with_column(
            "LOG_TENURE_MONTHS_CAPPED_2YR",
            F.log(10, F.col("TENURE_MONTHS_CAPPED_2YR") + F.lit(1)),
        )
        .with_column(
            "LONG_TENURE_X_LOG_TENURE",
            F.col("IS_LONG_TENURE") * F.col("LOG_TENURE"),
        )
        .with_column(
            "TENURE_BUCKET",
            F.when(F.col("TENURE_MONTHS") <= 1, F.lit("0_30d"))
            .when(F.col("TENURE_MONTHS") <= 3, F.lit("31_90d"))
            .when(F.col("TENURE_MONTHS") <= 6, F.lit("91_180d"))
            .otherwise(F.lit("180d_plus")),
        )        
        # Handle overnight shifts
        .with_column(
            "SHIFT_DURATION",
            F.when(
                F.col("SHIFT_END_HOUR") < F.col("SHIFT_START_HOUR"),
                24 - F.col("SHIFT_START_HOUR") + F.col("SHIFT_END_HOUR"),
            ).otherwise(F.col("SHIFT_END_HOUR") - F.col("SHIFT_START_HOUR")),
        )
        .with_column("MAX_POSSIBLE_HOURS", F.col("CLOSE_HOUR") - F.col("OPEN_HOUR"))
        .with_column(
            "GOALS_SCHEDULED_HOURS_CORRECTED",
            F.when(
                F.col("GOALS_SCHEDULED_HOURS") > F.col("SHIFT_DURATION"),
                F.col("SHIFT_DURATION"),
            )
            .when(
                F.col("GOALS_SCHEDULED_HOURS") > F.col("MAX_POSSIBLE_HOURS"),
                F.col("MAX_POSSIBLE_HOURS"),
            )
            .otherwise(F.col("GOALS_SCHEDULED_HOURS")),
        )
        .with_column(
            "SEASONAL_HIRE",
            F.when(
                F.date_part("month", F.col("FIRST_SELLING_DATE")).isin([11, 12]),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .with_column(
            "IS_HOLIDAY_SEASON",
            F.when(F.col("MONTH").isin([11, 12]), F.lit(1)).otherwise(F.lit(0)),
        )
        .with_column(
            "IS_WEEKEND",
            F.when(F.col("DAY_OF_WEEK").isin([0, 5, 6]), F.lit(1)).otherwise(F.lit(0)),
        )
        .with_column(
            "SALES_PER_GOAL_HOUR",
            F.div0(F.col("GROSS_SALES_AMOUNT_DLABS"),
                   F.col("GOALS_SCHEDULED_HOURS_CORRECTED")
                ).cast(DecimalType(38, 2))
        )
    )

    for weeks in ROLLING_WEEK_WINDOWS:
        associate_window = associate_rolling_windows[weeks]
        store_window = store_rolling_windows[weeks]

        final_associate_goals_tbl = (final_associate_goals_tbl
            .with_column(
                f"GOAL_ATTAIN_PCT_{weeks}WKS",
                F.div0(
                    F.sum(F.col("GROSS_SALES_AMOUNT_DLABS")).over(associate_window),
                    F.sum(F.col("GOAL")).over(associate_window)
                ) - 1
            )
            .with_column(
                f"SPGH_{weeks}WKS",
                F.avg(F.col("SALES_PER_GOAL_HOUR")).over(associate_window),
            )
            .with_column(
                f"GOAL_ATTAIN_PCT_{weeks}WKS_STORE",
                F.div0(
                    F.sum(F.col("GROSS_SALES_AMOUNT_DLABS")).over(store_window),
                    F.sum(F.col("GOAL")).over(store_window)
                ) - 1
            )
            .with_column(
                f"SPGH_{weeks}WKS_STORE",
                F.avg(F.col("SALES_PER_GOAL_HOUR")).over(store_window),
            )
            .with_column(
                f"EMP_EXPOSURE_HOURS_{weeks}WKS",
                F.sum(F.col("GOALS_SCHEDULED_HOURS_CORRECTED")).over(associate_window),
            )
            .with_column(
                f"EMP_WEIGHT_{weeks}WKS",
                F.div0(
                    F.col(f"EMP_EXPOSURE_HOURS_{weeks}WKS"),
                    F.col(f"EMP_EXPOSURE_HOURS_{weeks}WKS") + F.lit(BLEND_PRIOR_HOURS),
                ),
            )
            .with_column(
                f"SPGH_{weeks}WKS_SMOOTH",
                F.when(
                    F.col(f"SPGH_{weeks}WKS").is_null(),
                    F.col(f"SPGH_{weeks}WKS_STORE"),
                ).when(
                    F.col(f"SPGH_{weeks}WKS_STORE").is_null(),
                    F.col(f"SPGH_{weeks}WKS"),
                ).otherwise(
                    F.col(f"EMP_WEIGHT_{weeks}WKS") * F.col(f"SPGH_{weeks}WKS")
                    + (F.lit(1.0) - F.col(f"EMP_WEIGHT_{weeks}WKS"))
                    * F.col(f"SPGH_{weeks}WKS_STORE")
                ),
            )
            .with_column(
                f"GOAL_ATTAIN_PCT_{weeks}WKS_SMOOTH",
                F.when(
                    F.col(f"GOAL_ATTAIN_PCT_{weeks}WKS").is_null(),
                    F.col(f"GOAL_ATTAIN_PCT_{weeks}WKS_STORE"),
                ).when(
                    F.col(f"GOAL_ATTAIN_PCT_{weeks}WKS_STORE").is_null(),
                    F.col(f"GOAL_ATTAIN_PCT_{weeks}WKS"),
                ).otherwise(
                    F.col(f"EMP_WEIGHT_{weeks}WKS") * F.col(f"GOAL_ATTAIN_PCT_{weeks}WKS")
                    + (F.lit(1.0) - F.col(f"EMP_WEIGHT_{weeks}WKS"))
                    * F.col(f"GOAL_ATTAIN_PCT_{weeks}WKS_STORE")
                ),
            )
        )

    return final_associate_goals_tbl


def main():
    """Command-line interface for data fetching."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch and process associate goals data"
    )
    parser.add_argument(
        "--analysis-start-date",
        default=ANALYSIS_START_DATE,
        help="Start date for transaction data analysis (YYYY-MM-DD)",
    )
    parser.add_argument("--database", default=SNOWFLAKE_DATABASE, help="Snowflake database name")
    parser.add_argument(
        "--schema", default=SNOWFLAKE_SCHEMA, help="Snowflake schema name"
    )

    args = parser.parse_args()

    final_associate_goals_tbl = fetch_associate_goals_data(
        analysis_start_date=args.analysis_start_date,
        database=args.database,
        schema=args.schema,
    )

    # Convert to pandas
    logger.info("Converting to pandas DataFrame")

    session = Session.builder.getOrCreate()
    session.use_database(args.database)
    session.use_schema(args.schema)
    logger.info(
        f"Saving data back to Snowflake table {session.get_current_database()}.{session.get_current_schema()}.{SNOWFLAKE_TABLE}"
    )
    final_associate_goals_tbl.write.save_as_table(
        SNOWFLAKE_TABLE, mode="overwrite"
    )


if __name__ == "__main__":
    main()
