import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark import Window
from snowflake.snowpark import functions as F

from smart_goals.utils.data_loaders import load_store_data, load_store_hours_snowpark


def build_hourly_pct_pd(
    session: Session,
    analysis_start_date: str = "2023-01-01",
    use_dlabs: bool = True,
) -> pd.DataFrame:
    """Build the final hourly_pct_pd table as a pandas DataFrame."""

    store_meta = load_store_data(session).select("STORE_NUMBER", "CHNL_DESC")
    store_hours_tbl = load_store_hours_snowpark(session, deduplicate=True)
    
    if use_dlabs:
        coh_na_txn_tbl = session.table("prd_crm.dlab.dlab_coh_na_txn")
        
        store_daily_sales_hourly = (
        coh_na_txn_tbl.filter(
            (F.nvl(F.col("STORE_NBR"), F.lit("-9999")) > F.lit("0"))
            & (F.col("STORE_NBR") != F.lit("ECOM"))
        )
        .with_column(
            "TXN_DATEHOUR",
            F.trunc(F.col("TXN_DATETIME").cast("timestamp"), F.lit("HOUR")),
        )
        .group_by("TXN_DATE", "TXN_DATEHOUR", "STORE_NBR")
        .agg(F.sum("USD_ITEM_NET_AMT").alias("NET_SALES_AMOUNT"))
        .rename(
                {
                "TXN_DATE": "BUSINESS_DATE",
                "STORE_NBR": "STORE_NUMBER",
                "TXN_DATEHOUR": "BUSINESS_DATEHOUR",
                }
            )
        )

    else:
        raise NotImplementedError("Zipline Data does not have Timestamps and cant be used.")
        from smart_goals.utils.data_loaders import load_associate_daily_sales_snowpark
        store_daily_sales_hourly = load_associate_daily_sales_snowpark(session)
        store_daily_sales_hourly = store_daily_sales_hourly.group_by(
                ["BUSINESS_DATE", "STORE_NUMBER"]
            ).agg(F.sum("NET_SALES_AMOUNT").alias("NET_SALES_AMOUNT_ZIPLINE"))
            # Cast TXN_DATE to date type, STORE_NBR to integer
        
        store_daily_sales_hourly = store_daily_sales_hourly.with_column(
                "BUSINESS_DATE", F.to_date(store_daily_sales_hourly["BUSINESS_DATE"])
            ).with_column(
                "STORE_NUMBER", store_daily_sales_hourly["STORE_NUMBER"].cast("string")
        )

    store_hours_tbl = store_hours_tbl.select(
        "STORE_NUMBER",
        "BUSINESS_DATE",
        F.hour(F.col("OPEN_TIME").cast("timestamp")).alias("OPEN_HOUR"),
        F.hour(F.col("CLOSE_TIME").cast("timestamp")).alias("CLOSE_HOUR"),
    )

    store_hours_simplified = (
        store_hours_tbl.group_by(F.col("STORE_NUMBER"))
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

    window_spec = Window.partition_by("STORE_NUMBER", "BUSINESS_DATE")

    store_daily_sales_hourly_adjusted = (
        store_daily_sales_hourly.filter(F.col("BUSINESS_DATE") > F.lit(analysis_start_date))
        .join(store_hours_simplified, using_columns=["STORE_NUMBER"], how="inner")
        .with_column(
            "PCT_DAILY_SALES",
            F.col("NET_SALES_AMOUNT")
            / F.nullifzero(F.sum("NET_SALES_AMOUNT").over(window_spec)),
        )
        .order_by(["STORE_NUMBER", "BUSINESS_DATE", "BUSINESS_DATEHOUR"])
    )

    store_hourly_pct = (
        store_daily_sales_hourly_adjusted.with_column("HOUR", F.hour("BUSINESS_DATEHOUR"))
        .with_column("MONTH", F.month("BUSINESS_DATE"))
        .with_column("DAY_OF_WEEK", F.dayofweek("BUSINESS_DATE"))
        .with_column("WEEK_OF_YEAR", F.weekofyear("BUSINESS_DATE"))
        .with_column("HOURS_SINCE_OPEN", F.col("HOUR") - F.col("OPEN_HOUR") + 1)
        .with_column(
            "HOURS_SINCE_OPEN_CORRECTED",
            F.when(F.col("HOURS_SINCE_OPEN") <= 1, 1)
            .when(
                F.col("HOURS_SINCE_OPEN") >= (F.col("CLOSE_HOUR") - F.col("OPEN_HOUR")),
                F.col("CLOSE_HOUR") - F.col("OPEN_HOUR"),
            )
            .otherwise(F.col("HOURS_SINCE_OPEN")),
        )
        .join(store_meta, using_columns="STORE_NUMBER")
    )

    hourly_pct_pd = store_hourly_pct.select(
        "BUSINESS_DATE",
        "HOUR",
        "MONTH",
        "DAY_OF_WEEK",
        "WEEK_OF_YEAR",
        "STORE_NUMBER",
        "PCT_DAILY_SALES",
        "NET_SALES_AMOUNT",
        "HOURS_SINCE_OPEN_CORRECTED",
        "CHNL_DESC",
    )

    return hourly_pct_pd


def main(use_dlabs: bool = True, agg = 'by-store') -> pd.DataFrame:
    session = Session.builder.getOrCreate()
    session.use_database("DEV_DS")
    session.use_schema("DBT_DWERNERSEXTON")
    hourly_pct_pd = build_hourly_pct_pd(session=session, analysis_start_date="2023-01-01", use_dlabs=use_dlabs)
    
    if agg == 'store':
        hourly_pct_pd = (
            hourly_pct_pd.group_by("STORE_NUMBER", "HOUR")
            .agg(
                F.avg("PCT_DAILY_SALES").alias("PCT_DAILY_SALES"),
                F.avg("NET_SALES_AMOUNT").alias("NET_SALES_AMOUNT"),
                F.avg("HOURS_SINCE_OPEN_CORRECTED").alias("HOURS_SINCE_OPEN_CORRECTED"),
            )
        )
    elif agg == 'week':
        hourly_pct_pd = (
            hourly_pct_pd.group_by("STORE_NUMBER", "WEEK_OF_YEAR", "HOUR")
            .agg(
                F.avg("PCT_DAILY_SALES").alias("PCT_DAILY_SALES"),
                F.avg("NET_SALES_AMOUNT").alias("NET_SALES_AMOUNT"),
                F.avg("HOURS_SINCE_OPEN_CORRECTED").alias("HOURS_SINCE_OPEN_CORRECTED"),
            )
        )
    elif agg == 'month':
        hourly_pct_pd = (
            hourly_pct_pd.group_by("STORE_NUMBER", "MONTH", "HOUR")
            .agg(
                F.avg("PCT_DAILY_SALES").alias("PCT_DAILY_SALES"),
                F.avg("NET_SALES_AMOUNT").alias("NET_SALES_AMOUNT"),
                F.avg("HOURS_SINCE_OPEN_CORRECTED").alias("HOURS_SINCE_OPEN_CORRECTED"),
            )
        )
    
    else:
        raise ValueError("Invalid aggregation level. Choose either 'store', 'week', or 'month'.")
    
    return hourly_pct_pd


if __name__ == "__main__":
    main()
