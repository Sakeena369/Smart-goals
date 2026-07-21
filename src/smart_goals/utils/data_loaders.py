import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_store_goals_snowpark(session):
    filepath = '/home/sagemaker-user/smart-goals/raw-data/Store_Sales_Goal_Data.csv'
    df = pd.read_csv(filepath, parse_dates=['Business_Date'], dtype={'Store_Num': str})
    df.columns = df.columns.str.upper()
    logger.info("Renaming STORE_NUM to STORE_NUMBER")
    df = df.rename(columns={'STORE_NUM': 'STORE_NUMBER'})
    logger.info("Selecting columns: STORE_NUMBER, BUSINESS_DATE, FINANCE_TEAM_GOAL, STORE_MANAGER_GOAL")
    df = df[['STORE_NUMBER', 'BUSINESS_DATE', 'FINANCE_TEAM_GOAL', 'STORE_MANAGER_GOAL']]
    return session.create_dataframe(df)

def load_associate_daily_sales_snowpark(session):
    filepath = '/home/sagemaker-user/smart-goals/raw-data/Associate_Daily_Sales_Data.csv'
    df = pd.read_csv(filepath, parse_dates=['Business_Date'], dtype={'Store_Number': str, 'Employee_Num': str})
    df.columns = df.columns.str.upper()
    logger.info("Renaming EMPLOYEE_NUM to SALES_ASSOCIATE")
    df = df.rename(columns={'EMPLOYEE_NUM': 'SALES_ASSOCIATE'})
    logger.info("Selecting columns: STORE_NUMBER, BUSINESS_DATE, SALES_ASSOCIATE, NET_SALES_AMOUNT")
    df = df[['STORE_NUMBER', 'BUSINESS_DATE', 'SALES_ASSOCIATE', 'NET_SALES_AMOUNT']]
    return session.create_dataframe(df)

def load_store_hours_snowpark(session, deduplicate=True):
    filepath = '/home/sagemaker-user/smart-goals/raw-data/Coach_NA_Store_Operating_Hours.csv'
    df = pd.read_csv(filepath, parse_dates=['BusinessDate', 'OpenTime', 'CloseTime'], dtype={'StoreNum': str})
    df.columns = df.columns.str.upper()
    logger.info("Renaming columns: STORENUM->STORE_NUMBER, BUSINESSDATE->BUSINESS_DATE, OPENTIME->OPEN_TIME, CLOSETIME->CLOSE_TIME")
    df = df.rename(columns={'STORENUM': 'STORE_NUMBER', 'BUSINESSDATE': 'BUSINESS_DATE', 'OPENTIME': 'OPEN_TIME', 'CLOSETIME': 'CLOSE_TIME'})
    if deduplicate:
        logger.info("Dropping duplicate rows based on STORE_NUMBER and BUSINESS_DATE")
        df = df.drop_duplicates(subset=['STORE_NUMBER', 'BUSINESS_DATE'])
    return session.create_dataframe(df)

def load_associate_goals_snowpark(session):
    filepath = '/home/sagemaker-user/smart-goals/raw-data/Associate_Goal_&_Hours_Data.csv'
    df = pd.read_csv(filepath, parse_dates=['Business_Date'], dtype={'Store_Num': str, 'Employee_Num': str})
    df.columns = df.columns.str.upper()
    logger.info("Renaming columns: EMPLOYEE_NUM->SALES_ASSOCIATE, STORE_NUM->STORE_NUMBER")
    df = df.rename(columns={'EMPLOYEE_NUM': 'SALES_ASSOCIATE', 'STORE_NUM': 'STORE_NUMBER'})
    logger.info("Selecting columns: STORE_NUMBER, BUSINESS_DATE, SALES_ASSOCIATE, FINANCE_TEAM_GOAL, STORE_MANAGER_GOAL, LEGION_SCHEDULED_HOURS, GOALS_SCHEDULED_HOURS, GOAL")
    df = df[['STORE_NUMBER', 'BUSINESS_DATE', 'SALES_ASSOCIATE', 'FINANCE_TEAM_GOAL', 'STORE_MANAGER_GOAL','LEGION_SCHEDULED_HOURS', 'GOALS_SCHEDULED_HOURS', 'GOAL']]
    return session.create_dataframe(df)

def load_associate_daily_schedule_snowpark(session, deduplicate=True):
    filepath = '/home/sagemaker-user/smart-goals/raw-data/Legion_Schedule_Raw_Data_FY25_FY26.csv'
    df = pd.read_csv(filepath, parse_dates=['Business_Date'], dtype={'Store_Num': str, 'Employee_Num': str})
    df.columns = df.columns.str.upper()
    if deduplicate:
        logger.info("Sorting by MODIFIED_DATE and dropping duplicates based on STORE_NUM, BUSINESS_DATE, EMPLOYEE_NUM")
        df = df.sort_values('MODIFIED_DATE', ascending=False)
        df = df.drop_duplicates(subset=['STORE_NUM', 'BUSINESS_DATE', 'EMPLOYEE_NUM'], keep='first')
    logger.info("Renaming columns: EMPLOYEE_NUM->SALES_ASSOCIATE, STORE_NUM->STORE_NUMBER")
    df = df.rename(columns={'EMPLOYEE_NUM': 'SALES_ASSOCIATE', 'STORE_NUM': 'STORE_NUMBER'})
    logger.info("Selecting columns: STORE_NUMBER, BUSINESS_DATE, SALES_ASSOCIATE, SHIFT_START, SHIFT_END, SHIFT_TYPE")
    df = df[['STORE_NUMBER', 'BUSINESS_DATE', 'SALES_ASSOCIATE', 'SHIFT_START', 'SHIFT_END', 'SHIFT_TYPE']]
    return session.create_dataframe(df)

def load_store_data(session):
    filepath = '/home/sagemaker-user/smart-goals/raw-data/Snowflake_VW_RPT_SPOG_EMP_CURRENT_DAY_2026-01-21.csv'
    df = pd.read_csv(filepath, parse_dates=['CAL_DATE'], dtype={'MST_LOCATION_CODE': str, 'EMP_ID': str})
    df.columns = df.columns.str.upper()
    logger.info("Renaming columns: EMP_ID->SALES_ASSOCIATE, MST_LOCATION_CODE->STORE_NUMBER, CAL_DATE->BUSINESS_DATE")
    df = df.rename(columns={'EMP_ID': 'SALES_ASSOCIATE', 'MST_LOCATION_CODE': 'STORE_NUMBER', 'CAL_DATE': 'BUSINESS_DATE'})
    logger.info("Selecting columns: STORE_NUMBER, CHNL_DESC, REGION_DESC, DISTRICT_DESC")
    df = df[['STORE_NUMBER', 'CHNL_DESC', 'REGION_DESC', 'DISTRICT_DESC']]
    logger.info("Dropping duplicate rows based on STORE_NUMBER")
    df = df.drop_duplicates(subset=['STORE_NUMBER'])
    return session.create_dataframe(df)