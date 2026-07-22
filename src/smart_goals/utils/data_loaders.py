import pandas as pd
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_store_goals(filepath="Store_Goals_Data.csv", output_path="data/prepared/store_goals.csv"):
    filepath = 'Store_Goals_Data.csv'
    df = pd.read_csv(filepath,
    encoding='utf-8-sig',
    header=0,#treats the first row as header rows
    #the headers contains broken header names, so we need to use the names parameter to specify the column names
    names=["GOAL_ID",
    "BUSINESS_DATE",
    "STORE_NUMBER",
     "BRAND",
     "FINANCE_TEAM_GOAL",
     "STORE_MANAGER_GOAL",
     "MODIFIED_DATE",
     "MODIFIED_BY",
     ],
     #to parse the dates as real dates and not strings and also keep the goal_id and store_number as strings so leading zeroes are not lost.
     parse_dates=['BUSINESS_DATE'], dtype={"GOAL_ID": str, "STORE_NUMBER": str},
     )
    #rename the STORE_NUM column to STORE_NUMBER
    logger.info("Renaming STORE_NUM to STORE_NUMBER")
    df = df.rename(columns={'STORE_NUM': 'STORE_NUMBER'})
    #sort by modified date and drop duplicates based on GOAL_ID
    logger.info("Dropping duplicate rows based on GOAL_ID and keeping latest modified date")
    df = df.sort_values('MODIFIED_DATE', ascending=False)
    df = df.drop_duplicates(subset=['GOAL_ID'], keep='first')
    #keep only rows with STORE_MANAGER_GOAL>0
    logger.info("keeping rows with STORE_MANAGER_GOAL>0 ")
    df=df[df['STORE_MANAGER_GOAL']>0]
    logger.info("Selecting columns: GOAL_ID,STORE_NUMBER, BUSINESS_DATE, FINANCE_TEAM_GOAL, STORE_MANAGER_GOAL")
    df = df[
        [
            'GOAL_ID', #important for joining with other dataframes
            'STORE_NUMBER', 
            'BUSINESS_DATE', 
            'FINANCE_TEAM_GOAL', 
            'STORE_MANAGER_GOAL']]

    #output the dataframe to the output path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Dataframe saved to {output_path}")
    return df












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