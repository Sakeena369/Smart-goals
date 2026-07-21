# Smart Goals Snowflake Loader

This utility uploads CSVs from the `raw-data/` folder into Snowflake tables using the Snowflake Snowpark Python library.

## Prerequisites

- Python 3.9+
- Network access to your Snowflake account
- Environment variables set:
  - `SNOWFLAKE_ACCOUNT`
  - `SNOWFLAKE_USER`
  - `SNOWFLAKE_PASSWORD`
  - `SNOWFLAKE_ROLE`
  - `SNOWFLAKE_WAREHOUSE`
  - `SNOWFLAKE_DATABASE`
  - `SNOWFLAKE_SCHEMA`

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Dry-run (no Snowflake connection)

```bash
python src/upload_raw_data.py --dry-run --raw-dir raw-data
```

## Upload data

Append to tables (create if needed):

```bash
python src/upload_raw_data.py --raw-dir raw-data --mode append
```

Overwrite tables:

```bash
python src/upload_raw_data.py --raw-dir raw-data --mode overwrite
```

Optionally, set a prefix for table names (e.g., `RAW_`):

```bash
python src/upload_raw_data.py --raw-dir raw-data --mode append --table-prefix RAW
```

## Notes

- Table names are derived from file names and sanitized to Snowflake-friendly uppercase identifiers. Column names are also sanitized to uppercase identifiers.
- For large files, using stages + `COPY INTO` may be more efficient. This tool uses Snowpark to create DataFrames from local CSVs for simplicity and portability.
