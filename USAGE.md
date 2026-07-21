# Smart Goals Model Training and Inference

This package provides end-to-end data fetching, training, and inference for sales goal prediction using LightGBM with Optuna hyperparameter optimization and MLflow tracking.

## Features

- **Data Fetching**: Automated data extraction and processing from Snowflake
- **Snowflake Integration**: Load data directly from Snowflake using Snowpark
- **Optuna Optimization**: Automated hyperparameter tuning with time series cross-validation
- **MLflow Tracking**: Full experiment tracking and model versioning
- **Time Series CV**: Proper temporal validation using TimeSeriesSplit
- **Target Encoding**: Categorical feature encoding with sklearn/category-encoders
- **Production Ready**: Includes data fetching, training, and inference pipelines

## Installation

```bash
# Install package with dependencies
pip install -e .

# Or install dependencies separately
pip install lightgbm mlflow optuna category-encoders scikit-learn snowflake-snowpark-python[pandas]
```

## Quick Start

### 0. Fetch Data from Snowflake

Before training, you can fetch and process the associate goals data:

```bash
# Fetch data with default settings (saves to data/associate_goals_df.csv)
python -m smart_goals.fetch_data

# Customize fetch parameters
python -m smart_goals.fetch_data \
    --analysis-start-date "2023-01-01" \
    --associate-tenure-start-date "2020-01-01" \
    --goal-start-date "2025-11-01" \
    --output-path "/path/to/output.csv" \
    --format csv
```

Python API:

```python
from smart_goals import fetch_associate_goals_data

# Fetch and process data
df = fetch_associate_goals_data(
    analysis_start_date='2023-01-01',
    associate_tenure_start_date='2020-01-01',
    goal_start_date='2025-11-01',
    output_path='data/associate_goals_df.csv',
    save_format='csv'  # Options: 'csv', 'feather', 'parquet'
)

print(f"Fetched {len(df):,} records")
print(f"Columns: {list(df.columns)}")
```

### 1. Set up Snowflake Connection

Ensure your Snowflake credentials are configured. The scripts use `Session.builder.getOrCreate()` which reads from:
- Environment variables
- `~/.snowflake/connections.toml`
- Snowflake config file

### 2. Start MLflow Server (Optional)

```bash
# Start local MLflow server
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

### 3. Train Model

```bash
# Basic training (uses default MLflow tracking)
python -m smart_goals.train \
    --table-name "your_schema.associate_goals_regression"

# With MLflow server
python -m smart_goals.train \
    --table-name "your_schema.associate_goals_regression" \
    --mlflow-uri "http://127.0.0.1:5000" \
    --filters "BUSINESS_DATE >= '2025-01-01'"
```

### 4. Make Predictions

```bash
# Batch predictions
python -m smart_goals.predict \
    --run-id "your_mlflow_run_id" \
    --input-table "your_schema.new_data" \
    --output-table "your_schema.predictions" \
    --mlflow-uri "http://127.0.0.1:5000"
```

## Python API Usage

### Training

```python
from smart_goals.train import main

# Run training pipeline
model, metrics, best_params = main(
    table_name="your_schema.associate_goals_regression",
    filters="BUSINESS_DATE >= '2025-01-01'",
    mlflow_tracking_uri="http://127.0.0.1:5000"
)

print(f"Validation RMSE: {metrics['valid_rmse']:.4f}")
print(f"Best parameters: {best_params}")
```

### Inference

```python
from smart_goals.predict import predict_from_snowflake, load_model_from_mlflow
from smart_goals.train import create_preprocessing_pipeline

# Load model
model = load_model_from_mlflow(run_id="your_run_id")

# Create preprocessing pipeline
calendar_tf, preprocess = create_preprocessing_pipeline()

# Make predictions
predictions_df = predict_from_snowflake(
    model=model,
    calendar_tf=calendar_tf,
    preprocess=preprocess,
    table_name="your_schema.new_data",
    output_table="your_schema.predictions"
)
```

## Required Data Columns

The training data must include these columns:
- `BUSINESS_DATE`: Date of the transaction
- `SALES_ASSOCIATE`: Associate identifier
- `STORE_NUMBER`: Store identifier
- `NET_SALES_AMOUNT_DLABS`: Net sales amount
- `GOALS_SCHEDULED_HOURS_CORRECTED`: Scheduled hours (corrected)
- `STORE_MANAGER_GOAL`: Store-level goal
- `HOURS_SINCE_OPEN_CORRECTED`: Hours since store opening
- `TENURE_DAYS`: Associate tenure in days

> **Note**: The `fetch_data.py` script automatically generates a dataset with all required columns plus additional features. The output includes all necessary columns for training, such as:
> - Sales data from multiple sources (DLABS and Zipline)
> - Associate schedules and shift information
> - Store operating hours
> - Calculated features (tenure, shift duration, time of day)
> - Date features (month, day of week, week of year)

Full column list from `fetch_data.py`:
```
SALES_ASSOCIATE, BUSINESS_DATE, STORE_NUMBER, NET_SALES_AMOUNT_DLABS, 
GROSS_SALES_AMOUNT_DLABS, FINANCE_TEAM_GOAL, STORE_MANAGER_GOAL, 
LEGION_SCHEDULED_HOURS, GOALS_SCHEDULED_HOURS, GOAL, NET_SALES_AMOUNT_ZIPLINE, 
SHIFT_TYPE, SHIFT_START_HOUR, SHIFT_END_HOUR, OPEN_HOUR, CLOSE_HOUR, 
HOURS_SINCE_OPEN, HOURS_SINCE_OPEN_CORRECTED, TIME_OF_DAY, FIRST_SELLING_DAY, 
MONTH, DAY_OF_WEEK, WEEK_OF_YEAR, TENURE_DAYS, SHIFT_DURATION, MAX_POSSIBLE_HOURS, 
GOALS_SCHEDULED_HOURS_CORRECTED, CHNL_DESC, REGION_DESC, DISTRICT_DESC
```

## Configuration

### Optuna Settings

Edit in `train.py`:
```python
N_TRIALS = 50  # Number of hyperparameter trials
N_SPLITS = 5   # Number of time series CV splits
```

### Feature Engineering

The pipeline creates:
- **LogTenure**: `log(1 + TENURE_DAYS)`
- **Cyclical Date Features**: Sin/cos transformations for day of week, month, week of year
- **Target Encoding**: For categorical variables (Associate, Store)
- **Count Encoding**: Frequency-based encoding for categories

### Target Variable

```python
SalesPerGoalHour = NET_SALES_AMOUNT_DLABS / GOALS_SCHEDULED_HOURS_CORRECTED
```

## MLflow Tracking

The training pipeline logs:
- All hyperparameters (both Optuna search space and final model)
- Metrics: RMSE, MAE, R² for train and validation sets
- Artifacts: 
  - Trained model
  - Optuna trial history
  - Feature importance
  - Input example

### Viewing Results

```bash
# Open MLflow UI
mlflow ui --port 5000

# Or if using remote server
# Navigate to your MLflow server URL
```

## Advanced Usage

### Custom Optuna Search Space

Edit the `objective()` function in `train.py`:

```python
params = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 5000),
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
    # Add or modify parameters
}
```

### Time Series Split Configuration

```python
# Change number of CV folds
tscv = TimeSeriesSplit(n_splits=10)  # More splits = smaller validation sets
```

### Adding Custom Features

Edit `add_calendar_features()` function:

```python
def add_calendar_features(X_in: pd.DataFrame) -> pd.DataFrame:
    Xo = X_in.copy()
    
    # Add your custom features
    Xo["is_weekend"] = Xo["BUSINESS_DATE"].dt.dayofweek >= 5
    Xo["is_holiday"] = ...  # Your logic
    
    return Xo
```

## Model Performance

The model is evaluated using:
- **RMSE**: Root Mean Squared Error (primary metric)
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

Time series cross-validation ensures the model generalizes to future dates.

## Troubleshooting

### Snowflake Connection Issues

```python
# Test connection
from snowflake.snowpark import Session
session = Session.builder.getOrCreate()
print(session.sql("SELECT CURRENT_USER()").collect())
```

### MLflow Server Not Found

```bash
# Check if server is running
curl http://127.0.0.1:5000

# Or use local file-based tracking
export MLFLOW_TRACKING_URI=file:///path/to/mlruns
```

### Memory Issues with Large Datasets

```python
# Add sampling in prepare_data()
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

## Production Deployment

### 1. Save Preprocessing Pipeline

The current implementation recreates the preprocessing pipeline. For production:

```python
import joblib

# After training
joblib.dump(calendar_tf, "calendar_transformer.pkl")
joblib.dump(preprocess, "preprocessor.pkl")

# Log as MLflow artifacts
mlflow.log_artifact("calendar_transformer.pkl")
mlflow.log_artifact("preprocessor.pkl")
```

### 2. Register Model

```python
# Register model in MLflow Model Registry
mlflow.register_model(
    f"runs:/{run_id}/model",
    "smart_goals_model"
)

# Transition to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="smart_goals_model",
    version=1,
    stage="Production"
)
```

### 3. Scheduled Batch Predictions

```bash
# Add to cron or orchestrator (Airflow, Prefect, etc.)
0 2 * * * python -m smart_goals.predict \
    --run-id <run_id> \
    --input-table daily_data \
    --output-table daily_predictions
```

## License

[Your License]

## Contact

[Your Contact Info]
