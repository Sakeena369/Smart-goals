from dataclasses import dataclass
from typing import Literal

Encoder = Literal["target", "onehot", "numeric", "drop"]
CastDtype = Literal["category", "float64", "int64", "string"]

@dataclass(frozen=True)
class SnowflakeConfig:
    database: str = "DEV_DS"
    schema: str = "DBT_DWERNERSEXTON"
    warehouse: str = "DEV_WHDS"
    table: str = "ZIPLINE_ASSOCIATE_GOALS_DATA"

@dataclass(frozen=True)
class MlflowConfig:
    model_name: str = "smart_goals"
    uri: str = "https://mlflow.analytics-dev.tapestry.com"
    experiment_name: str = "smart_goals_training"


@dataclass(frozen=True)
class DataConfig:
    analysis_start_date: str = "2023-01-01"
    test_holdout_start_date: str = "2026-01-01"
    target_col: str = "SALES_PER_GOAL_HOUR"
    date_col: str = "BUSINESS_DATE"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    encoder: Encoder
    cast_dtype: CastDtype | None = None


@dataclass(frozen=True)
class FeatureConfig:
    specs: tuple[FeatureSpec, ...]

    @property
    def feature_cols(self) -> list[str]:
        return [spec.name for spec in self.specs]

    @property
    def cast_dtypes(self) -> dict[str, CastDtype]:
        return {
            spec.name: spec.cast_dtype for spec in self.specs if spec.cast_dtype is not None
        }

    def encoder_cols(self, encoder: Encoder) -> list[str]:
        return [spec.name for spec in self.specs if spec.encoder == encoder]

MLFLOW_CONFIG = MlflowConfig()
DATA_CONFIG = DataConfig()
SNOWFLAKE_CONFIG = SnowflakeConfig()
FEATURE_CONFIG = FeatureConfig(
    specs=(
    FeatureSpec("SALES_ASSOCIATE", encoder="target", cast_dtype="string"),
    FeatureSpec("STORE_NUMBER", encoder="target", cast_dtype="string"),
    FeatureSpec("CHNL_DESC", encoder="target", cast_dtype="string"),
    FeatureSpec("REGION_DESC", encoder="target", cast_dtype="string"),
    FeatureSpec("DAY_OF_WEEK", encoder="onehot", cast_dtype="string"),
    FeatureSpec("TENURE_MONTHS_CAPPED_2YR", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("LONG_TENURE_X_LOG_TENURE", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("TENURE_BUCKET", encoder="target", cast_dtype="string"),
    FeatureSpec("LOG_TENURE_MONTHS_CAPPED_2YR", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("HOURS_SINCE_OPEN_CORRECTED", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("STORE_MANAGER_GOAL", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("IS_HOLIDAY_SEASON", encoder="numeric", cast_dtype="boolean"),
    FeatureSpec("IS_WEEKEND", encoder="numeric", cast_dtype="boolean"),
    FeatureSpec("IS_LONG_TENURE", encoder="numeric", cast_dtype="boolean"),
    FeatureSpec("SPGH_2WKS_SMOOTH", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("SPGH_4WKS_SMOOTH", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("SPGH_8WKS_SMOOTH", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("SPGH_16WKS_SMOOTH", encoder="numeric", cast_dtype="float64"),
    FeatureSpec("SPGH_32WKS_SMOOTH", encoder="numeric", cast_dtype="float64"),
    ),
)
