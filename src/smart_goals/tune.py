"""
LightGBM training with Ray Tune (OptunaSearch) for parallel trials,
TimeSeriesSplit CV, and MLflow logging (nested trial runs + final model).
"""

import os
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from typing import Tuple, Dict, Any

from snowflake.snowpark import Session

from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import set_config

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from optuna.trial import Trial

from smart_goals.config import DATA_CONFIG, MLFLOW_CONFIG
from smart_goals.pipeline import preprocessing_pipeline
from smart_goals.data import load_data_from_snowflake, prepare_data
from smart_goals.viz import plot_diagnostics, plot_partial_dependence, plot_importance

set_config(transform_output="pandas")

TARGET_COL = DATA_CONFIG.target_col
DATE_COL = DATA_CONFIG.date_col
TEST_HOLDOUT_START_DATE = DATA_CONFIG.test_holdout_start_date
MLFLOW_EXPERIMENT_NAME = MLFLOW_CONFIG.experiment_name
MLFLOW_URL = MLFLOW_CONFIG.uri

N_TRIALS = 50
N_SPLITS = 3

# Ray resource controls
CPUS_PER_TRIAL = 2  # adjust based on preprocessing cost; keep LGBM n_jobs=1 regardless
MAX_CONCURRENT_TRIALS = (
    None  # optionally set an int; otherwise Ray schedules based on cluster resources
)


def build_param_space(trial: Trial) -> Dict[str, Any]:
    """
    Single source of truth for Optuna define-by-run search space.
    """
    objective = trial.suggest_categorical("objective", ["tweedie", "regression"])
    trial.suggest_int("n_estimators", 50, 3000)
    trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
    trial.suggest_int("num_leaves", 20, 150)
    trial.suggest_int("max_depth", 3, 12)
    trial.suggest_int("min_child_samples", 20, 500)
    trial.suggest_float("subsample", 0.5, 1.0)
    trial.suggest_float("colsample_bytree", 0.5, 1.0)
    trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    trial.suggest_float("min_split_gain", 0.0, 1.0)

    if objective == "tweedie":
        trial.suggest_float("tweedie_variance_power", 1.0, 2.0)

    return {
        "metric": "tweedie" if objective == "tweedie" else "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "subsample_freq": 1,
        "n_jobs": 1,
    }


def train_test_split_date(
    df: pd.DataFrame, split_date=TEST_HOLDOUT_START_DATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based train-test split.
    Train: all data before split_date
    Test: all data from split_date onward
    """
    train_df = df[df[DATE_COL] < pd.to_datetime(split_date)]
    test_df = df[df[DATE_COL] >= pd.to_datetime(split_date)]

    X_train = train_df.drop(columns=[TARGET_COL, DATE_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL, DATE_COL])
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test


def visualize(model: Pipeline, X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series):
    """
    Generates diagnostic plots for the given model and data.
    """
    clf = model.named_steps["regressor"]
    X_form = model.named_steps["preprocessing"].transform(X)
    
    [plot_partial_dependence(clf, X_form, feature) for feature in [
        "num__TENURE_MONTHS_CAPPED_2YR",
        "num__LOG_TENURE_MONTHS_CAPPED_2YR",
        "num__LOG_TENURE",
        "num__HOURS_SINCE_OPEN_CORRECTED",
    ]]

    # Feature importance plot
    plot_importance(clf, importance_type="gain", max_num_features=20)

    # Diagnostic plots
    plot_diagnostics(model, y_true, y_pred)


def build_model_pipeline(params: Dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("preprocessing", preprocessing_pipeline()),
            ("regressor", LGBMRegressor(**params)),
        ]
    )


def train(
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    X_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | None = None,
) -> Pipeline:
    model_pipeline = build_model_pipeline(params)
    model_pipeline.fit(X, y)

    mlflow.log_params(params)

    if X_eval is not None and y_eval is not None:
        y_pred = model_pipeline.predict(X_eval)
        test_rmse = root_mean_squared_error(y_eval, y_pred)
        mlflow.log_metric("test_rmse", test_rmse)
        visualize(model_pipeline, X_eval, y_eval, y_pred)

    return model_pipeline


def evaluate(params: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> float:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    model_pipeline = build_model_pipeline(params)

    # IMPORTANT: keep folds single-threaded; concurrency comes from parallel trials
    scores = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    return float(-np.mean(scores))


def objective(config: Dict[str, Any], X, y) -> None:
    """
    One Ray Tune trial.
    Logs params/metrics to a nested MLflow run under a parent run (if provided).
    """

    set_config(transform_output="pandas")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("ray_trial_id", tune.get_context().get_trial_id())
            mlflow.log_params(config)
            mean_rmse = evaluate(config, X, y)
            mlflow.log_metric("mean_cv_rmse", mean_rmse)

    tune.report({"mean_cv_rmse": mean_rmse})


def run_tune(
    X: pd.DataFrame, y: pd.Series, n_trials: int = N_TRIALS
) -> tune.ResultGrid:
    """
    Runs Ray Tune with OptunaSearch, returns best config and best metric.
    """

    # Put data in Ray object store once (avoid copying each trial)
    X_ref = ray.put(X)
    y_ref = ray.put(y)

    algo = OptunaSearch(
        space=build_param_space,
        metric="mean_cv_rmse",
        mode="min",
    )

    if MAX_CONCURRENT_TRIALS:
        algo = ConcurrencyLimiter(algo, max_concurrent=MAX_CONCURRENT_TRIALS)

    tune_config = tune.TuneConfig(
        search_alg=algo,
        num_samples=n_trials,
    )

    # Wrap trainable with fixed references
    trainable = tune.with_parameters(objective, X=X_ref, y=y_ref)
    trainable = tune.with_resources(trainable, resources={"cpu": CPUS_PER_TRIAL})

    tuner = tune.Tuner(
        trainable=trainable,
        tune_config=tune_config,
    )

    return tuner.fit()


def main():
        
    parser = argparse.ArgumentParser(
        description="Train LightGBM model with Ray Tune (OptunaSearch) and MLflow logging."
    )
    
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=MLFLOW_URL,
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of Ray Tune trials to run (default: {N_TRIALS})",
    )

    args = parser.parse_args()
    session = Session.builder.getOrCreate()
    df = load_data_from_snowflake(session)
    df = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split_date(
        df, split_date=TEST_HOLDOUT_START_DATE
    )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as parent_run:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_NAME
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id

        train_dataset = mlflow.data.from_pandas(  # ty:ignore[unresolved-attribute]
            X_train.assign(**{TARGET_COL: y_train.values}),
            name="smart_goals_train_split",
            targets=TARGET_COL,
        )
        test_dataset = mlflow.data.from_pandas(  # ty:ignore[unresolved-attribute]
            X_test.assign(**{TARGET_COL: y_test.values}),
            name="smart_goals_test_split",
            targets=TARGET_COL,
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="evaluation")

        ray.init(runtime_env={"working_dir": "./src"})

        # Parallel tuning (trials run concurrently on Ray)
        results = run_tune(X_train, y_train, n_trials=args.n_trials)
        best_result = results.get_best_result(metric="mean_cv_rmse", mode="min")
        # best_result.config["n_jobs"] = -1  # use all cores for final training 

        if best_result.config is None:
            raise ValueError("No successful trials with a valid config found.")

        best_model = train(
            best_result.config,  
            X_train,
            y_train,
            X_eval=X_test,
            y_eval=y_test,
        )

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            input_example=X_train.sample(10),
        )
        return best_model


if __name__ == "__main__":
    main()
