from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from smart_goals.config import FEATURE_CONFIG

set_config(transform_output="pandas")


def _build_column_transformer() -> ColumnTransformer:
    target_encode_cols = FEATURE_CONFIG.encoder_cols("target")
    onehot_cols = FEATURE_CONFIG.encoder_cols("onehot")
    numeric_cols = FEATURE_CONFIG.encoder_cols("numeric")

    transformers = [
        (
            "target_enc",
            TargetEncoder(smooth="auto", cv=3, shuffle=False),
            target_encode_cols,
        ),
        ("num", "passthrough", numeric_cols),
    ]

    if onehot_cols:
        transformers.insert(
            2,
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_cols),
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("encode", _build_column_transformer()),
        ]
    )
