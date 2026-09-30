"""
Preprocessing: fill remaining gaps, encode categories, scale numbers.

All of this lives inside a scikit-learn Pipeline, so the fill-in values and
scaling are learnt ONLY from the training data. This avoids "data leakage",
where information from the test set sneaks into training.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing step shared by all three models."""
    numeric_steps = Pipeline(steps=[
        # Missing numbers are filled with the median (robust to outliers).
        ("impute", SimpleImputer(strategy="median")),
        # Put every number on the same scale (mean 0, spread 1).
        ("scale", StandardScaler()),
    ])

    categorical_steps = Pipeline(steps=[
        # Missing categories are filled with the most common category.
        ("impute", SimpleImputer(strategy="most_frequent")),
        # One-hot encoding: each category becomes its own 0/1 column.
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_steps, config.NUMERIC_FEATURES),
        ("cat", categorical_steps, config.CATEGORICAL_FEATURES),
    ])


def readable_feature_names(preprocessor: ColumnTransformer) -> list:
    """Turn names like 'cat__cp_asymptomatic' into 'Chest pain type = asymptomatic'."""
    names = []
    for raw_name in preprocessor.get_feature_names_out():
        group, name = raw_name.split("__", 1)
        if group == "num":
            names.append(config.FEATURE_LABELS.get(name, name))
        else:
            # One-hot names look like 'cp_asymptomatic'; match the longest column prefix.
            col = max((c for c in config.CATEGORICAL_FEATURES if name.startswith(c + "_")), key=len)
            value = name[len(col) + 1:]
            names.append(f"{config.FEATURE_LABELS.get(col, col)} = {value}")
    return names
