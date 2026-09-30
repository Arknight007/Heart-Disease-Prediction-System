"""
The three models we compare, each with a small set of settings to try.

- Logistic Regression: simple, fast and easy to explain -> our baseline.
- Random Forest: many decision trees voting together.
- XGBoost: trees built one after another, each fixing the last one's mistakes.

Docs:
  Scikit-learn: https://scikit-learn.org/stable/supervised_learning.html
  XGBoost:      https://xgboost.readthedocs.io
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .preprocessing import build_preprocessor


def get_model_specs() -> dict:
    """Return {model_name: (estimator, settings_to_try)}."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
            {
                # C controls how strongly the model is kept simple (smaller = simpler).
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=config.RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + model in one object, so the app can feed it raw patient data."""
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
