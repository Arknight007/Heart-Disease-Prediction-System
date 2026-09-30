"""
Train, compare and save the three heart disease models.

Run from the project folder:
    python train.py              # train from data/heart_disease_uci.csv
    python train.py --to-db      # also store the raw dataset in the database
    python train.py --from-db    # read the dataset from the database instead of the CSV

Outputs:
    models/   - one saved pipeline per model + best_model.joblib + model_info.json
    reports/  - metrics tables and charts (confusion matrices, ROC curves, ...)
"""
import argparse
import json
import warnings
from datetime import datetime

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import config, database
from src.data import clean_data, load_raw_data, missing_value_report, split_features_target
from src.evaluate import (
    compute_metrics, feature_importance_table, plot_class_balance,
    plot_confusion_matrices, plot_feature_importance, plot_metric_comparison,
    plot_missing_values, plot_roc_curves,
)
from src.models import build_pipeline, get_model_specs

warnings.filterwarnings("ignore", category=UserWarning)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main(from_db: bool = False, to_db: bool = False):
    # ------------------------------------------------------------------ 1. Load
    section("1. Loading data")
    if from_db:
        raw = database.read_dataset_from_db()
        print(f"Read {len(raw)} rows from the database ({database.backend_name()})")
    else:
        raw = load_raw_data()
        print(f"Read {len(raw)} rows, {raw.shape[1]} columns from {config.DATA_PATH.name}")
    if to_db:
        n = database.load_dataset_to_db(raw)
        database.init_db()
        print(f"Stored {n} rows in table '{database.RAW_TABLE}' ({database.backend_name()})")

    # ---------------------------------------------------- 2. Missing values
    section("2. Checking missing values")
    report = missing_value_report(raw)
    print(report[report["missing_count"] > 0].to_string())
    report.to_csv(config.REPORTS_DIR / "missing_values.csv")
    plot_missing_values(report, config.REPORTS_DIR / "missing_values.png")

    # ------------------------------------------------------------ 3. Clean
    section("3. Cleaning")
    df = clean_data(raw)
    X, y = split_features_target(df)
    print(f"Clean data: {len(df)} patients, {X.shape[1]} input features")
    print(f"Numeric features    : {config.NUMERIC_FEATURES}")
    print(f"Categorical features: {config.CATEGORICAL_FEATURES}")
    print(f"Target balance      : {y.value_counts().to_dict()}  (1 = heart disease)")
    remaining = X.isna().sum()
    print("Values still missing (filled in by the pipeline):",
          remaining[remaining > 0].to_dict())
    plot_class_balance(y, config.REPORTS_DIR / "class_balance.png")

    # ------------------------------------------------------------ 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    print(f"Train: {len(X_train)} patients | Test: {len(X_test)} patients (stratified)")

    # -------------------------------------------------- 5. Train each model
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results = {}
    for key, (estimator, grid) in get_model_specs().items():
        section(f"4. Training {config.MODEL_DISPLAY_NAMES[key]}")
        search = GridSearchCV(
            build_pipeline(estimator), grid, cv=cv, scoring=SCORING,
            refit="roc_auc",  # choose settings by ROC-AUC (does not depend on a cut-off)
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        i = search.best_index_
        cv_metrics = {m: float(search.cv_results_[f"mean_test_{m}"][i]) for m in SCORING}
        cv_std = {m: float(search.cv_results_[f"std_test_{m}"][i]) for m in SCORING}

        prob = best.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        test_metrics = compute_metrics(y_test, pred, prob)

        print("Best settings:", {k.replace("model__", ""): v for k, v in search.best_params_.items()})
        print("Cross-validation (train):",
              {m: f"{cv_metrics[m]:.3f} ± {cv_std[m]:.3f}" for m in SCORING})
        print("Test set:", {m: round(test_metrics[m], 3)
                            for m in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]})
        print(classification_report(y_test, pred, target_names=["No disease", "Disease"], digits=3))

        joblib.dump(best, config.MODELS_DIR / f"{key}.joblib")
        results[key] = {
            "pipeline": best,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv_metrics": cv_metrics,
            "cv_std": cv_std,
            "test_metrics": test_metrics,
            "test_prob": prob,
            "feature_importance": feature_importance_table(best),
            "classification_report": classification_report(
                y_test, pred, target_names=["No disease", "Disease"], digits=3),
        }

    # --------------------------------------------------------- 6. Compare
    section("5. Comparing the models")
    test_df = pd.DataFrame({k: r["test_metrics"] for k, r in results.items()}).T
    test_df.index.name = "model"
    cv_df = pd.DataFrame({k: r["cv_metrics"] for k, r in results.items()}).T
    cv_df.index.name = "model"
    cv_df.columns = [f"cv_{c}" for c in cv_df.columns]
    summary = test_df.join(cv_df)
    summary.to_csv(config.REPORTS_DIR / "model_comparison.csv", float_format="%.4f")

    show_cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc",
                 "false_negatives", "false_positives", "cv_f1", "cv_recall"]
    print(summary[show_cols].rename(index=config.MODEL_DISPLAY_NAMES).round(3).to_string())

    # The final model is picked on CROSS-VALIDATION scores from the training data,
    # not on the test set, so the test set stays a fair, untouched final check.
    # F1 is used because it balances catching sick patients (recall) with not
    # raising too many false alarms (precision); recall breaks any tie.
    ranking = cv_df.sort_values(["cv_f1", "cv_recall"], ascending=False)
    best_key = ranking.index[0]
    print(f"\nSelected model: {config.MODEL_DISPLAY_NAMES[best_key]} "
          f"(highest cross-validated F1 = {ranking.loc[best_key, 'cv_f1']:.3f})")
    joblib.dump(results[best_key]["pipeline"], config.MODELS_DIR / "best_model.joblib")

    # ------------------------------------------------------ 7. Save reports
    plot_confusion_matrices(results, config.REPORTS_DIR / "confusion_matrices.png")
    plot_roc_curves(results, y_test, config.REPORTS_DIR / "roc_curves.png")
    plot_metric_comparison(test_df, config.REPORTS_DIR / "metric_comparison.png")
    plot_feature_importance(results, config.REPORTS_DIR / "feature_importance.png")
    for key, r in results.items():
        r["feature_importance"].to_csv(config.REPORTS_DIR / f"feature_importance_{key}.csv", index=False)
        (config.REPORTS_DIR / f"classification_report_{key}.txt").write_text(r["classification_report"])

    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best_key,
        "selection_rule": "highest cross-validated F1 on the training set (recall breaks ties)",
        "n_patients": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "decision_threshold": 0.5,
        "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                             "pandas": pd.__version__},
        "models": {
            k: {
                "best_params": r["best_params"],
                "cv_metrics": r["cv_metrics"],
                "cv_std": r["cv_std"],
                "test_metrics": r["test_metrics"],
            } for k, r in results.items()
        },
    }
    (config.MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2, default=float))
    print(f"\nSaved models to {config.MODELS_DIR} and charts/tables to {config.REPORTS_DIR}")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-db", action="store_true", help="read the dataset from the database")
    parser.add_argument("--to-db", action="store_true", help="store the raw dataset in the database")
    args = parser.parse_args()
    main(from_db=args.from_db, to_db=args.to_db)
