"""
Evaluation: measure each model on more than accuracy, and draw the charts.

Metrics used (all for the "has heart disease" class):
- Accuracy    : share of all patients classified correctly.
- Precision   : of the patients we flagged as sick, how many really were.
- Recall      : of the patients who really were sick, how many we caught.
                (The most important one here - a missed sick patient is the worst error.)
- Specificity : of the healthy patients, how many we correctly cleared.
- F1 score    : a single balance of precision and recall.
- ROC-AUC     : how well the model ranks sick patients above healthy ones (1.0 = perfect).
"""
import matplotlib

matplotlib.use("Agg")  # draw charts to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from . import config
from .preprocessing import readable_feature_names

sns.set_theme(style="whitegrid")
MODEL_COLORS = {
    "logistic_regression": "#4C72B0",
    "random_forest": "#55A868",
    "xgboost": "#C44E52",
}


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def _name(key):
    return config.MODEL_DISPLAY_NAMES.get(key, key)


def plot_confusion_matrices(results: dict, path):
    """One confusion matrix per model, side by side."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.3))
    for ax, (key, res) in zip(np.atleast_1d(axes), results.items()):
        m = res["test_metrics"]
        cm = np.array([[m["true_negatives"], m["false_positives"]],
                       [m["false_negatives"], m["true_positives"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 16},
                    xticklabels=["No disease", "Disease"],
                    yticklabels=["No disease", "Disease"])
        ax.set_title(_name(key), fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the test set", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(results: dict, y_test, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for key, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["test_prob"])
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS.get(key),
                label=f"{_name(key)} (AUC = {res['test_metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate (healthy patients wrongly flagged)")
    ax.set_ylabel("True positive rate (sick patients caught)")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(metrics_df: pd.DataFrame, path):
    """Grouped bar chart: every metric, every model."""
    cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    long = metrics_df.reset_index().melt(id_vars="model", value_vars=cols,
                                         var_name="metric", value_name="score")
    long["model"] = long["model"].map(_name)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="metric", y="score", hue="model", ax=ax,
                palette=[MODEL_COLORS[k] for k in metrics_df.index])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("")
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison across metrics")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
              fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_importance_table(pipeline) -> pd.DataFrame:
    """How much each input matters to a trained model."""
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = readable_feature_names(pre)
    if hasattr(model, "coef_"):  # Logistic Regression: size of each coefficient
        values = model.coef_[0]
        df = pd.DataFrame({"feature": names, "importance": np.abs(values), "coefficient": values})
    else:                          # tree models: built-in importance
        df = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    df["importance"] = df["importance"] / df["importance"].sum()
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(results: dict, path, top_n: int = 10):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    for ax, (key, res) in zip(np.atleast_1d(axes), results.items()):
        imp = res["feature_importance"].head(top_n).iloc[::-1]
        ax.barh(imp["feature"], imp["importance"], color=MODEL_COLORS.get(key))
        ax.set_title(_name(key), fontsize=12)
        ax.set_xlabel("Relative importance")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"Top {top_n} most important inputs per model", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_missing_values(report: pd.DataFrame, path):
    data = report[report["missing_percent"] > 0].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#C44E52" if p > config.MISSING_DROP_THRESHOLD * 100 else "#4C72B0"
              for p in data["missing_percent"]]
    ax.barh(data.index, data["missing_percent"], color=colors)
    ax.axvline(config.MISSING_DROP_THRESHOLD * 100, ls="--", color="grey")
    ax.text(config.MISSING_DROP_THRESHOLD * 100 + 1, 0, "drop threshold", color="grey", fontsize=9)
    for y, p in enumerate(data["missing_percent"]):
        ax.text(p + 0.5, y, f"{p:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% of values missing")
    ax.set_title("Missing values per column (red = dropped)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_balance(y, path):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["No disease", "Disease"], counts.values, color=["#55A868", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f"{v} ({v / counts.sum():.0%})", ha="center")
    ax.set_ylabel("Number of patients")
    ax.set_title("Target class balance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
