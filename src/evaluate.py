import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

def compute_metrics(y_true, y_prob):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob))
    }

def roc_fig(y_true, y_prob):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    return fig

def pr_fig(y_true, y_prob):
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Precision-Recall Curve")
    return fig
