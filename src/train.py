import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import sys
# Fix relative imports when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import dagshub

# ---------- DagsHub + MLflow ----------
dagshub.init(repo_owner="Ganeshm2003", repo_name="DiabetesCareMLOps", mlflow=True)

# Use GitHub Secrets (set in repo → Settings → Secrets)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

MLFLOW_EXPERIMENT = "hospital-readmission"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# ---------- Imports from src ----------
from src.data_preprocessing import build_preprocessor, enrich_and_clean
from src.evaluate import compute_metrics, roc_fig, pr_fig
from src.utils import map_readmitted


class ReadmissionPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input):
        return self.pipeline.predict_proba(model_input)[:, 1]


def maybe_load_map(path: str, key_col: str, desc_col: str):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df = df[[key_col, desc_col]].dropna()
        return df
    return None


def main(args):
    df = pd.read_csv(args.data)

    # binary target
    df["readmitted_30"] = df["readmitted"].apply(map_readmitted)

    # optional mapping lookups
    adm_type_map = maybe_load_map(args.adm_type_map, "admission_type_id", "description")
    disch_map = maybe_load_map(args.discharge_map, "discharge_disposition_id", "description")
    adm_src_map = maybe_load_map(args.adm_src_map, "admission_source_id", "description")

    df = enrich_and_clean(df, adm_type_map, disch_map, adm_src_map)
    target_col = "readmitted_30"
    df = df.dropna(subset=[target_col])

    # class imbalance handling
    pos = (df[target_col] == 1).sum()
    neg = (df[target_col] == 0).sum()
    spw = max((neg / max(pos, 1)), 1.0)

    pre, num_cols, cat_cols = build_preprocessor(df, target_col)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=spw,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="xgb_uci_diabetes") as run:
        # params
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("scale_pos_weight", float(spw))
        mlflow.log_param("n_estimators", clf.n_estimators)
        mlflow.log_param("max_depth", clf.max_depth)
        mlflow.log_param("learning_rate", clf.learning_rate)
        mlflow.log_param("dataset_file", args.data)

        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_va)[:, 1]
        metrics = compute_metrics(y_va, y_prob)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_figure(roc_fig(y_va, y_prob), "figures/roc.png")
        mlflow.log_figure(pr_fig(y_va, y_prob), "figures/pr.png")

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ReadmissionPyfuncModel(pipe),
            registered_model_name=args.register if args.register else None,
            pip_requirements="requirements.txt",
        )

        # extra artifacts
        joblib.dump(pipe, "../readmission_model.joblib")
        mlflow.log_artifact("../readmission_model.joblib")
        mlflow.log_artifact(args.data)

        # console output
        accuracy = accuracy_score(y_va, (y_prob > 0.5).astype(int))
        print(f"Hospital readmission model logged! Accuracy: {accuracy:.2f}")
        if "f1" in metrics:
            print(f"F1: {metrics['f1']:.2f}")
        if "auc" in metrics:
            print(f"AUC: {metrics['auc']:.2f}")
        print(f"[run_id] {run.info.run_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--adm_type_map", default=None)
    p.add_argument("--discharge_map", default=None)
    p.add_argument("--adm_src_map", default=None)
    p.add_argument("--register", default="hospital_readmission")
    args = p.parse_args()
    main(args)
