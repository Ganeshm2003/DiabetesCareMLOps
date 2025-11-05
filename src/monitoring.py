import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import json
from pathlib import Path

# --- NEW EVIDENTLY API (>=0.4) ---
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DataDriftTable,
    TargetDrift,
)
from evidently.metric_preset import DataDriftPreset


class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame, model_version: str = "v1"):
        self.reference_data = reference_data
        self.model_version = model_version
        self.reports_dir = Path("monitoring/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_data_quality_report(self, current_data: pd.DataFrame) -> Report:
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=None,
        )
        return report

    def generate_target_drift_report(
        self,
        current_data: pd.DataFrame,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Report:
        ref = self.reference_data.copy()
        cur = current_data.copy()
        ref["prediction"] = reference_predictions
        cur["prediction"] = current_predictions

        column_mapping = ColumnMapping()
        column_mapping.prediction = "prediction"

        report = Report(metrics=[TargetDrift()])
        report.run(
            reference_data=ref,
            current_data=cur,
            column_mapping=column_mapping,
        )
        return report

    def save_report(self, report: Report, report_type: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{self.model_version}_{timestamp}.html"
        report.save_html(self.reports_dir / filename)

    def calculate_prediction_drift_metrics(
        self, reference_predictions: np.ndarray, current_predictions: np.ndarray
    ) -> Dict[str, float]:
        from scipy import stats

        ks_statistic, ks_pvalue = stats.ks_2samp(reference_predictions, current_predictions)

        def _js_div(p, q):
            p = np.asarray(p) + 1e-10
            q = np.asarray(q) + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            m = (p + q) / 2.0
            return (stats.entropy(p, m) + stats.entropy(q, m)) / 2.0

        hist_ref, _ = np.histogram(reference_predictions, bins=20, density=True)
        hist_cur, _ = np.histogram(current_predictions, bins=20, density=True)
        js_div = _js_div(hist_ref, hist_cur)

        return {"ks_statistic": ks_statistic, "ks_pvalue": ks_pvalue, "jensen_shannon_div": js_div}

    def detect_data_drift(
        self, current_data: pd.DataFrame, drift_threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, dict]]:
        from scipy import stats

        drift_metrics = {}
        significant_drift = False

        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue
            ref_col = self.reference_data[column].dropna()
            cur_col = current_data[column].dropna()
            if ref_col.empty or cur_col.empty:
                continue

            if pd.api.types.is_numeric_dtype(ref_col):
                stat, p = stats.ks_2samp(ref_col, cur_col)
                drift_metrics[column] = {"statistic": stat, "pvalue": p, "drift_detected": p < drift_threshold}
                if p < drift_threshold:
                    significant_drift = True
            else:
                contingency = pd.crosstab(
                    pd.concat([ref_col, cur_col]),
                    pd.Series(["ref"] * len(ref_col) + ["cur"] * len(cur_col))
                )
                chi2, p, *_ = stats.chi2_contingency(contingency)
                drift_metrics[column] = {"statistic": chi2, "pvalue": p, "drift_detected": p < drift_threshold}
                if p < drift_threshold:
                    significant_drift = True

        return significant_drift, drift_metrics

    def save_drift_metrics(self, metrics: Dict, prefix: str = "drift"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.model_version}_{timestamp}.json"
        with open(self.reports_dir / filename, "w") as f:
            json.dump(metrics, f, indent=2)

    def analyze_and_save_all(
        self,
        current_data: pd.DataFrame,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ):
        # 1. Data drift
        report = self.generate_data_quality_report(current_data)
        self.save_report(report, "data_drift")

        # 2. Target drift
        report = self.generate_target_drift_report(current_data, reference_predictions, current_predictions)
        self.save_report(report, "target_drift")

        # 3. Prediction drift
        pred_metrics = self.calculate_prediction_drift_metrics(reference_predictions, current_predictions)
        self.save_drift_metrics(pred_metrics, "prediction_drift")

        # 4. Feature drift
        drift_detected, drift_metrics = self.detect_data_drift(current_data)
        self.save_drift_metrics(drift_metrics, "feature_drift")

        return {
            "drift_detected": drift_detected,
            "prediction_metrics": pred_metrics,
            "drift_metrics": drift_metrics,
        }
