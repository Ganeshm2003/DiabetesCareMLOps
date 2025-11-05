import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import json
from pathlib import Path

# ----------------------------------------------------------------------
#  Evidently import – works with both new (0.4+) and old (<0.4) versions
# ----------------------------------------------------------------------
try:
    # New API (>=0.4)
    from evidently import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    from evidently.metrics import *
except ImportError:                     # pragma: no cover
    # Old API (<0.4)
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    from evidently.metrics import *


class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame, model_version: str):
        """Initialize monitor with reference data (training set)"""
        self.reference_data = reference_data
        self.model_version = model_version
        self.reports_dir = Path("monitoring/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    #  Data‑quality + drift report
    # ------------------------------------------------------------------
    def generate_data_quality_report(self, current_data: pd.DataFrame) -> Report:
        report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=current_data)
        return report

    # ------------------------------------------------------------------
    #  Target (prediction) drift report
    # ------------------------------------------------------------------
    def generate_target_drift_report(
        self,
        current_data: pd.DataFrame,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Report:
        report = Report(metrics=[TargetDriftPreset()])

        ref = self.reference_data.copy()
        cur = current_data.copy()

        ref["prediction"] = reference_predictions
        cur["prediction"] = current_predictions

        # column_mapping tells Evidently which column is the “target”
        report.run(
            reference_data=ref,
            current_data=cur,
            column_mapping={"target": "prediction"},
        )
        return report

    # ------------------------------------------------------------------
    #  Save HTML report
    # ------------------------------------------------------------------
    def save_report(self, report: Report, report_type: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{self.model_version}_{timestamp}.html"
        report.save_html(self.reports_dir / filename)

    # ------------------------------------------------------------------
    #  Prediction‑drift statistical metrics
    # ------------------------------------------------------------------
    def calculate_prediction_drift_metrics(
        self, reference_predictions: np.ndarray, current_predictions: np.ndarray
    ) -> Dict[str, float]:
        from scipy import stats

        # KS test
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_predictions, current_predictions)

        # Jensen‑Shannon divergence (histogram based)
        def _js_div(p, q):
            p = np.asarray(p)
            q = np.asarray(q)
            p = p / p.sum()
            q = q / q.sum()
            m = (p + q) / 2.0
            return (stats.entropy(p, m) + stats.entropy(q, m)) / 2.0

        hist_ref, _ = np.histogram(reference_predictions, bins=20, density=True)
        hist_cur, _ = np.histogram(current_predictions, bins=20, density=True)
        js_div = _js_div(hist_ref, hist_cur)

        return {"ks_statistic": ks_statistic, "ks_pvalue": ks_pvalue, "jensen_shannon_div": js_div}

    # ------------------------------------------------------------------
    #  Feature‑level drift detection
    # ------------------------------------------------------------------
    def detect_data_drift(
        self, current_data: pd.DataFrame, drift_threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, dict]]:
        from scipy import stats

        drift_metrics: Dict[str, dict] = {}
        significant_drift = False

        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue

            ref_col = self.reference_data[column].dropna()
            cur_col = current_data[column].dropna()

            if ref_col.empty or cur_col.empty:
                continue

            if pd.api.types.is_numeric_dtype(ref_col):
                # KS test for numeric
                statistic, pvalue = stats.ks_2samp(ref_col, cur_col)
                drift_metrics[column] = {
                    "statistic": statistic,
                    "pvalue": pvalue,
                    "drift_detected": pvalue < drift_threshold,
                }
                if pvalue < drift_threshold:
                    significant_drift = True
            else:
                # Chi‑square for categorical
                contingency = pd.crosstab(
                    pd.concat([ref_col, cur_col]),
                    pd.Series(
                        ["reference"] * len(ref_col) + ["current"] * len(cur_col)
                    ),
                )
                chi2, pvalue, *_ = stats.chi2_contingency(contingency)
                drift_metrics[column] = {
                    "statistic": chi2,
                    "pvalue": pvalue,
                    "drift_detected": pvalue < drift_threshold,
                }
                if pvalue < drift_threshold:
                    significant_drift = True

        return significant_drift, drift_metrics

    # ------------------------------------------------------------------
    #  Save JSON metrics
    # ------------------------------------------------------------------
    def save_drift_metrics(self, metrics: Dict, prefix: str = "drift_metrics"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.model_version}_{timestamp}.json"
        with open(self.reports_dir / filename, "w") as f:
            json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    #  One‑call orchestrator
    # ------------------------------------------------------------------
    def analyze_and_save_all(
        self,
        current_data: pd.DataFrame,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ):
        # 1. Data quality + drift
        quality_report = self.generate_data_quality_report(current_data)
        self.save_report(quality_report, "data_quality")

        # 2. Target drift
        target_report = self.generate_target_drift_report(
            current_data, reference_predictions, current_predictions
        )
        self.save_report(target_report, "target_drift")

        # 3. Prediction drift metrics
        pred_metrics = self.calculate_prediction_drift_metrics(
            reference_predictions, current_predictions
        )
        self.save_drift_metrics(pred_metrics, "prediction_drift")

        # 4. Feature drift
        drift_detected, drift_metrics = self.detect_data_drift(current_data)
        self.save_drift_metrics(drift_metrics, "feature_drift")

        return {
            "drift_detected": drift_detected,
            "prediction_metrics": pred_metrics,
            "drift_metrics": drift_metrics,
        }
