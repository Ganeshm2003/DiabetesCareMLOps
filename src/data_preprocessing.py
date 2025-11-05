import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Optional  # ← ADD THIS
from .utils import age_midpoint, icd_to_chapter, clean_special_strings


def enrich_and_clean(
    df: pd.DataFrame,
    adm_type_map: Optional[pd.DataFrame] = None,   # ← FIXED
    disch_map: Optional[pd.DataFrame] = None,      # ← FIXED
    adm_src_map: Optional[pd.DataFrame] = None,    # ← FIXED
) -> pd.DataFrame:
    df = df.copy()

    # uniform missing / special tokens
    for c in df.columns:
        df[c] = df[c].apply(clean_special_strings)

    # join id→description (optional)
    if adm_type_map is not None and "admission_type_id" in df:
        df = df.merge(
            adm_type_map.rename(columns={"description": "admission_type_desc"}),
            how="left",
            left_on="admission_type_id",
            right_on="admission_type_id",
        )
    if disch_map is not None and "discharge_disposition_id" in df:
        df = df.merge(
            disch_map.rename(columns={"description": "discharge_disposition_desc"}),
            how="left",
            left_on="discharge_disposition_id",
            right_on="discharge_disposition_id",
        )
    if adm_src_map is not None and "admission_source_id" in df:
        df = df.merge(
            adm_src_map.rename(columns={"description": "admission_source_desc"}),
            how="left",
            left_on="admission_source_id",
            right_on="admission_source_id",
        )

    # age bucket → numeric midpoint
    if "age" in df:
        df["age_years"] = df["age"].apply(age_midpoint)

    # ICD chapters for diagnosis columns
    for dcol in ["diag_1", "diag_2", "diag_3"]:
        if dcol in df:
            df[f"{dcol}_chapter"] = df[dcol].apply(icd_to_chapter)

    # casting numeric-ish columns
    numeric_like = [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses", "weight"
    ]
    for c in numeric_like:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # tidy feature set
    drop_cols = [
        "encounter_id", "patient_nbr", "payer_code", "medical_specialty",
        "age", "diag_1", "diag_2", "diag_3"
    ]
    for c in drop_cols:
        if c in df:
            df.drop(columns=c, inplace=True)

    return df


def build_preprocessor(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor, num_cols, cat_cols
