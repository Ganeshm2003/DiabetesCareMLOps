import numpy as np
import pandas as pd
import re

ICD_CHAPTERS = {
    "infectious": (1, 139),
    "neoplasms": (140, 239),
    "endocrine_metabolic": (240, 279),
    "blood": (280, 289),
    "mental": (290, 319),
    "nervous": (320, 389),
    "circulatory": (390, 459),
    "respiratory": (460, 519),
    "digestive": (520, 579),
    "genitourinary": (580, 629),
    "pregnancy": (630, 679),
    "skin": (680, 709),
    "musculoskeletal": (710, 739),
    "congenital": (740, 759),
    "perinatal": (760, 779),
    "ill_defined": (780, 799),
    "injury": (800, 999)
}

def age_midpoint(age_bucket: str) -> float:
    # e.g. "[60-70)" -> 65
    if isinstance(age_bucket, str):
        m = re.match(r"\[(\d+)-(\d+)\)", age_bucket.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (a + b) / 2.0
    return np.nan

def icd_to_chapter(icd: str) -> str:
    # dataset has numeric strings or 'V'/'E' codes
    if not isinstance(icd, str) or icd.strip() in ("?", ""):
        return "unknown"
    s = icd.strip()
    if s[0] in ("V", "v"):
        return "supplemental_v"
    if s[0] in ("E", "e"):
        return "supplemental_e"
    try:
        x = float(s.split(".")[0])
    except Exception:
        return "unknown"
    for name, (lo, hi) in ICD_CHAPTERS.items():
        if lo <= x <= hi:
            return name
    return "other"

def clean_special_strings(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x in {"?", "NULL", "Not Available"}:
            return np.nan
        if x.lower() in {"none", "no"}:  # keep as string (categorical) for meds/max_glu/A1C
            return x.capitalize() if x.lower()=="none" else "No"
    return x

def map_readmitted(label: str) -> int:
    # 30-day target
    if not isinstance(label, str): return 0
    val = label.strip().upper()
    if val == "<30": return 1
    # treat >30 and NO as negative
    return 0
