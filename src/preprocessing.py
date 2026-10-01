"""
Data Preprocessing and Feature Engineering Module
=================================================
Transforms raw student-performance features into analytical forms,
provides readable categorical decoders for dashboard visuals,
and constructs leak-free ML feature matrices.

Important Data Science Distinction:
-----------------------------------
- Regime A (Full Academic): Uses all attributes including G1 and G2.
- Regime B (Early Performance): Excludes G1 and G2 to evaluate true pre-exam
  predictive power without temporal data leakage.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Human-readable label decoders for presentation and dashboards
LABEL_MAPPINGS: Dict[str, Dict[Union[int, str], str]] = {
    "school": {
        "GP": "Gabriel Pereira",
        "MS": "Mousinho da Silveira"
    },
    "sex": {
        "F": "Female",
        "M": "Male"
    },
    "address": {
        "U": "Urban",
        "R": "Rural"
    },
    "famsize": {
        "LE3": "≤ 3 members",
        "GT3": "> 3 members"
    },
    "Pstatus": {
        "T": "Living Together",
        "A": "Living Apart"
    },
    "Medu": {
        0: "None",
        1: "Primary (4th grade)",
        2: "5th to 9th grade",
        3: "Secondary education",
        4: "Higher education"
    },
    "Fedu": {
        0: "None",
        1: "Primary (4th grade)",
        2: "5th to 9th grade",
        3: "Secondary education",
        4: "Higher education"
    },
    "traveltime": {
        1: "< 15 min",
        2: "15 - 30 min",
        3: "30 min - 1 hr",
        4: "> 1 hr"
    },
    "studytime": {
        1: "< 2 hours",
        2: "2 - 5 hours",
        3: "5 - 10 hours",
        4: "> 10 hours"
    },
    "failures": {
        0: "0 failures",
        1: "1 failure",
        2: "2 failures",
        3: "3 failures",
        4: "4+ failures"
    },
    "internet": {
        "yes": "Internet Access",
        "no": "No Internet"
    },
    "higher": {
        "yes": "Plans Higher Ed",
        "no": "No Higher Ed"
    },
    "romantic": {
        "yes": "In Relationship",
        "no": "Single"
    }
}

BINARY_COLUMNS: List[str] = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

NOMINAL_COLUMNS: List[str] = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian"
]

ORDINAL_COLUMNS: List[str] = [
    "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health"
]

NUMERIC_CONTINUOUS_COLUMNS: List[str] = [
    "age", "absences"
]

GRADE_COLUMNS: List[str] = [
    "G1", "G2"
]

TARGET_COLUMN: str = "G3"


def add_readable_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append descriptive label columns (prefixed with 'label_') to facilitate
    clear, readable charting and tooltips in Streamlit dashboards.
    """
    df_out = df.copy()
    for col, mapping in LABEL_MAPPINGS.items():
        if col in df_out.columns:
            label_col = f"label_{col}"
            df_out[label_col] = df_out[col].map(mapping).fillna(df_out[col].astype(str))
    return df_out


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely inspect and handle missing values in the dataset.
    Raw UCI data contains 0 missing entries; this function guarantees
    integrity if noisy external records are introduced.
    """
    df_clean = df.copy()
    if df_clean.isnull().sum().sum() == 0:
        return df_clean

    # Numeric imputation: median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # Categorical imputation: mode
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)

    return df_clean


def prepare_ml_features(
    df: pd.DataFrame,
    include_prior_grades: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Encode features and perform train/test split for regression models.

    Parameters:
    -----------
    df : pd.DataFrame
        Input student DataFrame.
    include_prior_grades : bool, default=False
        - True: Regime A (includes G1, G2). Warning: High correlation introduces leakage!
        - False: Regime B (excludes G1, G2). Evaluates socio-demographic baseline.
    test_size : float, default=0.2
        Fraction reserved for test partition.
    random_state : int, default=42
        Seed for reproducibility.

    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names
    """
    df_work = handle_missing_values(df.copy())

    # Drop non-feature and target columns
    drop_cols = [TARGET_COLUMN]
    if "subject" in df_work.columns:
        drop_cols.append("subject")
    for col in list(LABEL_MAPPINGS.keys()):
        label_col = f"label_{col}"
        if label_col in df_work.columns:
            drop_cols.append(label_col)

    if not include_prior_grades:
        drop_cols.extend([c for c in GRADE_COLUMNS if c in df_work.columns])

    y = df_work[TARGET_COLUMN].copy()
    X_raw = df_work.drop(columns=[c for c in drop_cols if c in df_work.columns])

    # Convert binary yes/no columns to numeric 0/1
    for col in BINARY_COLUMNS:
        if col in X_raw.columns:
            X_raw[col] = (X_raw[col].str.lower() == "yes").astype(int)

    # One-hot encode nominal categorical columns
    nominal_in_df = [c for c in NOMINAL_COLUMNS if c in X_raw.columns]
    X_encoded = pd.get_dummies(X_raw, columns=nominal_in_df, drop_first=True, dtype=float)

    feature_names = list(X_encoded.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names
