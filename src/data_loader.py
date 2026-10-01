"""
Data Loader Module for Student Performance Analysis Dashboard
=============================================================
Provides validated ingestion routines for the UCI Student Performance Dataset.
Supports single-subject, combined-subject, and linked-cohort extraction.

Provenance:
- Original research: P. Cortez and A. Silva, 2008 (University of Minho)
- Official UCI Repository: https://archive.ics.uci.edu/dataset/320/student+performance
- Kaggle mirror: https://www.kaggle.com/dskagglemt/student-performance-data-set/metadata
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

# Core columns expected in the UCI Student Performance dataset
EXPECTED_COLUMNS: List[str] = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "G1", "G2", "G3"
]

# Student demographic keys used by Cortez & Silva (2008) to link students across subjects
MERGE_KEYS: List[str] = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def resolve_data_path(filename: str, custom_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve absolute path to a raw data file safely.
    Checks custom directory, default repo directory, and working directory fallbacks.
    """
    candidates = []
    if custom_dir:
        candidates.append(Path(custom_dir) / filename)
    candidates.extend([
        DEFAULT_DATA_DIR / filename,
        Path("data/raw") / filename,
        Path("data") / filename,
        Path(filename)
    ])
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(
        f"Dataset file '{filename}' could not be located in any expected path: "
        f"{[str(c) for c in candidates]}"
    )


def load_raw_data(
    course: str = "mat",
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load a single student performance dataset.

    Parameters:
    -----------
    course : str
        Either 'mat' (Mathematics, 395 records) or 'por' (Portuguese, 649 records).
    data_dir : str or Path, optional
        Optional directory path where raw CSV files reside.

    Returns:
    --------
    pd.DataFrame
        Loaded and column-validated DataFrame.
    """
    course = course.lower().strip()
    if course not in ("mat", "por", "math", "portuguese"):
        raise ValueError(f"Invalid course '{course}'. Expected 'mat' or 'por'.")

    filename = "student-mat.csv" if course in ("mat", "math") else "student-por.csv"
    filepath = resolve_data_path(filename, custom_dir=data_dir)

    df = pd.read_csv(filepath, sep=";")
    validate_dataset(df)
    return df


def load_both_courses(
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load and concatenate both Mathematics and Portuguese datasets into a single
    unified DataFrame with a 'subject' column discriminator.

    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame (1,044 total records).
    """
    df_mat = load_raw_data("mat", data_dir=data_dir).copy()
    df_mat["subject"] = "Mathematics"

    df_por = load_raw_data("por", data_dir=data_dir).copy()
    df_por["subject"] = "Portuguese"

    combined = pd.concat([df_mat, df_por], ignore_index=True)
    return combined


def load_merged_cohort(
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Replicate the Cortez & Silva (2008) student-merge.R logic to identify the
    382 overlapping students present in both the Mathematics and Portuguese datasets.

    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with suffixed columns (_mat and _por) for subject-specific variables.
    """
    df_mat = load_raw_data("mat", data_dir=data_dir)
    df_por = load_raw_data("por", data_dir=data_dir)

    merged = pd.merge(
        df_mat,
        df_por,
        on=MERGE_KEYS,
        suffixes=("_mat", "_por")
    )
    return merged


def validate_dataset(df: pd.DataFrame) -> Dict[str, Union[bool, int, List[str]]]:
    """
    Validate that an ingested dataset conforms to the expected UCI schema and value constraints.

    Raises:
    -------
    ValueError: If required columns are missing or numeric bounds are violated.
    """
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    # Boundary validation on critical variables
    if not df["age"].between(15, 25).all():
        raise ValueError("Age values fall outside expected range [15, 25].")

    for grade_col in ["G1", "G2", "G3"]:
        if not df[grade_col].between(0, 20).all():
            raise ValueError(f"Grade '{grade_col}' contains values outside [0, 20].")

    if not df["studytime"].between(1, 4).all():
        raise ValueError("Study time values fall outside [1, 4].")

    return {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": int(df.isnull().sum().sum())
    }
