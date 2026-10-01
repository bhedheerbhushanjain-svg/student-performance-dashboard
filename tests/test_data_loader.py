"""
Unit Tests for Data Loader Module
=================================
Validates file ingestion, column schema conformance, boundary constraints,
multi-subject concatenation, and linked student cohort reconstruction.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data_loader import (
    load_raw_data,
    load_both_courses,
    load_merged_cohort,
    validate_dataset,
    resolve_data_path,
    EXPECTED_COLUMNS
)


class TestDataLoader:
    """Test suite for data loading and schema validation."""

    def test_load_math_dataset_shape_and_columns(self):
        """Mathematics dataset must contain exactly 395 rows and 33 attributes."""
        df = load_raw_data("mat")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (395, 33)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Expected column '{col}' missing from Mathematics dataset."

    def test_load_portuguese_dataset_shape_and_columns(self):
        """Portuguese dataset must contain exactly 649 rows and 33 attributes."""
        df = load_raw_data("por")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (649, 33)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Expected column '{col}' missing from Portuguese dataset."

    def test_load_both_courses_concatenation(self):
        """Combined loader must return 395 + 649 = 1,044 total records with a 'subject' column."""
        combined = load_both_courses()
        assert len(combined) == 1044
        assert "subject" in combined.columns
        assert set(combined["subject"].unique()) == {"Mathematics", "Portuguese"}
        assert (combined["subject"] == "Mathematics").sum() == 395
        assert (combined["subject"] == "Portuguese").sum() == 649

    def test_load_merged_cohort_count(self):
        """Replicating Cortez & Silva (2008) merge logic must produce 382 common students."""
        merged = load_merged_cohort()
        assert len(merged) == 382
        assert "G3_mat" in merged.columns
        assert "G3_por" in merged.columns

    def test_invalid_course_raises_value_error(self):
        """Requesting an unsupported course name must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid course"):
            load_raw_data("physics")

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Searching in an empty directory must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            resolve_data_path("non_existent_file.csv", custom_dir=tmp_path)

    def test_validate_dataset_schema_constraints(self):
        """Valid dataset passes validation without errors."""
        df = load_raw_data("mat")
        val_result = validate_dataset(df)
        assert val_result["is_valid"] is True
        assert val_result["row_count"] == 395
        assert val_result["missing_values"] == 0

    def test_validate_dataset_out_of_bounds_grade(self):
        """A grade outside [0, 20] must trigger a ValueError."""
        df = load_raw_data("mat").copy()
        df.loc[0, "G3"] = 25  # Exceeds maximum 20
        with pytest.raises(ValueError, match="outside \\[0, 20\\]"):
            validate_dataset(df)

    def test_validate_dataset_missing_column(self):
        """Dropping a required column must fail validation."""
        df = load_raw_data("mat").copy()
        df = df.drop(columns=["studytime"])
        with pytest.raises(ValueError, match="missing expected columns"):
            validate_dataset(df)
