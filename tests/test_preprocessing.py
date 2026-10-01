"""
Unit Tests for Preprocessing and Feature Preparation Module
===========================================================
Validates label mapping decoding, missing value handling, one-hot encoding,
and the leakage-aware feature extraction pipelines.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_raw_data
from src.preprocessing import (
    add_readable_labels,
    handle_missing_values,
    prepare_ml_features,
    LABEL_MAPPINGS
)


class TestPreprocessing:
    """Test suite for data preprocessing and feature transformations."""

    @pytest.fixture
    def raw_math_df(self):
        return load_raw_data("mat")

    def test_add_readable_labels(self, raw_math_df):
        """Readable labels should be attached as new columns prefixed with label_."""
        labeled = add_readable_labels(raw_math_df)
        assert "label_school" in labeled.columns
        assert "label_sex" in labeled.columns
        assert "label_studytime" in labeled.columns
        assert "label_failures" in labeled.columns

        # Verify mapping consistency
        assert labeled.loc[labeled["studytime"] == 1, "label_studytime"].iloc[0] == "< 2 hours"
        assert labeled.loc[labeled["sex"] == "F", "label_sex"].iloc[0] == "Female"

    def test_handle_missing_values_clean_data(self, raw_math_df):
        """Clean dataset should pass through unmodified."""
        cleaned = handle_missing_values(raw_math_df)
        assert cleaned.isnull().sum().sum() == 0
        assert len(cleaned) == len(raw_math_df)

    def test_handle_missing_values_with_injected_nans(self, raw_math_df):
        """Injected NaNs in numeric and categorical columns must be imputed properly."""
        corrupted = raw_math_df.copy()
        corrupted.loc[0, "age"] = np.nan
        corrupted.loc[1, "Mjob"] = np.nan

        imputed = handle_missing_values(corrupted)
        assert imputed.isnull().sum().sum() == 0
        # Age should be imputed with median age
        assert imputed.loc[0, "age"] == raw_math_df["age"].median()

    def test_prepare_ml_features_without_prior_grades(self, raw_math_df):
        """Regime B (Clean / Early warning) must NOT contain G1 or G2."""
        X_train, X_test, y_train, y_test, feature_names = prepare_ml_features(
            raw_math_df, include_prior_grades=False, test_size=0.2, random_state=42
        )

        assert "G1" not in feature_names
        assert "G2" not in feature_names
        assert "G3" not in feature_names
        assert len(X_train) == int(len(raw_math_df) * 0.8)
        assert len(X_test) == len(raw_math_df) - len(X_train)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        # All columns in X must be numeric/encoded
        assert X_train.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all()

    def test_prepare_ml_features_with_prior_grades(self, raw_math_df):
        """Regime A (With G1 & G2) must contain G1 and G2 for data leakage demonstration."""
        X_train, X_test, y_train, y_test, feature_names = prepare_ml_features(
            raw_math_df, include_prior_grades=True, test_size=0.2, random_state=42
        )

        assert "G1" in feature_names
        assert "G2" in feature_names
        assert "G3" not in feature_names
        # Regime A should have exactly 2 more features than Regime B (G1 and G2)
        _, _, _, _, feat_clean = prepare_ml_features(raw_math_df, include_prior_grades=False)
        assert len(feature_names) == len(feat_clean) + 2
