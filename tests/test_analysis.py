"""
Unit Tests for Analysis, Modeling, and Insight Generation Module
================================================================
Validates statistical metrics calculations, correlation boundaries,
Linear Regression & Random Forest evaluations, the data leakage demonstration,
and dynamic automated insight outputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_raw_data
from src.analysis import (
    compute_summary_statistics,
    compute_full_numeric_summary,
    compute_correlations,
    compute_subgroup_analysis,
    train_and_evaluate_models,
    compare_leakage_regimes,
    generate_automated_insights
)


class TestAnalysis:
    """Test suite for statistical computations and predictive modeling."""

    @pytest.fixture
    def raw_math_df(self):
        return load_raw_data("mat")

    def test_compute_summary_statistics(self, raw_math_df):
        """Descriptive statistics must match mathematical definitions."""
        stats = compute_summary_statistics(raw_math_df, "G3")
        assert stats["count"] == 395
        assert 10.0 <= stats["mean"] <= 11.0
        assert stats["min"] == 0.0
        assert stats["max"] == 20.0
        assert stats["iqr"] == stats["q75"] - stats["q25"]

    def test_compute_full_numeric_summary(self, raw_math_df):
        """Summary dataframe contains all numeric columns."""
        summary = compute_full_numeric_summary(raw_math_df)
        assert isinstance(summary, pd.DataFrame)
        assert "G3" in summary.index
        assert "age" in summary.index
        assert "absences" in summary.index
        assert "mean" in summary.columns
        assert "median" in summary.columns

    def test_compute_correlations(self, raw_math_df):
        """Correlations must fall strictly in [-1, 1], with G2 strongly correlated to G3."""
        corr_matrix, target_corr = compute_correlations(raw_math_df, "G3")
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert (corr_matrix >= -1.0).all().all() and (corr_matrix <= 1.0).all().all()

        # Target correlation with itself is 1.0
        assert np.isclose(target_corr["G3"], 1.0)
        # G2 correlation with G3 should be very high (> 0.85) in Mathematics
        assert target_corr["G2"] > 0.85

    def test_compute_subgroup_analysis(self, raw_math_df):
        """Subgroup aggregation produces valid counts and means."""
        subgroup = compute_subgroup_analysis(raw_math_df, "sex", "G3")
        assert len(subgroup) == 2
        assert set(subgroup["sex"].unique()) == {"F", "M"}
        assert subgroup["count"].sum() == 395

    def test_train_and_evaluate_models(self, raw_math_df):
        """Model evaluation returns expected metric dictionary with positive RMSE and valid R2."""
        results = train_and_evaluate_models(raw_math_df, include_prior_grades=False, random_state=42)
        assert "Linear Regression" in results
        assert "Random Forest" in results

        rf_test = results["Random Forest"]["test"]
        assert "MAE" in rf_test and "RMSE" in rf_test and "R2" in rf_test
        assert rf_test["RMSE"] > 0
        assert rf_test["MAE"] > 0

    def test_compare_leakage_regimes(self, raw_math_df):
        """
        Critical Academic Proof:
        Regime A (With G1 & G2) MUST achieve dramatically higher R² than Regime B (Without G1 & G2),
        empirically verifying the presence of temporal data leakage.
        """
        comparison = compare_leakage_regimes(raw_math_df, random_state=42)
        r2_leakage = comparison["with_leakage"]["Random Forest"]["test"]["R2"]
        r2_clean = comparison["clean"]["Random Forest"]["test"]["R2"]

        assert r2_leakage > 0.70, "Regime A should achieve high R² due to collinearity with G1/G2."
        assert r2_leakage > r2_clean + 0.30, "Leakage model should outperform clean model by a large margin."

    def test_generate_automated_insights(self, raw_math_df):
        """Automated insights must return structured observations derived from data."""
        insights = generate_automated_insights(raw_math_df)
        assert isinstance(insights, list)
        assert len(insights) >= 4
        for ins in insights:
            assert "category" in ins
            assert "title" in ins
            assert "detail" in ins
            assert len(ins["detail"]) > 10
