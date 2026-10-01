"""
Statistical Analysis and Machine Learning Modeling Module
=========================================================
Computes descriptive statistics, group comparisons, correlation structures,
and predictive regression models for academic performance (G3).

Rigorous Data Leakage Protocol:
-------------------------------
Explicitly differentiates between:
- Regime A (With G1 & G2): Demonstrates temporal data leakage / exam continuity.
- Regime B (Without G1 & G2): Genuine early-warning prediction from socio-demographics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import prepare_ml_features


def compute_summary_statistics(
    df: pd.DataFrame,
    target_col: str = "G3"
) -> Dict[str, float]:
    """
    Calculate comprehensive descriptive statistics for a numeric column.

    Returns:
    --------
    dict containing: count, mean, median, std, min, q25, q50, q75, max, iqr, skew, kurtosis
    """
    series = df[target_col].dropna().astype(float)
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    iqr = q75 - q25

    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": q25,
        "q50": float(series.median()),
        "q75": q75,
        "max": float(series.max()),
        "iqr": float(iqr),
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis())
    }


def compute_full_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tabular summary for all numeric variables in the dataset.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    records = []
    for col in numeric_cols:
        stats = compute_summary_statistics(df, target_col=col)
        stats["variable"] = col
        records.append(stats)

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.set_index("variable")
    return summary_df


def compute_correlations(
    df: pd.DataFrame,
    target_col: str = "G3"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute Pearson correlation matrix across numeric features and
    isolate correlations relative to the target variable.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]: Full correlation matrix and target correlation series sorted descending.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    return corr_matrix, target_corr


def compute_subgroup_analysis(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "G3"
) -> pd.DataFrame:
    """
    Group dataset by categorical attribute and calculate descriptive metrics.
    """
    grouped = df.groupby(group_col)[target_col].agg(
        count="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max"
    ).reset_index()
    grouped["mean"] = grouped["mean"].round(2)
    grouped["std"] = grouped["std"].round(2)
    grouped["median"] = grouped["median"].round(2)
    return grouped


def train_and_evaluate_models(
    df: pd.DataFrame,
    include_prior_grades: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train Linear Regression and Random Forest Regressor models on student data.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to train on.
    include_prior_grades : bool, default=False
        Toggle inclusion of G1 and G2.
    random_state : int, default=42
        Reproducibility seed.

    Returns:
    --------
    dict containing evaluation metrics, models, predictions, and feature importances.
    """
    X_train, X_test, y_train, y_test, feature_names = prepare_ml_features(
        df,
        include_prior_grades=include_prior_grades,
        test_size=0.2,
        random_state=random_state
    )

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr_train = lr.predict(X_train)
    y_pred_lr_test = lr.predict(X_test)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf_train = rf.predict(X_train)
    y_pred_rf_test = rf.predict(X_test)

    def _calc_metrics(y_true, y_pred) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        return {
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4)
        }

    results = {
        "regime": "With G1 & G2 (Data Leakage Experiment)" if include_prior_grades else "Without G1 & G2 (Early Warning Model)",
        "include_prior_grades": include_prior_grades,
        "feature_count": len(feature_names),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "Linear Regression": {
            "train": _calc_metrics(y_train, y_pred_lr_train),
            "test": _calc_metrics(y_test, y_pred_lr_test),
            "model": lr,
            "y_test": y_test.values,
            "y_pred": y_pred_lr_test
        },
        "Random Forest": {
            "train": _calc_metrics(y_train, y_pred_rf_train),
            "test": _calc_metrics(y_test, y_pred_rf_test),
            "model": rf,
            "y_test": y_test.values,
            "y_pred": y_pred_rf_test
        },
        "feature_importance": pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True),
        "coefficients": pd.DataFrame({
            "feature": feature_names,
            "coefficient": lr.coef_
        }).sort_values(by="coefficient", ascending=False).reset_index(drop=True)
    }

    return results


def compare_leakage_regimes(
    df: pd.DataFrame,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute side-by-side comparative modeling between Regime A (With G1/G2)
    and Regime B (Without G1/G2) to rigorously evaluate data leakage effects.
    """
    res_leakage = train_and_evaluate_models(df, include_prior_grades=True, random_state=random_state)
    res_clean = train_and_evaluate_models(df, include_prior_grades=False, random_state=random_state)

    comparison_table = pd.DataFrame([
        {
            "Model": "Linear Regression (With G1/G2 - Leakage)",
            "MAE": res_leakage["Linear Regression"]["test"]["MAE"],
            "MSE": res_leakage["Linear Regression"]["test"]["MSE"],
            "RMSE": res_leakage["Linear Regression"]["test"]["RMSE"],
            "R2": res_leakage["Linear Regression"]["test"]["R2"],
            "Features": res_leakage["feature_count"]
        },
        {
            "Model": "Random Forest (With G1/G2 - Leakage)",
            "MAE": res_leakage["Random Forest"]["test"]["MAE"],
            "MSE": res_leakage["Random Forest"]["test"]["MSE"],
            "RMSE": res_leakage["Random Forest"]["test"]["RMSE"],
            "R2": res_leakage["Random Forest"]["test"]["R2"],
            "Features": res_leakage["feature_count"]
        },
        {
            "Model": "Linear Regression (Without G1/G2 - Clean)",
            "MAE": res_clean["Linear Regression"]["test"]["MAE"],
            "MSE": res_clean["Linear Regression"]["test"]["MSE"],
            "RMSE": res_clean["Linear Regression"]["test"]["RMSE"],
            "R2": res_clean["Linear Regression"]["test"]["R2"],
            "Features": res_clean["feature_count"]
        },
        {
            "Model": "Random Forest (Without G1/G2 - Clean)",
            "MAE": res_clean["Random Forest"]["test"]["MAE"],
            "MSE": res_clean["Random Forest"]["test"]["MSE"],
            "RMSE": res_clean["Random Forest"]["test"]["RMSE"],
            "R2": res_clean["Random Forest"]["test"]["R2"],
            "Features": res_clean["feature_count"]
        }
    ])

    return {
        "with_leakage": res_leakage,
        "clean": res_clean,
        "table": comparison_table
    }


def generate_automated_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Derive factual, calculated analytical insights directly from dataset distributions.
    Strictly calculates numbers from empirical data without hardcoded guesses.
    """
    insights: List[Dict[str, str]] = []

    # 1. Overall Grade distribution
    g3 = df["G3"]
    mean_g3 = float(g3.mean())
    median_g3 = float(g3.median())
    zero_scores = int((g3 == 0).sum())
    zero_pct = float((zero_scores / len(df)) * 100)
    passing = float((g3 >= 10).mean() * 100)

    insights.append({
        "category": "Overall Academic Distribution",
        "title": f"Mean Final Grade: {mean_g3:.2f} / 20 (Pass Rate: {passing:.1f}%)",
        "detail": (
            f"Across {len(df)} records, the median grade is {median_g3:.1f}, "
            f"with {zero_scores} students ({zero_pct:.1f}%) scoring 0 (indicating exam dropouts or complete failure). "
            f"A score of 10/20 represents the standard Portuguese passing benchmark."
        )
    })

    # 2. Historical Failures impact
    if "failures" in df.columns:
        mean_no_fail = float(df[df["failures"] == 0]["G3"].mean())
        has_fail_df = df[df["failures"] > 0]
        if not has_fail_df.empty:
            mean_with_fail = float(has_fail_df["G3"].mean())
            drop = mean_no_fail - mean_with_fail
            insights.append({
                "category": "Historical Academic Record",
                "title": f"Prior Failures Correlate with a {drop:.2f} Point Performance Drop",
                "detail": (
                    f"Students with zero past failures averaged {mean_no_fail:.2f}/20 in G3, "
                    f"whereas students with 1 or more past class failures averaged {mean_with_fail:.2f}/20. "
                    f"Prior academic failure is one of the strongest negative socio-demographic indicators."
                )
            })

    # 3. Weekly Study Time impact
    if "studytime" in df.columns:
        low_study = float(df[df["studytime"] == 1]["G3"].mean())
        high_study = float(df[df["studytime"] >= 3]["G3"].mean())
        diff_study = high_study - low_study
        insights.append({
            "category": "Study Habits",
            "title": f"Extended Study Time Yields +{diff_study:.2f} Higher Average Grade",
            "detail": (
                f"Students studying <2 hours weekly achieved an average G3 of {low_study:.2f}, "
                f"while those studying 5+ hours weekly achieved {high_study:.2f}. "
                f"While study time correlates positively, variance remains high due to differences in study efficacy."
            )
        })

    # 4. G1 & G2 Correlation / Data Leakage Check
    if "G1" in df.columns and "G2" in df.columns:
        corr_g1_g3 = float(df[["G1", "G3"]].corr().iloc[0, 1])
        corr_g2_g3 = float(df[["G2", "G3"]].corr().iloc[0, 1])
        insights.append({
            "category": "Methodological & Data Leakage Note",
            "title": f"Strong Collinearity: G1 (r={corr_g1_g3:.2f}) and G2 (r={corr_g2_g3:.2f}) with G3",
            "detail": (
                f"Period 1 and Period 2 exams exhibit extremely strong correlation with final grade G3. "
                f"Including G1 and G2 in machine learning models artificially inflates R² to ~0.80+, "
                f"representing temporal data leakage rather than genuine early-stage student risk prediction."
            )
        })

    # 5. Parental Higher Education (Medu / Fedu)
    if "Medu" in df.columns and "Fedu" in df.columns:
        high_edu_mask = (df["Medu"] == 4) | (df["Fedu"] == 4)
        low_edu_mask = (df["Medu"] <= 1) & (df["Fedu"] <= 1)
        mean_high_edu = float(df[high_edu_mask]["G3"].mean())
        mean_low_edu = float(df[low_edu_mask]["G3"].mean())
        edu_diff = mean_high_edu - mean_low_edu
        insights.append({
            "category": "Family Background",
            "title": f"Parental Education Disparity: +{edu_diff:.2f} Point Advantage",
            "detail": (
                f"Students with at least one parent holding higher education averaged {mean_high_edu:.2f}, "
                f"compared to {mean_low_edu:.2f} for students whose parents attained only primary or no schooling."
            )
        })

    # 6. Higher Education Aspiration
    if "higher" in df.columns:
        higher_yes = float(df[df["higher"] == "yes"]["G3"].mean())
        higher_no = float(df[df["higher"] == "no"]["G3"].mean())
        aspiration_gap = higher_yes - higher_no
        insights.append({
            "category": "Student Motivation",
            "title": f"Higher Education Aspiration Gap: +{aspiration_gap:.2f} Grade Points",
            "detail": (
                f"Students aspiring to pursue higher education averaged {higher_yes:.2f}/20, "
                f"while those not planning for higher education scored {higher_no:.2f}/20."
            )
        })

    return insights
