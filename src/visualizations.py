"""
Data Visualization Module for Student Performance Dashboard
============================================================
Generates rich, publication-grade interactive Plotly visual representations
for student performance distributions, demographic impacts, correlation patterns,
and predictive modeling diagnostics.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.preprocessing import LABEL_MAPPINGS

COLOR_PRIMARY = "#2563EB"       # Academic Royal Blue
COLOR_ACCENT = "#10B981"        # Emerald Green
COLOR_WARNING = "#F59E0B"       # Amber
COLOR_DANGER = "#EF4444"        # Red
COLOR_NEUTRAL = "#6B7280"       # Gray
COLOR_PALETTE = ["#2563EB", "#10B981", "#8B5CF6", "#F59E0B", "#EC4899", "#3B82F6"]


def plot_grade_distribution(df: pd.DataFrame, target_col: str = "G3") -> go.Figure:
    """
    Generate an interactive histogram of student grades with a pass-fail threshold line at 10.
    """
    mean_val = df[target_col].mean()
    median_val = df[target_col].median()

    fig = px.histogram(
        df,
        x=target_col,
        nbins=21,
        color_discrete_sequence=[COLOR_PRIMARY],
        title=f"Final Grade (G3) Distribution [N = {len(df)}]",
        labels={target_col: "Final Grade (0 - 20 Scale)"},
        opacity=0.85
    )

    fig.add_vline(
        x=10,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_DANGER,
        annotation_text="Passing Threshold (10/20)",
        annotation_position="top left"
    )

    fig.add_vline(
        x=mean_val,
        line_width=2,
        line_dash="dot",
        line_color=COLOR_ACCENT,
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2, range=[-0.5, 20.5]),
        yaxis_title="Student Count",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_study_time_vs_grade(df: pd.DataFrame) -> go.Figure:
    """
    Boxplot showing final grade distribution segmented by weekly study hours.
    """
    df_plot = df.copy()
    study_map = LABEL_MAPPINGS["studytime"]
    df_plot["Study Hours"] = df_plot["studytime"].map(study_map)
    category_order = [study_map[1], study_map[2], study_map[3], study_map[4]]

    fig = px.box(
        df_plot,
        x="Study Hours",
        y="G3",
        color="Study Hours",
        category_orders={"Study Hours": category_order},
        color_discrete_sequence=COLOR_PALETTE,
        title="Weekly Study Time vs Final Grade (G3)",
        points="all",
        labels={"G3": "Final Grade (0-20)"}
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis=dict(range=[-0.5, 20.5]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_absences_vs_grade(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of school absences vs final grade with OLS trendline.
    """
    fig = px.scatter(
        df,
        x="absences",
        y="G3",
        color="sex",
        color_discrete_map={"M": COLOR_PRIMARY, "F": COLOR_ACCENT},
        title="School Absences vs Final Grade (G3) with Trendline",
        labels={"absences": "Number of School Absences", "G3": "Final Grade (0-20)", "sex": "Sex"},
        hover_data=["age", "studytime", "failures"]
    )

    # Compute explicit linear trendline using numpy (no statsmodels dependency)
    if len(df) > 1 and df["absences"].nunique() > 1:
        x_vals = df["absences"].values
        y_vals = df["G3"].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_line = slope * x_line + intercept

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=COLOR_DANGER, dash="dash", width=2),
                name=f"OLS Trendline (slope: {slope:.3f})"
            )
        )

    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[-0.5, 20.5]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_failures_vs_grade(df: pd.DataFrame) -> go.Figure:
    """
    Box plot depicting the severe impact of past course failures on final grade.
    """
    df_plot = df.copy()
    fail_map = LABEL_MAPPINGS["failures"]
    df_plot["Past Failures"] = df_plot["failures"].map(fail_map)
    order = [fail_map[0], fail_map[1], fail_map[2], fail_map[3]]

    fig = px.box(
        df_plot,
        x="Past Failures",
        y="G3",
        color="Past Failures",
        category_orders={"Past Failures": order},
        color_discrete_sequence=["#10B981", "#F59E0B", "#F97316", "#EF4444"],
        title="Impact of Past Course Failures on Final Grade (G3)",
        points="outliers",
        labels={"G3": "Final Grade (0-20)"}
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis=dict(range=[-0.5, 20.5]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_gender_performance(df: pd.DataFrame) -> go.Figure:
    """
    Comparative violin distribution of grades across female and male students.
    """
    df_plot = df.copy()
    df_plot["Gender"] = df_plot["sex"].map(LABEL_MAPPINGS["sex"])

    fig = px.violin(
        df_plot,
        x="Gender",
        y="G3",
        color="Gender",
        box=True,
        points="all",
        color_discrete_map={"Female": "#EC4899", "Male": "#3B82F6"},
        title="Gender vs Final Grade Performance",
        labels={"G3": "Final Grade (0-20)"}
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis=dict(range=[-0.5, 20.5]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_school_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Comparison of performance across the two secondary schools (GP vs MS).
    """
    df_plot = df.copy()
    df_plot["School Name"] = df_plot["school"].map(LABEL_MAPPINGS["school"])

    fig = px.box(
        df_plot,
        x="School Name",
        y="G3",
        color="School Name",
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_ACCENT],
        title="School Comparison: Gabriel Pereira (GP) vs Mousinho da Silveira (MS)",
        points="all",
        labels={"G3": "Final Grade (0-20)"}
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis=dict(range=[-0.5, 20.5]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Age distribution histogram segmented by academic outcome (pass/fail).
    """
    df_plot = df.copy()
    df_plot["Outcome"] = np.where(df_plot["G3"] >= 10, "Pass (>=10)", "Fail (<10)")

    fig = px.histogram(
        df_plot,
        x="age",
        color="Outcome",
        barmode="group",
        color_discrete_map={"Pass (>=10)": COLOR_ACCENT, "Fail (<10)": COLOR_DANGER},
        title="Age Distribution Segmented by Pass/Fail Outcome",
        labels={"age": "Student Age (Years)", "count": "Count"}
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_parental_education(df: pd.DataFrame) -> go.Figure:
    """
    Dual bar chart comparing student performance across Mother's and Father's education levels.
    """
    edu_map = LABEL_MAPPINGS["Medu"]
    medu_stats = df.groupby("Medu")["G3"].mean().reset_index()
    medu_stats["Education Level"] = medu_stats["Medu"].map(edu_map)
    medu_stats["Parent"] = "Mother (Medu)"

    fedu_stats = df.groupby("Fedu")["G3"].mean().reset_index()
    fedu_stats["Education Level"] = fedu_stats["Fedu"].map(edu_map)
    fedu_stats["Parent"] = "Father (Fedu)"

    combined = pd.concat([medu_stats, fedu_stats], ignore_index=True)
    level_order = [edu_map[0], edu_map[1], edu_map[2], edu_map[3], edu_map[4]]

    fig = px.bar(
        combined,
        x="Education Level",
        y="G3",
        color="Parent",
        barmode="group",
        category_orders={"Education Level": level_order},
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_ACCENT],
        title="Parental Education Level vs Mean Final Grade",
        labels={"G3": "Mean Final Grade (0-20)"}
    )

    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[0, 20]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_internet_access(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart depicting final grade based on home internet connectivity.
    """
    df_plot = df.copy()
    df_plot["Internet"] = df_plot["internet"].map(LABEL_MAPPINGS["internet"])

    summary = df_plot.groupby("Internet")["G3"].agg(["mean", "std", "count"]).reset_index()

    fig = px.bar(
        summary,
        x="Internet",
        y="mean",
        error_y="std",
        color="Internet",
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_WARNING],
        title="Home Internet Access vs Student Performance (Mean ± SD)",
        labels={"mean": "Average Final Grade (0-20)"}
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis=dict(range=[0, 20]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Full correlation matrix heatmap across all numerical features.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Feature Correlation Matrix (Pearson r)",
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_grade_leakage_relationship(df: pd.DataFrame) -> go.Figure:
    """
    Side-by-side scatter plots illustrating strong collinearity between G1, G2, and G3.
    Visually documents the academic data leakage phenomenon.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("G1 (First Period) vs G3 (Final)", "G2 (Second Period) vs G3 (Final)")
    )

    corr_g1 = df["G1"].corr(df["G3"])
    corr_g2 = df["G2"].corr(df["G3"])

    # G1 vs G3
    fig.add_trace(
        go.Scatter(
            x=df["G1"], y=df["G3"],
            mode="markers",
            marker=dict(color=COLOR_PRIMARY, opacity=0.6, size=8),
            name=f"G1 vs G3 (r = {corr_g1:.2f})"
        ),
        row=1, col=1
    )

    # G2 vs G3
    fig.add_trace(
        go.Scatter(
            x=df["G2"], y=df["G3"],
            mode="markers",
            marker=dict(color=COLOR_ACCENT, opacity=0.6, size=8),
            name=f"G2 vs G3 (r = {corr_g2:.2f})"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Collinearity & Temporal Data Leakage: Prior Grades vs Final Target G3",
        template="plotly_white",
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(title_text="G1 Grade (0-20)", row=1, col=1)
    fig.update_yaxes(title_text="G3 Grade (0-20)", row=1, col=1)
    fig.update_xaxes(title_text="G2 Grade (0-20)", row=1, col=2)
    fig.update_yaxes(title_text="G3 Grade (0-20)", row=1, col=2)

    return fig


def plot_model_comparison(comparison_table: pd.DataFrame) -> go.Figure:
    """
    Visual comparison of R² and RMSE across models and leakage regimes.
    """
    fig = px.bar(
        comparison_table,
        x="Model",
        y="R2",
        color="Model",
        text="R2",
        title="Model Generalization Comparison: R² Score across Regimes",
        color_discrete_sequence=[COLOR_PRIMARY, "#3B82F6", COLOR_WARNING, COLOR_DANGER],
        labels={"R2": "R² Score (Test Partition)"}
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[-0.1, 1.05]),
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    Horizontal bar chart showing the top N most influential features.
    """
    top_df = importance_df.head(top_n).sort_values(by="importance", ascending=True)

    fig = px.bar(
        top_df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        title=f"Top {top_n} Features by Relative Importance (Random Forest)",
        labels={"importance": "Relative Importance Weight", "feature": "Attribute"}
    )

    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
