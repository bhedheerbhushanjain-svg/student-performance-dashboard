"""
Student Performance Analysis Dashboard - Streamlit Application
=============================================================
An interactive, academic Open Source Technologies (OST) dashboard for
exploring student performance distributions, demographic factors,
statistical summaries, and machine learning models with rigorous data leakage controls.

Author: Bhedheer Bhushan Jain (PRN: 25030422033)
Dataset Provenance: UCI Machine Learning Repository (Cortez & Silva, 2008)
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    load_raw_data,
    load_both_courses,
    load_merged_cohort,
    EXPECTED_COLUMNS
)
from src.preprocessing import (
    LABEL_MAPPINGS,
    add_readable_labels,
    handle_missing_values,
    BINARY_COLUMNS,
    NOMINAL_COLUMNS,
    GRADE_COLUMNS,
    TARGET_COLUMN
)
from src.analysis import (
    compute_summary_statistics,
    compute_full_numeric_summary,
    compute_correlations,
    compute_subgroup_analysis,
    compare_leakage_regimes,
    generate_automated_insights
)
from src.visualizations import (
    plot_grade_distribution,
    plot_study_time_vs_grade,
    plot_absences_vs_grade,
    plot_failures_vs_grade,
    plot_gender_performance,
    plot_school_comparison,
    plot_age_distribution,
    plot_parental_education,
    plot_internet_access,
    plot_correlation_heatmap,
    plot_grade_leakage_relationship,
    plot_model_comparison,
    plot_feature_importance
)

# Page configuration
st.set_page_config(
    page_title="Student Performance Analysis Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #dbeafe;
        color: #1e40af;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_dataset(dataset_choice: str) -> pd.DataFrame:
    """Load and cache the selected dataset."""
    if "All Students" in dataset_choice or "Combined" in dataset_choice:
        return load_both_courses()
    elif "Mathematics" in dataset_choice:
        return load_raw_data("mat")
    elif "Portuguese" in dataset_choice:
        return load_raw_data("por")
    elif "Matched" in dataset_choice:
        return _normalise_merged_cohort(load_merged_cohort())
    return load_both_courses()


def _normalise_merged_cohort(merged: pd.DataFrame) -> pd.DataFrame:
    """
    The Matched Cohort comes from pd.merge() with suffixes ('_mat', '_por').
    Non-key columns that appear in both subjects get renamed to
    col_mat / col_por.  This helper collapses the DataFrame back to the
    standard UCI schema by:
      1. Renaming every *_mat column -> bare name (Math values used as
         canonical value for shared behavioural features).
      2. Dropping the *_por duplicates.
      3. Keeping all MERGE_KEYS columns (already un-suffixed).
    The result has exactly the same column set as load_raw_data() so every
    sidebar filter, chart function, and ML pipeline works without changes.
    """
    rename_map = {}
    drop_cols = []
    for col in merged.columns:
        if col.endswith("_mat"):
            rename_map[col] = col[:-4]   # strip '_mat'
        elif col.endswith("_por"):
            drop_cols.append(col)        # drop Portuguese duplicate

    df = merged.drop(columns=drop_cols).rename(columns=rename_map)

    # Add a helper column so users know this is the 382-student overlap cohort
    df["subject"] = "Matched (Math+Por)"
    return df


@st.cache_data(show_spinner=False)
def get_cached_ml_results(cache_key: str, _df: pd.DataFrame):
    """
    Train ML models once per unique dataset and cache the result.
    cache_key encodes dataset identity; _df is not hashed by Streamlit
    (leading underscore convention) — cache_key is used instead.
    Returns None when there are too few rows for a reliable split.
    """
    if len(_df) < 50:
        return None
    return compare_leakage_regimes(_df, random_state=42)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
st.sidebar.title("Student Analytics")
st.sidebar.markdown("**Open Source Technologies (OST) Project**")
st.sidebar.markdown("---")

st.sidebar.subheader("📁 Dataset Selection")
dataset_choice = st.sidebar.selectbox(
    "Select Academic Cohort:",
    [
        "All Students (Combined Dataset - 1,044 Records)",
        "Mathematics Cohort (student-mat.csv - 395 Records)",
        "Portuguese Cohort (student-por.csv - 649 Records)",
        "Matched Cohort (Overlapping Students - 382 Records)"
    ],
    index=0
)

# Ingest data
raw_df = get_cached_dataset(dataset_choice)
labeled_df = add_readable_labels(raw_df)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Interactive Filters")

# Filter: School
available_schools = sorted(raw_df["school"].unique())
school_labels = [LABEL_MAPPINGS["school"].get(s, s) for s in available_schools]
selected_school_labels = st.sidebar.multiselect(
    "School:",
    school_labels,
    default=school_labels
)
school_reverse_map = {v: k for k, v in LABEL_MAPPINGS["school"].items()}
selected_schools = [school_reverse_map.get(s, s) for s in selected_school_labels]

# Filter: Gender
available_sex = sorted(raw_df["sex"].unique())
sex_labels = [LABEL_MAPPINGS["sex"].get(s, s) for s in available_sex]
selected_sex_labels = st.sidebar.multiselect(
    "Sex / Gender:",
    sex_labels,
    default=sex_labels
)
sex_reverse_map = {v: k for k, v in LABEL_MAPPINGS["sex"].items()}
selected_sex = [sex_reverse_map.get(s, s) for s in selected_sex_labels]

# Filter: Age Range
min_age = int(raw_df["age"].min())
max_age = int(raw_df["age"].max())
selected_age_range = st.sidebar.slider(
    "Age Range:",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# Filter: Study Time
study_map = LABEL_MAPPINGS["studytime"]
study_options = [study_map[i] for i in sorted(study_map.keys())]
selected_study_labels = st.sidebar.multiselect(
    "Weekly Study Time:",
    study_options,
    default=study_options
)
study_reverse_map = {v: k for k, v in study_map.items()}
selected_study = [study_reverse_map[s] for s in selected_study_labels]

# Filter: Failures
fail_map = LABEL_MAPPINGS["failures"]
fail_options = [fail_map[i] for i in sorted(fail_map.keys()) if i in raw_df["failures"].unique()]
selected_fail_labels = st.sidebar.multiselect(
    "Past Class Failures:",
    fail_options,
    default=fail_options
)
fail_reverse_map = {v: k for k, v in fail_map.items()}
selected_failures = [fail_reverse_map[f] for f in selected_fail_labels]

# Filter: Internet
internet_map = LABEL_MAPPINGS["internet"]
internet_options = [internet_map[k] for k in sorted(internet_map.keys())]
selected_internet_labels = st.sidebar.multiselect(
    "Home Internet Access:",
    internet_options,
    default=internet_options
)
internet_reverse_map = {v: k for k, v in internet_map.items()}
selected_internet = [internet_reverse_map[i] for i in selected_internet_labels]

# Apply filter masks
filtered_df = raw_df[
    (raw_df["school"].isin(selected_schools)) &
    (raw_df["sex"].isin(selected_sex)) &
    (raw_df["age"].between(selected_age_range[0], selected_age_range[1])) &
    (raw_df["studytime"].isin(selected_study)) &
    (raw_df["failures"].isin(selected_failures)) &
    (raw_df["internet"].isin(selected_internet))
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Records:** `{len(filtered_df)}` / `{len(raw_df)}`")
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Student Details:**
- **Author:** Bhedheer Bhushan Jain
- **PRN:** `25030422033`
- **License:** MIT License
- **Source:** UCI ML Repository (Cortez & Silva, 2008)
""")

# ==========================================
# MAIN DASHBOARD HEADER & KPI METRICS
# ==========================================
st.title("🎓 Student Performance Analysis Dashboard")
st.markdown(
    '<span class="badge-pill">Academic OST Project</span>'
    '<span class="badge-pill">UCI Machine Learning Dataset</span>'
    '<span class="badge-pill">Interactive Analytics</span>'
    '<span class="badge-pill">Machine Learning Lab</span>',
    unsafe_allow_html=True
)

if len(filtered_df) == 0:
    st.warning("⚠️ No student records match the active filter criteria. Please broaden your sidebar filters.")
    st.stop()

# Key KPI Cards
g3_col = filtered_df["G3"]
mean_g3 = float(g3_col.mean())
median_g3 = float(g3_col.median())
std_g3 = float(g3_col.std()) if len(g3_col) > 1 else 0.0
pass_rate = float((g3_col >= 10).mean() * 100)
zero_count = int((g3_col == 0).sum())

# Stable cache key for ML results — shared by Tab 5 and Presentation tab
ml_cache_key = f"{dataset_choice}|{len(filtered_df)}|{int(filtered_df['G3'].sum())}"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Records", f"{len(filtered_df):,}", f"{(len(filtered_df)/len(raw_df))*100:.1f}% of cohort")
col2.metric("Mean Final Grade (G3)", f"{mean_g3:.2f} / 20", f"SD: {std_g3:.2f}")
col3.metric("Median Final Grade", f"{median_g3:.1f} / 20", "Passing = 10")
col4.metric("Passing Rate (≥ 10)", f"{pass_rate:.1f}%", f"{(g3_col >= 10).sum()} students")
col5.metric("Zero Scores (Dropouts)", f"{zero_count}", f"{(zero_count/len(filtered_df))*100:.1f}% of active")

st.markdown("---")

# ==========================================
# DASHBOARD TABS
# ==========================================
tab_overview, tab_data, tab_eda, tab_stats, tab_ml, tab_insights, tab_presentation = st.tabs([
    "🏠 Overview & Architecture",
    "📁 Dataset Explorer",
    "📈 Performance Visualizations",
    "🧮 Statistical Deep-Dive",
    "🔬 ML & Data Leakage Lab",
    "💡 Automated Insights",
    "📊 Presentation"
])

# ------------------------------------------
# TAB 1: OVERVIEW & ARCHITECTURE
# ------------------------------------------
with tab_overview:
    st.header("Project Overview & Architecture")

    col_intro1, col_intro2 = st.columns([3, 2])
    with col_intro1:
        st.subheader("Objective")
        st.write(
            "The **Student Performance Analysis Dashboard** is an academic Open Source Technologies (OST) "
            "project designed to analyze, visualize, and model secondary education student outcomes. "
            "Built with **Python**, **Streamlit**, and **Plotly**, the system incorporates reproducible data ingestion, "
            "rigorous statistical evaluations, and comparative machine learning pipelines."
        )

        st.subheader("Key Academic Research Questions")
        st.markdown("""
        1. **Grade Distribution:** How are final secondary school grades ($G3$) distributed across subjects?
        2. **Study Habits:** Does increasing weekly study time produce statistically significant grade improvements?
        3. **Attendance & Absences:** How strongly do school absences impact academic retention and final scores?
        4. **Historical Academic Record:** Does past failure history predict future academic distress?
        5. **Socio-Demographic Disparities:** What disparities exist across parental education levels and home internet access?
        6. **Collinearity & Data Leakage:** Why does predicting $G3$ with prior term grades ($G1$, $G2$) create temporal data leakage?
        """)

    with col_intro2:
        st.subheader("Repository & Provenance")
        st.info("""
        **Author:** Bhedheer Bhushan Jain  
        **PRN:** `25030422033`  
        **Course:** Open Source Technologies (OST)  
        **Primary Dataset:** UCI Student Performance Dataset  
        **Citation:** P. Cortez and A. Silva (2008), University of Minho, Portugal  
        **Kaggle Mirror:** `kaggle.com/dskagglemt/student-performance-data-set`  
        **License:** MIT License (Original Project Code)
        """)

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("""
```
┌────────────────────────────────────────────────────────────────────────┐
│                   Student Performance Analysis Dashboard                │
├──────────────────┬───────────────────┬─────────────────────────────────┤
│ Data Pipeline    │ Analytical Core   │ Presentation & Interface        │
├──────────────────┼───────────────────┼─────────────────────────────────┤
│ • data_loader.py │ • analysis.py     │ • Streamlit (dashboard/app.py)  │
│   - Raw Ingestion│   - Stats/IQR     │ • Interactive Plotly Engine     │
│   - Cohort Merge │   - Regressions   │ • Dynamic KPI Cards             │
│ • preprocessing  │   - RF Regressor  │ • Leakage Sandbox               │
│   - Encoders     │ • visual.py       │ • Dynamic Findings Generator    │
│   - Leakage Safe │   - 12+ Visuals   │                                 │
└──────────────────┴───────────────────┴─────────────────────────────────┘
```
    """)

    st.markdown("---")
    st.subheader("Data Privacy & Ethical Use Statement")
    st.markdown("""
    > **Ethical Note:** This dataset contains fully anonymized educational research records collected in Portugal.
    > No personally identifiable information (PII) such as student names, national IDs, addresses, or telephone numbers
    > are included or processed.
    > 
    > **Caution on Predictive Modeling:** Machine learning predictions generated in this project are intended
    > solely for educational and diagnostic exploration. Correlation does **not** imply causation. Model outputs
    > should never be used to make high-stakes automated decisions regarding student admissions, tracking, or disciplinary actions.
    """)

# ------------------------------------------
# TAB 2: DATASET EXPLORER
# ------------------------------------------
with tab_data:
    st.header("Dataset Inspection & Metadata")
    st.write(f"Displaying **{len(filtered_df)}** records matching the active filters out of **{len(raw_df)}** records in `{dataset_choice}`.")

    st.dataframe(filtered_df, use_container_width=True, height=350)

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.subheader("Dataset Shape & Quality")
        st.markdown(f"""
        - **Total Rows (Cohort):** `{len(raw_df):,}`
        - **Filtered Rows:** `{len(filtered_df):,}`
        - **Total Columns:** `{len(raw_df.columns)}`
        - **Missing Values:** `{int(filtered_df.isnull().sum().sum())}` (Zero missing in benchmark)
        - **Memory Usage:** `{filtered_df.memory_usage().sum() / 1024:.2f} KB`
        """)

    with col_meta2:
        st.subheader("Attribute Glossary")
        glossary_df = pd.DataFrame([
            {"Feature": "school", "Type": "Binary", "Description": "GP (Gabriel Pereira) or MS (Mousinho da Silveira)"},
            {"Feature": "sex", "Type": "Binary", "Description": "Student sex: F (Female) or M (Male)"},
            {"Feature": "age", "Type": "Numeric", "Description": "Student age (15 to 22)"},
            {"Feature": "studytime", "Type": "Ordinal", "Description": "1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h"},
            {"Feature": "failures", "Type": "Numeric", "Description": "Past class failures (0 to 4)"},
            {"Feature": "absences", "Type": "Numeric", "Description": "School absences (0 to 93)"},
            {"Feature": "G1", "Type": "Numeric", "Description": "First period exam grade (0 to 20)"},
            {"Feature": "G2", "Type": "Numeric", "Description": "Second period exam grade (0 to 20)"},
            {"Feature": "G3", "Type": "Numeric", "Description": "Final grade (Target output, 0 to 20)"}
        ])
        st.dataframe(glossary_df, use_container_width=True, hide_index=True)

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_student_performance.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("👤 Individual Student Profile Inspector")
    st.write("Inspect any individual student record from the active cohort and compare their metrics to class-wide averages:")

    indexed_students = filtered_df.reset_index(drop=True)
    stu_labels = [
        f"Student #{i+1} | {row['school']} | {LABEL_MAPPINGS['sex'].get(row['sex'], row['sex'])} | Age {row['age']} | Final Grade: {row['G3']}/20"
        for i, row in indexed_students.iterrows()
    ]
    selected_stu_idx = st.selectbox(
        "Select Student to Inspect:",
        range(len(stu_labels)),
        format_func=lambda idx: stu_labels[idx],
        key="sb_inspect_student"
    )
    sel_student = indexed_students.iloc[selected_stu_idx]

    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    c_s1.metric(
        "Final Grade (G3)",
        f"{sel_student['G3']} / 20",
        f"{sel_student['G3'] - mean_g3:+.2f} vs Cohort Mean ({mean_g3:.1f})"
    )
    grade_trend = (
        "📈 Rising" if sel_student['G3'] > sel_student['G1']
        else "📉 Declining" if sel_student['G3'] < sel_student['G1']
        else "➖ Consistent"
    )
    c_s2.metric(
        "Term Trajectory",
        f"G1: {sel_student['G1']} → G2: {sel_student['G2']}",
        grade_trend
    )
    c_s3.metric(
        "Absences",
        f"{sel_student['absences']} days",
        f"{sel_student['absences'] - float(filtered_df['absences'].mean()):+.1f} vs Cohort Mean"
    )
    c_s4.metric(
        "Academic Status",
        "✅ Passing" if sel_student['G3'] >= 10 else "⚠️ Needs Support",
        f"Past Failures: {sel_student['failures']}"
    )

    with st.expander(f"📋 Full Background Dossier: Student #{selected_stu_idx+1}", expanded=False):
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.markdown(f"""
            **Demographics & Environment:**
            - **School:** {sel_student['school']} ({LABEL_MAPPINGS['school'].get(sel_student['school'], sel_student['school'])})
            - **Gender:** {LABEL_MAPPINGS['sex'].get(sel_student['sex'], sel_student['sex'])}
            - **Age:** {sel_student['age']}
            - **Area:** {LABEL_MAPPINGS['address'].get(sel_student['address'], sel_student['address'])}
            - **Family Size:** {LABEL_MAPPINGS['famsize'].get(sel_student['famsize'], sel_student['famsize'])}
            - **Parents Cohabitation:** {LABEL_MAPPINGS['Pstatus'].get(sel_student['Pstatus'], sel_student['Pstatus'])}
            """)
        with d_col2:
            st.markdown(f"""
            **Academic Habits & Support:**
            - **Study Time:** {LABEL_MAPPINGS['studytime'].get(sel_student['studytime'], str(sel_student['studytime']))}
            - **Past Failures:** {sel_student['failures']}
            - **School Extra Support:** {str(sel_student['schoolsup']).title()}
            - **Family Educational Support:** {str(sel_student['famsup']).title()}
            - **Paid Extra Classes:** {str(sel_student['paid']).title()}
            - **Higher Education Ambition:** {str(sel_student['higher']).title()}
            """)
        with d_col3:
            st.markdown(f"""
            **Home Life & Lifestyle:**
            - **Mother's Education:** {LABEL_MAPPINGS['Medu'].get(sel_student['Medu'], str(sel_student['Medu']))}
            - **Father's Education:** {LABEL_MAPPINGS['Fedu'].get(sel_student['Fedu'], str(sel_student['Fedu']))}
            - **Home Internet Access:** {str(sel_student['internet']).title()}
            - **In Relationship:** {str(sel_student['romantic']).title()}
            - **Free Time / Socializing:** {sel_student['freetime']}/5 | {sel_student['goout']}/5
            - **Health Rating:** {sel_student['health']}/5
            """)


# ------------------------------------------
# TAB 3: PERFORMANCE VISUALIZATIONS
# ------------------------------------------
with tab_eda:
    st.header("Exploratory Data Analysis")
    st.write("Interactive charts exploring distributions, lifestyle habits, and demographic influences.")

    # Row 1: Grade Distribution & Study Time
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.plotly_chart(plot_grade_distribution(filtered_df), use_container_width=True)
    with r1_col2:
        st.plotly_chart(plot_study_time_vs_grade(filtered_df), use_container_width=True)

    # Row 2: Absences vs Grade & Failures vs Grade
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.plotly_chart(plot_absences_vs_grade(filtered_df), use_container_width=True)
    with r2_col2:
        st.plotly_chart(plot_failures_vs_grade(filtered_df), use_container_width=True)

    # Row 3: Gender & School Comparisons
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        st.plotly_chart(plot_gender_performance(filtered_df), use_container_width=True)
    with r3_col2:
        st.plotly_chart(plot_school_comparison(filtered_df), use_container_width=True)

    # Row 4: Age Distribution & Parental Education
    r4_col1, r4_col2 = st.columns(2)
    with r4_col1:
        st.plotly_chart(plot_age_distribution(filtered_df), use_container_width=True)
    with r4_col2:
        st.plotly_chart(plot_parental_education(filtered_df), use_container_width=True)

    # Row 5: Internet Access & G1/G2/G3 Collinearity
    r5_col1, r5_col2 = st.columns(2)
    with r5_col1:
        st.plotly_chart(plot_internet_access(filtered_df), use_container_width=True)
    with r5_col2:
        st.plotly_chart(plot_grade_leakage_relationship(filtered_df), use_container_width=True)

# ------------------------------------------
# TAB 4: STATISTICAL DEEP-DIVE
# ------------------------------------------
with tab_stats:
    st.header("Comprehensive Statistical Summary")
    st.write("Parametric and non-parametric statistical metrics computed across all numeric attributes.")

    numeric_summary = compute_full_numeric_summary(filtered_df)
    st.dataframe(
        numeric_summary.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["mean", "median"]),
        use_container_width=True
    )

    stats_csv = numeric_summary.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Statistical Summary as CSV",
        data=stats_csv,
        file_name="statistical_summary_metrics.csv",
        mime="text/csv",
        key="btn_download_stats"
    )

    st.markdown("---")
    st.subheader("Correlation Analysis")

    corr_col1, corr_col2 = st.columns([3, 2])
    with corr_col1:
        st.plotly_chart(plot_correlation_heatmap(filtered_df), use_container_width=True)
    with corr_col2:
        st.write("**Top Feature Correlations with Final Grade (G3):**")
        _, target_corr = compute_correlations(filtered_df, "G3")
        target_corr_df = target_corr.to_frame(name="Pearson r").reset_index()
        target_corr_df.columns = ["Variable", "Pearson Correlation (r)"]
        st.dataframe(
            target_corr_df.style.format({"Pearson Correlation (r)": "{:.4f}"}),
            use_container_width=True,
            height=380
        )

# ------------------------------------------
# TAB 5: ML & DATA LEAKAGE LAB
# ------------------------------------------
with tab_ml:
    st.header("Machine Learning & Data Leakage Laboratory")
    st.markdown("""
    ### The Critical Academic Distinction in Student Modeling:
    - **Regime A (With G1 & G2):** Evaluates models when midterm exam scores are included. Because $G1$, $G2$, and $G3$ are consecutive period exams, $G2$ alone achieves $r > 0.90$ with $G3$. This introduces **severe temporal data leakage**. The model achieves high $R^2$ ($\sim 0.82$) simply by predicting $G3 \approx G2$.
    - **Regime B (Without G1 & G2 - Early Warning Model):** Excludes prior exam results, forcing the model to rely solely on socio-demographic indicators, study habits, parental education, and school attendance. This represents a **genuine early-warning model** deployable before the semester starts.
    """)

    if len(filtered_df) < 50:
        st.warning("⚠️ Insufficient samples for reliable train/test machine learning split. Please broaden active filters.")
    else:
        with st.spinner("Training Linear Regression and Random Forest Regressors..."):
            ml_results = get_cached_ml_results(ml_cache_key, filtered_df)

        st.subheader("Empirical Model Performance Comparison")
        st.dataframe(
            ml_results["table"].style.format({
                "MAE": "{:.4f}",
                "MSE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R2": "{:.4f}"
            }),
            use_container_width=True
        )

        ml_chart_col1, ml_chart_col2 = st.columns(2)
        with ml_chart_col1:
            st.plotly_chart(plot_model_comparison(ml_results["table"]), use_container_width=True)
        with ml_chart_col2:
            st.plotly_chart(
                plot_feature_importance(ml_results["clean"]["feature_importance"]),
                use_container_width=True
            )

        st.markdown("---")
        st.subheader(f"👥 Cohort-Wide Early Warning Risk Assessment (Evaluated Across ALL {len(filtered_df):,} Students)")
        st.write(
            "Using the trained **Regime B Clean Model (Zero Data Leakage)**, each student's final performance is "
            "forecasted solely from pre-enrollment variables (study habits, past failures, attendance, and demographics)."
        )

        # Batch prediction across all active students
        clean_model = ml_results["clean"]["Random Forest"]["model"]
        trained_features = list(clean_model.feature_names_in_)

        df_eval = handle_missing_values(filtered_df.copy())
        drop_eval_cols = [TARGET_COLUMN, "subject"] + [f"label_{c}" for c in LABEL_MAPPINGS.keys()] + GRADE_COLUMNS
        X_eval_raw = df_eval.drop(columns=[c for c in drop_eval_cols if c in df_eval.columns])
        for col in BINARY_COLUMNS:
            if col in X_eval_raw.columns:
                X_eval_raw[col] = (X_eval_raw[col].astype(str).str.lower() == "yes").astype(int)
        nom_cols = [c for c in NOMINAL_COLUMNS if c in X_eval_raw.columns]
        X_eval_encoded = pd.get_dummies(X_eval_raw, columns=nom_cols, drop_first=True, dtype=float)
        X_eval_aligned = X_eval_encoded.reindex(columns=trained_features, fill_value=0.0)

        all_pred_grades = clean_model.predict(X_eval_aligned)
        all_pred_grades = np.clip(np.round(all_pred_grades, 2), 0.0, 20.0)

        subject_series = filtered_df["subject"] if "subject" in filtered_df.columns else pd.Series(["Core"] * len(filtered_df))

        predictions_df = pd.DataFrame({
            "Student ID": [f"STU_{i+1:04d}" for i in range(len(filtered_df))],
            "Subject": subject_series.values,
            "School": filtered_df["school"].values,
            "Gender": [LABEL_MAPPINGS["sex"].get(s, s) for s in filtered_df["sex"].values],
            "Age": filtered_df["age"].values,
            "Study Time": filtered_df["studytime"].map(LABEL_MAPPINGS["studytime"]).values,
            "Failures": filtered_df["failures"].values,
            "Absences": filtered_df["absences"].values,
            "Actual G3": filtered_df["G3"].values,
            "Predicted G3": all_pred_grades,
            "Error (Actual - Pred)": np.round(filtered_df["G3"].values - all_pred_grades, 2)
        })

        def _classify_risk_tier(row):
            pred = row["Predicted G3"]
            fail = row["Failures"]
            if pred < 10.0 or fail >= 2:
                return "🔴 High Academic Risk"
            elif pred < 11.5 or fail == 1:
                return "🟡 Moderate Watch"
            else:
                return "🟢 On Track / Low Risk"

        predictions_df["Early Warning Tier"] = predictions_df.apply(_classify_risk_tier, axis=1)

        high_risk_n = int((predictions_df["Early Warning Tier"] == "🔴 High Academic Risk").sum())
        mod_risk_n  = int((predictions_df["Early Warning Tier"] == "🟡 Moderate Watch").sum())
        safe_n      = int((predictions_df["Early Warning Tier"] == "🟢 On Track / Low Risk").sum())

        kpi_r1, kpi_r2, kpi_r3, kpi_r4 = st.columns(4)
        kpi_r1.metric("Students Assessed", f"{len(predictions_df):,}", "100% of Active Cohort")
        kpi_r2.metric("🔴 High Risk Flagged", f"{high_risk_n:,}", f"{high_risk_n/len(predictions_df)*100:.1f}% need intervention")
        kpi_r3.metric("🟡 Moderate Watch", f"{mod_risk_n:,}", f"{mod_risk_n/len(predictions_df)*100:.1f}% borderline")
        kpi_r4.metric("🟢 On Track (Safe)", f"{safe_n:,}", f"{safe_n/len(predictions_df)*100:.1f}% predicted passing")

        filter_col_a, filter_col_b = st.columns([2, 2])
        with filter_col_a:
            selected_tier = st.selectbox(
                "Filter All Students by Early-Warning Status:",
                ["All Students", "🔴 High Academic Risk", "🟡 Moderate Watch", "🟢 On Track / Low Risk"]
            )
        with filter_col_b:
            stu_query = st.text_input("🔍 Search Student ID (e.g. STU_0012) or Subject / School:", "")

        view_df = predictions_df.copy()
        if selected_tier != "All Students":
            view_df = view_df[view_df["Early Warning Tier"] == selected_tier]
        if stu_query:
            query_mask = view_df.astype(str).apply(lambda r: r.str.contains(stu_query, case=False).any(), axis=1)
            view_df = view_df[query_mask]

        st.write(f"Displaying **{len(view_df):,}** students:")
        st.dataframe(
            view_df.style.background_gradient(subset=["Predicted G3", "Actual G3"], cmap="Blues"),
            use_container_width=True,
            height=380
        )

        all_preds_csv = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Risk Assessment for All {len(view_df):,} Students (CSV)",
            data=all_preds_csv,
            file_name="cohort_student_early_warning_predictions.csv",
            mime="text/csv",
            key="btn_dl_all_preds"
        )

        st.markdown("---")
        with st.expander("🎯 Simulate Single Hypothetical Student Profile (What-If Sandbox)", expanded=False):
            st.write("Adjust parameters to test individual hypothetical student scenarios against the Regime B model:")
            pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
            with pred_col1:
                input_school = st.selectbox("School", ["GP", "MS"], key="pred_school")
                input_sex = st.selectbox("Sex", ["F", "M"], key="pred_sex")
                input_age = st.slider("Age", 15, 22, 16, key="pred_age")
            with pred_col2:
                input_study = st.selectbox("Study Time", [1, 2, 3, 4], format_func=lambda x: LABEL_MAPPINGS["studytime"][x], key="pred_study")
                input_failures = st.selectbox("Past Failures", [0, 1, 2, 3], key="pred_fail")
                input_absences = st.slider("Absences", 0, 50, 4, key="pred_abs")
            with pred_col3:
                input_medu = st.selectbox("Mother Education", [0, 1, 2, 3, 4], format_func=lambda x: LABEL_MAPPINGS["Medu"][x], key="pred_medu")
                input_fedu = st.selectbox("Father Education", [0, 1, 2, 3, 4], format_func=lambda x: LABEL_MAPPINGS["Fedu"][x], key="pred_fedu")
                input_internet = st.selectbox("Internet Access", ["yes", "no"], key="pred_net")
            with pred_col4:
                input_higher = st.selectbox("Aims for Higher Ed", ["yes", "no"], key="pred_high")
                input_romantic = st.selectbox("In Relationship", ["yes", "no"], key="pred_rom")
                input_freetime = st.slider("Free Time (1-5)", 1, 5, 3, key="pred_free")

            sample_dict = {
                "age": input_age,
                "Medu": input_medu,
                "Fedu": input_fedu,
                "traveltime": 1,
                "studytime": input_study,
                "failures": input_failures,
                "schoolsup": 0,
                "famsup": 1,
                "paid": 0,
                "activities": 1,
                "nursery": 1,
                "higher": 1 if input_higher == "yes" else 0,
                "internet": 1 if input_internet == "yes" else 0,
                "romantic": 1 if input_romantic == "yes" else 0,
                "famrel": 4,
                "freetime": input_freetime,
                "goout": 3,
                "Dalc": 1,
                "Walc": 2,
                "health": 4,
                "absences": input_absences,
                "school_MS": 1 if input_school == "MS" else 0,
                "sex_M": 1 if input_sex == "M" else 0,
                "address_U": 1,
                "famsize_LE3": 0,
                "Pstatus_T": 1,
                "Mjob_health": 0,
                "Mjob_other": 1,
                "Mjob_services": 0,
                "Mjob_teacher": 0,
                "Fjob_health": 0,
                "Fjob_other": 1,
                "Fjob_services": 0,
                "Fjob_teacher": 0,
                "reason_home": 0,
                "reason_other": 0,
                "reason_reputation": 1,
                "guardian_mother": 1,
                "guardian_other": 0
            }

            aligned_values = [sample_dict.get(fn, 0) for fn in trained_features]
            input_array = np.array(aligned_values).reshape(1, -1)
            sim_pred_grade = float(clean_model.predict(input_array)[0])
            st.success(
                f"🎯 **Simulated Student Predicted Grade (G3):** `{sim_pred_grade:.2f} / 20` "
                f"({'✅ Passing / Low Risk' if sim_pred_grade >= 10 else '⚠️ High Academic Risk - Early Intervention Advised'})"
            )

# ------------------------------------------
# TAB 6: AUTOMATED DATA-DRIVEN INSIGHTS
# ------------------------------------------
with tab_insights:
    st.header("Automated Data-Driven Observations")
    st.write("Dynamic insights calculated mathematically from the active student records.")

    insights_list = generate_automated_insights(filtered_df)

    for i, ins in enumerate(insights_list, 1):
        with st.expander(f"📌 {ins['category']}: {ins['title']}", expanded=True):
            st.write(ins["detail"])

    st.markdown("---")
    st.info("""
    **Analytical Caution on Open Source Data Science:**
    These calculated insights demonstrate correlational trends within the selected sample cohort.
    They should not be construed as immutable causal mechanisms. Educational performance is multifaceted,
    and institutional interventions should encompass qualitative, socio-economic, and psychological support.
    """)


# ------------------------------------------
# TAB 7: PRESENTATION
# ------------------------------------------
with tab_presentation:
    # ── Compute all live metrics from the currently filtered dataset ──
    total_students = len(filtered_df)
    raw_total = len(raw_df)
    avg_g3    = float(filtered_df["G3"].mean())
    pass_pct  = float((filtered_df["G3"] >= 10).mean() * 100)
    fail_cnt  = int((filtered_df["G3"] == 0).sum())
    std_g3_p  = float(filtered_df["G3"].std())

    female_cnt = int((filtered_df["sex"] == "F").sum())
    male_cnt   = int((filtered_df["sex"] == "M").sum())
    female_pct = female_cnt / total_students * 100
    male_pct   = male_cnt   / total_students * 100
    f_avg = float(filtered_df[filtered_df["sex"] == "F"]["G3"].mean())
    m_avg = float(filtered_df[filtered_df["sex"] == "M"]["G3"].mean())

    urban_cnt = int((filtered_df["address"] == "U").sum())
    rural_cnt = int((filtered_df["address"] == "R").sum())
    urban_pct = urban_cnt / total_students * 100
    u_avg = float(filtered_df[filtered_df["address"] == "U"]["G3"].mean())
    r_avg = float(filtered_df[filtered_df["address"] == "R"]["G3"].mean())

    inet_yes = int((filtered_df["internet"] == "yes").sum())
    inet_pct = inet_yes / total_students * 100
    inet_avg = float(filtered_df[filtered_df["internet"] == "yes"]["G3"].mean())
    no_inet_avg = float(filtered_df[filtered_df["internet"] == "no"]["G3"].mean())

    # Study time averages
    st_avgs = {k: round(filtered_df[filtered_df["studytime"] == k]["G3"].mean(), 2)
               for k in sorted(filtered_df["studytime"].unique())}
    st_counts = filtered_df["studytime"].value_counts().sort_index().to_dict()

    # Failures averages
    fail_avgs = {k: round(filtered_df[filtered_df["failures"] == k]["G3"].mean(), 2)
                 for k in sorted(filtered_df["failures"].unique())}
    fail_counts = filtered_df["failures"].value_counts().sort_index().to_dict()

    # Correlations
    num_cols_p = filtered_df.select_dtypes(include="number").columns
    corr_g3 = filtered_df[num_cols_p].corr()["G3"].drop("G3").abs().sort_values(ascending=False)

    # Grade buckets
    bins_p  = [0, 9, 11, 13, 15, 20]
    lbls_p  = ["Fail (0\u20139)", "Pass (10\u201311)", "Average (12\u201313)", "Good (14\u201315)", "Excellent (16\u201320)"]
    buckets = pd.cut(filtered_df["G3"], bins=bins_p, labels=lbls_p, include_lowest=True).value_counts()

    # Romantic
    rom_yes_avg = float(filtered_df[filtered_df["romantic"] == "yes"]["G3"].mean()) if "romantic" in filtered_df else 0.0
    rom_no_avg  = float(filtered_df[filtered_df["romantic"] == "no"]["G3"].mean())  if "romantic" in filtered_df else 0.0

    study_labels_map = {1: "<2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: ">10 hrs"}
    fail_labels_map  = {0: "0 failures", 1: "1 failure", 2: "2 failures", 3: "3 failures"}

    # ── Slide navigation ──
    SLIDES = [
        "1 · Title",
        "2 · Project Overview",
        "3 · The Dataset",
        "4 · Student Demographics",
        "5 · Grade Analysis",
        "6 · What Affects Grades?",
        "7 · Study Time Impact",
        "8 · Failures & Absences",
        "9 · Social Insights",
        "10 · ML Results",
        "11 · Open-Source Workflow",
        "12 · Summary",
    ]

    SLIDE_CSS = """
    <style>
    .slide-box {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2f45 100%);
        border-radius: 16px;
        padding: 40px 48px;
        color: #ffffff;
        min-height: 420px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .slide-title {
        font-size: 2.0rem;
        font-weight: 800;
        color: #00c9a7;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .slide-subtitle {
        font-size: 1.1rem;
        color: #90caf9;
        margin-bottom: 24px;
    }
    .slide-body { font-size: 1.0rem; line-height: 1.7; }
    .stat-card {
        background: #1e3a5f;
        border-left: 4px solid #00c9a7;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 6px 0;
    }
    .stat-number { font-size: 1.6rem; font-weight: 700; color: #ffd166; }
    .stat-label  { font-size: 0.85rem; color: #90caf9; }
    .insight-box {
        background: #0a3d2e;
        border: 1px solid #00c9a7;
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 16px;
        font-style: italic;
        color: #b2dfdb;
    }
    .slide-number {
        font-size: 0.75rem;
        color: #546e7a;
        text-align: right;
        margin-top: 24px;
    }
    </style>
    """
    st.markdown(SLIDE_CSS, unsafe_allow_html=True)

    col_nav_prev, col_nav_select, col_nav_next = st.columns([1, 6, 1])
    with col_nav_select:
        slide_choice = st.select_slider(
            "Navigate Slides",
            options=SLIDES,
            label_visibility="collapsed"
        )
    slide_idx = SLIDES.index(slide_choice) + 1

    with col_nav_prev:
        if st.button("◀ Prev", use_container_width=True) and slide_idx > 1:
            slide_choice = SLIDES[slide_idx - 2]
            slide_idx -= 1
    with col_nav_next:
        if st.button("Next ▶", use_container_width=True) and slide_idx < len(SLIDES):
            slide_choice = SLIDES[slide_idx]
            slide_idx += 1

    st.markdown(f"**Slide {slide_idx} of {len(SLIDES)}** — *{slide_choice}*")
    st.markdown("---")

    # ── SLIDE 1: TITLE ──
    if slide_idx == 1:
        st.markdown(f"""
        <div class="slide-box">
            <div style="text-align:center; padding: 20px 0;">
                <div style="font-size:3.5rem;">🎓</div>
                <div class="slide-title" style="font-size:2.4rem; text-align:center;">
                    Student Performance Analysis Dashboard
                </div>
                <div class="slide-subtitle" style="text-align:center;">
                    Open Source Technologies (OST) Academic Project
                </div>
                <hr style="border-color:#1e3a5f; margin: 20px auto; width: 60%;">
                <div style="color:#cfd8dc; font-size:1.05rem;">
                    <b>Bhedheer Bhushan Jain</b> &nbsp;|&nbsp; PRN: 25030422033
                </div>
                <div style="margin-top:16px;">
                    <span style="background:#1e3a5f; color:#00c9a7; padding:4px 12px; border-radius:20px; font-size:0.85rem; margin:4px;">Python</span>
                    <span style="background:#1e3a5f; color:#00c9a7; padding:4px 12px; border-radius:20px; font-size:0.85rem; margin:4px;">Streamlit</span>
                    <span style="background:#1e3a5f; color:#00c9a7; padding:4px 12px; border-radius:20px; font-size:0.85rem; margin:4px;">Git & GitHub</span>
                    <span style="background:#1e3a5f; color:#00c9a7; padding:4px 12px; border-radius:20px; font-size:0.85rem; margin:4px;">Machine Learning</span>
                    <span style="background:#1e3a5f; color:#00c9a7; padding:4px 12px; border-radius:20px; font-size:0.85rem; margin:4px;">Docker</span>
                    <span style="background:#1e3a5f; color:#00c9a7; padding:4px 12px; border-radius:20px; font-size:0.85rem; margin:4px;">CI/CD</span>
                </div>
                <div style="margin-top:24px; color:#546e7a; font-size:0.8rem;">
                    🔗 github.com/bhedheerbhushanjain-svg/student-performance-dashboard
                </div>
            </div>
            <div class="slide-number">Slide 1 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 2: PROJECT OVERVIEW ──
    elif slide_idx == 2:
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">What Did We Build?</div>
            <div class="slide-subtitle">A complete open-source data analysis system for student performance</div>
            <div class="slide-body">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:8px;">
                <div class="stat-card">
                    <div style="font-size:1.5rem;">📊</div>
                    <b style="color:#00c9a7;">Data Analysis</b><br>
                    UCI Student Performance Dataset &mdash; <b>{raw_total:,}</b> real student records from Portuguese schools
                </div>
                <div class="stat-card">
                    <div style="font-size:1.5rem;">🐍</div>
                    <b style="color:#00c9a7;">Python Pipeline</b><br>
                    pandas, NumPy, Plotly, scikit-learn &mdash; full modular src/ architecture with 21 automated tests
                </div>
                <div class="stat-card">
                    <div style="font-size:1.5rem;">🌐</div>
                    <b style="color:#00c9a7;">Live Dashboard</b><br>
                    Streamlit web app with 7 interactive tabs, sidebar filters, ML predictor & insights engine
                </div>
            </div>
            <div class="insight-box">
                💡 This project demonstrates the complete open-source development lifecycle: Git branching,
                GitHub CI/CD, MIT licensing, issue templates, Docker containerisation, and reproducible research.
            </div>
            </div>
            <div class="slide-number">Slide 2 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 3: DATASET ──
    elif slide_idx == 3:
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">About the Data</div>
            <div class="slide-subtitle">UCI Machine Learning Repository — Cortez & Silva, 2008 (CC BY 4.0)</div>
            <div class="slide-body" style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
                <div>
                    <b style="color:#00c9a7;">Dataset at a Glance</b>
                    <div class="stat-card" style="margin-top:12px;">
                        <span class="stat-number">{raw_total:,}</span>
                        <span class="stat-label"> total student records</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">33</span>
                        <span class="stat-label"> features per student</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">2</span>
                        <span class="stat-label"> subjects: Mathematics + Portuguese</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">G3</span>
                        <span class="stat-label"> is the target — Final Grade (0–20 scale)</span>
                    </div>
                </div>
                <div>
                    <b style="color:#00c9a7;">Features Include</b>
                    <ul style="margin-top:12px; color:#cfd8dc; line-height:2.0;">
                        <li>Age, sex, home address (Urban / Rural)</li>
                        <li>Parents&apos; education &amp; job type</li>
                        <li>Weekly study time &amp; free time</li>
                        <li>Number of past class failures</li>
                        <li>School &amp; family support received</li>
                        <li>Internet access at home</li>
                        <li>Health status &amp; school absences</li>
                        <li>Term grades: G1 &rarr; G2 &rarr; G3 (Final)</li>
                    </ul>
                </div>
            </div>
            <div class="slide-number">Slide 3 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 4: DEMOGRAPHICS ──
    elif slide_idx == 4:
        age_dist = filtered_df["age"].value_counts().sort_index()
        age_rows = " ".join([
            f"<td style='padding:4px 10px; text-align:center;'>{a}</td><td style='padding:4px 10px; text-align:center; color:#ffd166;'>{c}</td>"
            for a, c in age_dist.items()
        ])
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">Who Are the Students?</div>
            <div class="slide-subtitle">Demographics from {total_students:,} currently filtered records</div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:14px; margin-bottom:20px;">
                <div class="stat-card" style="text-align:center;">
                    <div class="stat-number">{female_cnt}</div>
                    <div class="stat-label">👩 Female ({female_pct:.1f}%) — avg {f_avg:.2f}</div>
                </div>
                <div class="stat-card" style="text-align:center;">
                    <div class="stat-number">{male_cnt}</div>
                    <div class="stat-label">👦 Male ({male_pct:.1f}%) — avg {m_avg:.2f}</div>
                </div>
                <div class="stat-card" style="text-align:center;">
                    <div class="stat-number">{urban_cnt}</div>
                    <div class="stat-label">🏙️ Urban ({urban_pct:.1f}%) — avg {u_avg:.2f}</div>
                </div>
                <div class="stat-card" style="text-align:center;">
                    <div class="stat-number">{rural_cnt}</div>
                    <div class="stat-label">🌾 Rural ({100-urban_pct:.1f}%) — avg {r_avg:.2f}</div>
                </div>
            </div>
            <b style="color:#00c9a7;">Age Distribution</b>
            <table style="margin-top:10px; border-collapse:collapse; width:100%; color:#cfd8dc;">
                <tr style="color:#90caf9;">
                    {"".join([f"<th style='padding:4px 10px;'>Age {a}</th>" for a in age_dist.index])}
                </tr>
                <tr>{age_rows}</tr>
            </table>
            <div class="insight-box">
                🌐 Internet at home: <b>{inet_yes} students ({inet_pct:.1f}%)</b> —
                students with internet average <b>{inet_avg:.2f}</b> vs <b>{no_inet_avg:.2f}</b> without
                (+{inet_avg - no_inet_avg:.2f} grade points advantage)
            </div>
            <div class="slide-number">Slide 4 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 5: GRADE ANALYSIS ──
    elif slide_idx == 5:
        bucket_rows = ""
        colors = {"Fail (0–9)": "#ef5350", "Pass (10–11)": "#ffd166",
                  "Average (12–13)": "#42a5f5", "Good (14–15)": "#66bb6a", "Excellent (16–20)": "#00c9a7"}
        for lbl in lbls_p:
            cnt = int(buckets.get(lbl, 0))
            pct = cnt / total_students * 100
            color = colors.get(lbl, "#ffffff")
            bar_w = int(pct * 3)
            bucket_rows += f"""
            <tr>
                <td style="padding:5px 12px; color:{color};">{lbl}</td>
                <td style="padding:5px 12px; color:#ffd166; font-weight:700;">{cnt}</td>
                <td style="padding:5px 12px;">{pct:.1f}%</td>
                <td style="padding:5px 12px;">
                    <div style="background:{color}; height:14px; width:{bar_w}px; border-radius:4px;"></div>
                </td>
            </tr>"""
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">How Did Students Perform?</div>
            <div class="slide-subtitle">Final Grade (G3) distribution across {total_students:,} students</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
                <div>
                    <table style="border-collapse:collapse; color:#cfd8dc; width:100%;">
                        <tr style="color:#90caf9; border-bottom:1px solid #1e3a5f;">
                            <th style="padding:5px 12px; text-align:left;">Range</th>
                            <th style="padding:5px 12px;">Count</th>
                            <th style="padding:5px 12px;">%</th>
                            <th style="padding:5px 12px; text-align:left;">Bar</th>
                        </tr>
                        {bucket_rows}
                    </table>
                </div>
                <div>
                    <div class="stat-card" style="text-align:center; margin-bottom:12px;">
                        <span class="stat-number">{avg_g3:.2f} / 20</span>
                        <div class="stat-label">Overall Average Final Grade</div>
                    </div>
                    <div class="stat-card" style="text-align:center; margin-bottom:12px;">
                        <span class="stat-number">{pass_pct:.1f}%</span>
                        <div class="stat-label">Overall Pass Rate (G3 ≥ 10)</div>
                    </div>
                    <div class="stat-card" style="text-align:center;">
                        <span class="stat-number">{std_g3_p:.2f}</span>
                        <div class="stat-label">Standard Deviation of G3</div>
                    </div>
                </div>
            </div>
            <div class="slide-number">Slide 5 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 6: CORRELATIONS ──
    elif slide_idx == 6:
        top10 = corr_g3.head(10)
        corr_rows = ""
        for feat, val in top10.items():
            bar_w = int(val * 280)
            color = "#ef5350" if val > 0.5 else "#ffd166" if val > 0.2 else "#66bb6a"
            corr_rows += f"""
            <tr>
                <td style="padding:4px 10px; color:#cfd8dc;">{feat}</td>
                <td style="padding:4px 10px; color:#ffd166; font-weight:700;">{val:.3f}</td>
                <td style="padding:4px 10px;">
                    <div style="background:{color}; height:12px; width:{bar_w}px; border-radius:4px;"></div>
                </td>
            </tr>"""
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">What Affects Final Grades?</div>
            <div class="slide-subtitle">Absolute Pearson correlation with G3 — calculated from active {total_students:,} students</div>
            <table style="border-collapse:collapse; width:100%; margin-top:8px;">
                <tr style="color:#90caf9; border-bottom:1px solid #1e3a5f;">
                    <th style="padding:4px 10px; text-align:left;">Feature</th>
                    <th style="padding:4px 10px;">|r|</th>
                    <th style="padding:4px 10px; text-align:left;">Strength</th>
                </tr>
                {corr_rows}
            </table>
            <div class="insight-box">
                💡 G1 and G2 (prior term grades) dominate due to temporal collinearity.
                For genuine early-warning prediction, study time and failures are the strongest
                actionable signals available <i>before</i> the school year begins.
            </div>
            <div class="slide-number">Slide 6 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 7: STUDY TIME ──
    elif slide_idx == 7:
        study_rows = ""
        for k in sorted(st_avgs.keys()):
            lbl = study_labels_map.get(k, str(k))
            cnt = st_counts.get(k, 0)
            avg = st_avgs[k]
            bar_w = int((avg / 20) * 240)
            study_rows += f"""
            <tr>
                <td style="padding:6px 12px; color:#90caf9;">{lbl}</td>
                <td style="padding:6px 12px; color:#cfd8dc;">{cnt} students</td>
                <td style="padding:6px 12px; color:#ffd166; font-weight:700;">{avg}</td>
                <td style="padding:6px 12px;">
                    <div style="background:#00c9a7; height:14px; width:{bar_w}px; border-radius:4px;"></div>
                </td>
            </tr>"""
        min_k = min(st_avgs.keys())
        max_k = max(st_avgs.keys())
        diff  = round(st_avgs[max_k] - st_avgs[min_k], 2)
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">Does More Study = Better Grades?</div>
            <div class="slide-subtitle">Weekly study time vs average final grade — {total_students:,} students</div>
            <table style="border-collapse:collapse; width:100%; margin-top:8px;">
                <tr style="color:#90caf9; border-bottom:1px solid #1e3a5f;">
                    <th style="padding:6px 12px; text-align:left;">Study Time / Week</th>
                    <th style="padding:6px 12px;">Students</th>
                    <th style="padding:6px 12px;">Avg G3</th>
                    <th style="padding:6px 12px; text-align:left;">Visual</th>
                </tr>
                {study_rows}
            </table>
            <div class="insight-box">
                📈 Students in the highest study-time bracket score on average
                <b>+{diff} grade points</b> more than the lowest bracket —
                an improvement of <b>{diff/st_avgs[min_k]*100:.1f}%</b>.
            </div>
            <div class="slide-number">Slide 7 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 8: FAILURES & ABSENCES ──
    elif slide_idx == 8:
        fail_rows = ""
        for k in sorted(fail_avgs.keys()):
            lbl = fail_labels_map.get(k, f"{k} failures")
            cnt = fail_counts.get(k, 0)
            avg = fail_avgs[k]
            pct_cnt = cnt / total_students * 100
            color = "#ef5350" if k > 0 else "#66bb6a"
            fail_rows += f"""
            <tr>
                <td style="padding:6px 12px; color:{color};">{lbl}</td>
                <td style="padding:6px 12px; color:#cfd8dc;">{cnt} ({pct_cnt:.1f}%)</td>
                <td style="padding:6px 12px; color:#ffd166; font-weight:700;">{avg}</td>
            </tr>"""
        avg_abs = round(float(filtered_df["absences"].mean()), 2)
        max_abs = int(filtered_df["absences"].max())
        zero_abs = int((filtered_df["absences"] == 0).sum())
        drop_val = round(fail_avgs.get(0, 0) - fail_avgs.get(1, 0), 2) if 1 in fail_avgs else 0
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">The Cost of Failures & Absences</div>
            <div class="slide-subtitle">Past failures and attendance — computed from {total_students:,} students</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
                <div>
                    <b style="color:#00c9a7;">Past Class Failures → Final Grade</b>
                    <table style="border-collapse:collapse; width:100%; margin-top:10px;">
                        <tr style="color:#90caf9; border-bottom:1px solid #1e3a5f;">
                            <th style="padding:6px 12px; text-align:left;">Failures</th>
                            <th style="padding:6px 12px;">Students</th>
                            <th style="padding:6px 12px;">Avg G3</th>
                        </tr>
                        {fail_rows}
                    </table>
                    <div style="margin-top:12px; color:#ef9a9a; font-size:0.9rem;">
                        ⚠️ Even 1 past failure drops average grade by <b>{drop_val} points</b>
                    </div>
                </div>
                <div>
                    <b style="color:#00c9a7;">School Absences</b>
                    <div class="stat-card" style="text-align:center; margin:12px 0;">
                        <span class="stat-number">{avg_abs}</span>
                        <div class="stat-label">Average absences per student</div>
                    </div>
                    <div class="stat-card" style="text-align:center; margin-bottom:12px;">
                        <span class="stat-number">{max_abs}</span>
                        <div class="stat-label">Maximum absences recorded</div>
                    </div>
                    <div class="stat-card" style="text-align:center;">
                        <span class="stat-number">{zero_abs}</span>
                        <div class="stat-label">Students with zero absences ({zero_abs/total_students*100:.1f}%)</div>
                    </div>
                </div>
            </div>
            <div class="slide-number">Slide 8 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 9: SOCIAL INSIGHTS ──
    elif slide_idx == 9:
        rom_yes_cnt = int((filtered_df["romantic"] == "yes").sum()) if "romantic" in filtered_df.columns else 0
        rom_no_cnt  = int((filtered_df["romantic"] == "no").sum())  if "romantic" in filtered_df.columns else 0
        rom_diff    = round(rom_no_avg - rom_yes_avg, 2)
        inet_diff   = round(inet_avg - no_inet_avg, 2)
        addr_diff   = round(u_avg - r_avg, 2)
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">Surprising Social Insights</div>
            <div class="slide-subtitle">Lifestyle factors and their measurable impact on grades — {total_students:,} students</div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:16px;">
                <div class="stat-card">
                    <div style="font-size:1.8rem; text-align:center;">💑</div>
                    <b style="color:#00c9a7;">Romantic Relationships</b>
                    <ul style="margin-top:10px; color:#cfd8dc; line-height:2.0; padding-left:16px;">
                        <li>In a relationship: <b>{rom_yes_cnt}</b> students</li>
                        <li>Avg grade: <b style="color:#ffd166;">{rom_yes_avg:.2f}</b></li>
                        <li>Not in one: <b>{rom_no_cnt}</b> students</li>
                        <li>Avg grade: <b style="color:#ffd166;">{rom_no_avg:.2f}</b></li>
                    </ul>
                    <div style="color:#ef9a9a; margin-top:8px;">
                        &minus;{rom_diff:.2f} pts for those in relationships
                    </div>
                </div>
                <div class="stat-card">
                    <div style="font-size:1.8rem; text-align:center;">🌐</div>
                    <b style="color:#00c9a7;">Internet Access</b>
                    <ul style="margin-top:10px; color:#cfd8dc; line-height:2.0; padding-left:16px;">
                        <li>With internet: <b>{inet_yes}</b> ({inet_pct:.1f}%)</li>
                        <li>Avg grade: <b style="color:#ffd166;">{inet_avg:.2f}</b></li>
                        <li>Without internet: <b>{total_students - inet_yes}</b></li>
                        <li>Avg grade: <b style="color:#ffd166;">{no_inet_avg:.2f}</b></li>
                    </ul>
                    <div style="color:#66bb6a; margin-top:8px;">
                        +{inet_diff:.2f} pts advantage with internet
                    </div>
                </div>
                <div class="stat-card">
                    <div style="font-size:1.8rem; text-align:center;">🏠</div>
                    <b style="color:#00c9a7;">Urban vs Rural</b>
                    <ul style="margin-top:10px; color:#cfd8dc; line-height:2.0; padding-left:16px;">
                        <li>Urban: <b>{urban_cnt}</b> ({urban_pct:.1f}%)</li>
                        <li>Avg grade: <b style="color:#ffd166;">{u_avg:.2f}</b></li>
                        <li>Rural: <b>{rural_cnt}</b> ({100-urban_pct:.1f}%)</li>
                        <li>Avg grade: <b style="color:#ffd166;">{r_avg:.2f}</b></li>
                    </ul>
                    <div style="color:#66bb6a; margin-top:8px;">
                        +{addr_diff:.2f} pts advantage for urban students
                    </div>
                </div>
            </div>
            <div class="insight-box">
                🔍 Socioeconomic access gaps (internet, location) create measurable academic disadvantages.
                These are systemic factors, not individual choices.
            </div>
            <div class="slide-number">Slide 9 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 10: ML RESULTS ──
    elif slide_idx == 10:
        ml_note = ""
        if total_students >= 50:
            try:
                ml_res_p = get_cached_ml_results(ml_cache_key, filtered_df)
                tbl = ml_res_p["table"]
                rf_leak_r2  = float(tbl.loc[(tbl["Regime"]=="Leaky") & (tbl["Model"]=="Random Forest"), "R2"].values[0])
                lr_leak_r2  = float(tbl.loc[(tbl["Regime"]=="Leaky") & (tbl["Model"]=="Linear Regression"), "R2"].values[0])
                rf_clean_r2 = float(tbl.loc[(tbl["Regime"]=="Clean") & (tbl["Model"]=="Random Forest"), "R2"].values[0])
                lr_clean_r2 = float(tbl.loc[(tbl["Regime"]=="Clean") & (tbl["Model"]=="Linear Regression"), "R2"].values[0])
                ml_note = f"""
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-top:16px;">
                    <div>
                        <b style="color:#ef9a9a;">⚠️ Regime A — With G1 &amp; G2 (Data Leakage)</b>
                        <div class="stat-card" style="margin-top:10px;">
                            <div>🌳 Random Forest R²: <span class="stat-number">{rf_leak_r2:.4f}</span></div>
                        </div>
                        <div class="stat-card">
                            <div>📉 Linear Regression R²: <span class="stat-number">{lr_leak_r2:.4f}</span></div>
                        </div>
                        <div style="color:#ef9a9a; font-size:0.85rem; margin-top:8px;">
                            High accuracy — but G2≈G3 creates temporal leakage
                        </div>
                    </div>
                    <div>
                        <b style="color:#66bb6a;">✅ Regime B — Without G1 &amp; G2 (Honest Early-Warning)</b>
                        <div class="stat-card" style="margin-top:10px;">
                            <div>🌳 Random Forest R²: <span class="stat-number">{rf_clean_r2:.4f}</span></div>
                        </div>
                        <div class="stat-card">
                            <div>📉 Linear Regression R²: <span class="stat-number">{lr_clean_r2:.4f}</span></div>
                        </div>
                        <div style="color:#66bb6a; font-size:0.85rem; margin-top:8px;">
                            Genuine model using only background factors
                        </div>
                    </div>
                </div>"""
            except Exception:
                ml_note = "<div class='insight-box'>⚠️ ML results unavailable for current filter selection.</div>"
        else:
            ml_note = "<div class='insight-box'>⚠️ Too few records for ML training. Broaden filters to see live results.</div>"

        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">Can We Predict Student Grades?</div>
            <div class="slide-subtitle">Two ML regimes — with and without data leakage — trained on {total_students:,} students</div>
            {ml_note}
            <div class="insight-box" style="margin-top:16px;">
                🎯 Key Takeaway: Early intervention models (Regime B) are weaker but more honest and deployable.
                Study habits, failures, and background are the real actionable signals.
            </div>
            <div class="slide-number">Slide 10 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 11: OPEN-SOURCE WORKFLOW ──
    elif slide_idx == 11:
        st.markdown(f"""
        <div class="slide-box">
            <div class="slide-title">Complete Open-Source Development Lifecycle</div>
            <div class="slide-subtitle">Every OST concept demonstrated end-to-end</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:12px;">
                <div>
                    <b style="color:#00c9a7;">Git &amp; GitHub</b>
                    <ul style="color:#cfd8dc; line-height:2.0; padding-left:16px;">
                        <li>8 branches: <code>main</code>, <code>develop</code>, <code>feature/*</code>, <code>docs/*</code></li>
                        <li>Conventional commits on every branch</li>
                        <li>Pull Request merges into main</li>
                        <li>GitHub Actions CI/CD — all tests passing ✅</li>
                        <li>Multi-OS (Ubuntu + Windows) × Python 3.10/3.11/3.12</li>
                    </ul>
                    <b style="color:#00c9a7;">Documentation</b>
                    <ul style="color:#cfd8dc; line-height:2.0; padding-left:16px;">
                        <li>README.md, CHANGELOG.md, SECURITY.md</li>
                        <li>CODE_OF_CONDUCT.md, CONTRIBUTING.md</li>
                        <li>Issue &amp; PR templates in .github/</li>
                        <li>docs/: dataset, methodology, architecture, linux-commands</li>
                    </ul>
                </div>
                <div>
                    <b style="color:#00c9a7;">Code Quality</b>
                    <ul style="color:#cfd8dc; line-height:2.0; padding-left:16px;">
                        <li>21 automated pytest tests</li>
                        <li>Modular src/ architecture</li>
                        <li>MIT Open-Source License</li>
                        <li>requirements.txt with pinned versions</li>
                    </ul>
                    <b style="color:#00c9a7;">Containerisation</b>
                    <ul style="color:#cfd8dc; line-height:2.0; padding-left:16px;">
                        <li>Dockerfile (python:3.11-slim, non-root)</li>
                        <li>docker-compose.yml</li>
                        <li>Makefile for one-command dev workflow</li>
                    </ul>
                </div>
            </div>
            <div class="slide-number">Slide 11 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SLIDE 12: SUMMARY ──
    elif slide_idx == 12:
        st.markdown(f"""
        <div class="slide-box">
            <div style="text-align:center;">
                <div class="slide-title" style="text-align:center;">Project Summary</div>
                <div class="slide-subtitle" style="text-align:center;">
                    What we built, what we proved, and what we learned
                </div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-top:16px;">
                <div>
                    <b style="color:#00c9a7;">Deliverables</b>
                    <ul style="color:#cfd8dc; line-height:2.2; padding-left:16px; margin-top:8px;">
                        <li>✅ {raw_total:,} real students analysed</li>
                        <li>✅ 12 interactive Plotly visualisations</li>
                        <li>✅ 2 ML models (RF + Linear Regression)</li>
                        <li>✅ 21 automated tests — all passing</li>
                        <li>✅ 8 Git branches with full commit history</li>
                        <li>✅ GitHub Actions CI (3 OS × 3 Python versions)</li>
                        <li>✅ Streamlit live dashboard — 7 interactive tabs</li>
                    </ul>
                </div>
                <div>
                    <b style="color:#00c9a7;">OST Concepts Covered</b>
                    <ul style="color:#cfd8dc; line-height:2.2; padding-left:16px; margin-top:8px;">
                        <li>✅ Git, GitHub, Branching, Pull Requests</li>
                        <li>✅ Open-Source License (MIT)</li>
                        <li>✅ Issue Templates &amp; PR Templates</li>
                        <li>✅ Code of Conduct &amp; Contributing Guide</li>
                        <li>✅ Docker / Containerisation</li>
                        <li>✅ Automated Testing &amp; CI/CD</li>
                        <li>✅ Data Analysis &amp; ML in Python</li>
                    </ul>
                </div>
            </div>
            <div style="text-align:center; margin-top:24px; color:#546e7a; font-size:0.9rem;">
                🔗 github.com/bhedheerbhushanjain-svg/student-performance-dashboard
            </div>
            <div style="text-align:center; margin-top:12px; font-size:1.2rem; color:#00c9a7; font-weight:700;">
                ✨ Thank You — Bhedheer Bhushan Jain | PRN: 25030422033
            </div>
            <div class="slide-number">Slide 12 / {len(SLIDES)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"📊 Presentation data is live — computed from **{total_students:,}** currently filtered student records. Use sidebar to change cohort or filters.")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Student Performance Analysis Dashboard | Academic OST Project | Author: Bhedheer Bhushan Jain (PRN: 25030422033) | MIT License</small></center>",
    unsafe_allow_html=True
)
