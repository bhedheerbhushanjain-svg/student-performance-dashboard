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
    add_readable_labels
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
    if dataset_choice == "Mathematics (student-mat.csv)":
        return load_raw_data("mat")
    elif dataset_choice == "Portuguese (student-por.csv)":
        return load_raw_data("por")
    elif dataset_choice == "Combined Dataset (Both Subjects)":
        return load_both_courses()
    elif dataset_choice == "Matched Cohort (Overlapping Students)":
        return load_merged_cohort()
    return load_raw_data("mat")


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
        "Mathematics (student-mat.csv)",
        "Portuguese (student-por.csv)",
        "Combined Dataset (Both Subjects)",
        "Matched Cohort (Overlapping Students)"
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
tab_overview, tab_data, tab_eda, tab_stats, tab_ml, tab_insights = st.tabs([
    "🏠 Overview & Architecture",
    "📁 Dataset Explorer",
    "📈 Performance Visualizations",
    "🧮 Statistical Deep-Dive",
    "🔬 ML & Data Leakage Lab",
    "💡 Automated Insights"
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
            ml_results = compare_leakage_regimes(filtered_df, random_state=42)

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
        st.subheader("🎯 Interactive Early Warning Predictor (Regime B - Clean)")
        st.write("Test the trained Random Forest Early Warning model on custom student profiles:")

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

        # Synthesize sample input matching feature names
        clean_model = ml_results["clean"]["Random Forest"]["model"]
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

        # Align features
        feature_names = ml_results["clean"]["feature_importance"]["feature"].tolist()
        aligned_values = [sample_dict.get(fn, 0) for fn in feature_names]
        input_array = np.array(aligned_values).reshape(1, -1)

        predicted_grade = float(clean_model.predict(input_array)[0])
        st.success(
            f"🎯 **Predicted Final Grade (G3):** `{predicted_grade:.2f} / 20` "
            f"({'✅ Passing' if predicted_grade >= 10 else '⚠️ Academic Risk - Intervention Advised'})"
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

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Student Performance Analysis Dashboard | Academic OST Project | Author: Bhedheer Bhushan Jain (PRN: 25030422033) | MIT License</small></center>",
    unsafe_allow_html=True
)
