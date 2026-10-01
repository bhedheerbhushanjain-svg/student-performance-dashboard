# Dataset Provenance, Schema, and Documentation

## Provenance & Attribution

The **Student Performance Analysis Dashboard** is built upon the genuine, publicly available **Student Performance Dataset** curated by Paulo Cortez and Alice Silva at the University of Minho, Portugal.

- **Primary Repository:** [UCI Machine Learning Repository - Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Secondary Access / Mirror:** [Kaggle Dataset Mirror](https://www.kaggle.com/dskagglemt/student-performance-data-set/metadata)
- **Original Research Publication:**
  > Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito & J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, EUROSIS, ISBN 978-9077381-39-7.

---

## Dataset Scope & Cohorts

The data was collected during the 2005–2006 academic year in two secondary education schools in the Alentejo region of Portugal:
1. **Gabriel Pereira (GP)**
2. **Mousinho da Silveira (MS)**

The dataset is partitioned into two core subject evaluations:
- **Mathematics (`student-mat.csv`):** 395 student instances, 33 attributes.
- **Portuguese Language (`student-por.csv`):** 649 student instances, 33 attributes.
- **Overlapping Students (`student-merge.R`):** 382 students who enrolled in both Mathematics and Portuguese courses, identifiable by matching 13 identical demographic attributes.

---

## Attribute Glossary & Data Dictionary

| # | Attribute | Variable Type | Domain / Values | Description |
|---|---|---|---|---|
| 1 | `school` | Binary | `"GP"` (Gabriel Pereira), `"MS"` (Mousinho da Silveira) | Student's school |
| 2 | `sex` | Binary | `"F"` (Female), `"M"` (Male) | Student's sex |
| 3 | `age` | Numeric | 15 to 22 | Student age in years |
| 4 | `address` | Binary | `"U"` (Urban), `"R"` (Rural) | Home address location |
| 5 | `famsize` | Binary | `"LE3"` ($\le 3$), `"GT3"` ($> 3$) | Family size |
| 6 | `Pstatus` | Binary | `"T"` (Together), `"A"` (Apart) | Parent's cohabitation status |
| 7 | `Medu` | Ordinal | `0`: none, `1`: primary (4th grade), `2`: 5th-9th grade, `3`: secondary, `4`: higher | Mother's education level |
| 8 | `Fedu` | Ordinal | `0`: none, `1`: primary (4th grade), `2`: 5th-9th grade, `3`: secondary, `4`: higher | Father's education level |
| 9 | `Mjob` | Nominal | `"teacher"`, `"health"`, `"services"`, `"at_home"`, `"other"` | Mother's occupation |
| 10 | `Fjob` | Nominal | `"teacher"`, `"health"`, `"services"`, `"at_home"`, `"other"` | Father's occupation |
| 11 | `reason` | Nominal | `"home"`, `"reputation"`, `"course"`, `"other"` | Reason for choosing school |
| 12 | `guardian` | Nominal | `"mother"`, `"father"`, `"other"` | Primary legal guardian |
| 13 | `traveltime` | Ordinal | `1`: $<15$ min, `2`: $15-30$ min, `3`: $30-60$ min, `4`: $>60$ min | Travel time to school |
| 14 | `studytime` | Ordinal | `1`: $<2$ hours, `2`: $2-5$ hours, `3`: $5-10$ hours, `4`: $>10$ hours | Weekly self-study hours |
| 15 | `failures` | Numeric | $0, 1, 2, 3$ (if $\ge 4$, recorded as $4$) | Past academic course failures |
| 16 | `schoolsup` | Binary | `"yes"`, `"no"` | Extra educational support |
| 17 | `famsup` | Binary | `"yes"`, `"no"` | Family educational support |
| 18 | `paid` | Binary | `"yes"`, `"no"` | Extra paid subject tutoring |
| 19 | `activities` | Binary | `"yes"`, `"no"` | Extracurricular activities |
| 20 | `nursery` | Binary | `"yes"`, `"no"` | Attended nursery / preschool |
| 21 | `higher` | Binary | `"yes"`, `"no"` | Aspires to pursue higher education |
| 22 | `internet` | Binary | `"yes"`, `"no"` | Internet access at home |
| 23 | `romantic` | Binary | `"yes"`, `"no"` | In a romantic relationship |
| 24 | `famrel` | Ordinal | $1$ (very bad) to $5$ (excellent) | Quality of family relationships |
| 25 | `freetime` | Ordinal | $1$ (very low) to $5$ (very high) | Free time after school |
| 26 | `goout` | Ordinal | $1$ (very low) to $5$ (very high) | Frequency of going out with friends |
| 27 | `Dalc` | Ordinal | $1$ (very low) to $5$ (very high) | Workday alcohol consumption |
| 28 | `Walc` | Ordinal | $1$ (very low) to $5$ (very high) | Weekend alcohol consumption |
| 29 | `health` | Ordinal | $1$ (very bad) to $5$ (very good) | Current health status |
| 30 | `absences` | Numeric | $0$ to $93$ | Number of school absences |
| 31 | `G1` | Numeric | $0$ to $20$ | First period exam grade |
| 32 | `G2` | Numeric | $0$ to $20$ | Second period exam grade |
| 33 | `G3` | Numeric | $0$ to $20$ (Target) | Final period exam grade |

---

## Critical Data Science Limitations & Leakage Analysis

### The $G1 - G2 - G3$ Collinearity Phenomenon
In Portuguese secondary education, the final grade $G3$ is heavily determined by ongoing cumulative performance reflected in earlier semester marks ($G1$ and $G2$). 

Empirical correlation analysis demonstrates:
- $\text{corr}(G2, G3) \approx 0.905$ (in Mathematics)
- $\text{corr}(G1, G3) \approx 0.801$ (in Mathematics)

### Two Experimental Modeling Regimes
To prevent academic deception and accurately communicate machine learning behavior, this project establishes two clear modeling regimes:

1. **Regime A (Full Academic / Late Prediction):**
   - Incorporates all features including $G1$ and $G2$.
   - Yields high predictive metrics ($R^2 \approx 0.82$, $\text{MAE} \approx 1.16$).
   - **Methodological Limitation:** Represents temporal data leakage. The model essentially functions as a near-identity approximator ($G3 \approx G2$).
2. **Regime B (Early Warning / Pre-Enrollment Model):**
   - Strictly excludes $G1$ and $G2$.
   - Uses exclusively socio-demographic indicators, past historical failures, and study habits.
   - Yields realistic baseline metrics ($R^2 \approx 0.27$, $\text{MAE} \approx 3.11$).
   - **Actionable Utility:** Identifies students at risk *before* the first period exam occurs, allowing timely institutional interventions.

---

## Data Privacy & Ethical Compliance

- **Zero Personally Identifiable Information:** The dataset is fully anonymized. No names, government IDs, addresses, or contact information exist.
- **Fairness & Bias Caution:** Variables such as parental education or home internet reflect broader socio-economic disparities. Model predictions should never be used punitive measures or automated gatekeeping.
