import streamlit as st
import pandas as pd

st.set_page_config(page_title="About Data", layout="wide")

st.title("Alzheimer's Disease Dataset")
st.subheader("Comprehensive Health Information for Alzheimer's Disease")

st.markdown("""
## About Dataset
This dataset contains extensive health information for **2,149 patients**, each uniquely identified with IDs ranging from **4751 to 6900**.
It includes demographic details, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, symptoms,
and a diagnosis of Alzheimer's Disease.

---

## Table of Contents
1. Patient Information  
2. Medical History  
3. Clinical Measurements  
4. Cognitive and Functional Assessments  
5. Symptoms  
6. Diagnosis Information  
7. Confidential Information  

---

## Patient Information

### Patient ID
- **PatientID**: Unique identifier (**4751–6900**)

### Demographic Details
- **Age**: 60–90 years  
- **Gender**: 0 = Male, 1 = Female  
- **Ethnicity**:  
  - 0: Caucasian  
  - 1: African American  
  - 2: Asian  
  - 3: Other  
- **EducationLevel**:  
  - 0: None  
  - 1: High School  
  - 2: Bachelor's  
  - 3: Higher  

### Lifestyle Factors
- **BMI**: 15–40  
- **Smoking**: 0 = No, 1 = Yes  
- **AlcoholConsumption**: 0–20 units/week  
- **PhysicalActivity**: 0–10 hours/week  
- **DietQuality**: 0–10  
- **SleepQuality**: 4–10  

---

## Medical History
- **FamilyHistoryAlzheimers**: 0 = No, 1 = Yes  
- **CardiovascularDisease**: 0 = No, 1 = Yes  
- **Diabetes**: 0 = No, 1 = Yes  
- **Depression**: 0 = No, 1 = Yes  
- **HeadInjury**: 0 = No, 1 = Yes  
- **Hypertension**: 0 = No, 1 = Yes  

---

## Clinical Measurements
- **SystolicBP**: 90–180 mmHg  
- **DiastolicBP**: 60–120 mmHg  
- **CholesterolTotal**: 150–300 mg/dL  
- **CholesterolLDL**: 50–200 mg/dL  
- **CholesterolHDL**: 20–100 mg/dL  
- **CholesterolTriglycerides**: 50–400 mg/dL  

---

## Cognitive and Functional Assessments
- **MMSE**: 0–30 (lower = more impairment)  
- **FunctionalAssessment**: 0–10 (lower = more impairment)  
- **MemoryComplaints**: 0 = No, 1 = Yes  
- **BehavioralProblems**: 0 = No, 1 = Yes  
- **ADL**: 0–10 (lower = more impairment)  

---

## Symptoms
- **Confusion**: 0 = No, 1 = Yes  
- **Disorientation**: 0 = No, 1 = Yes  
- **PersonalityChanges**: 0 = No, 1 = Yes  
- **DifficultyCompletingTasks**: 0 = No, 1 = Yes  
- **Forgetfulness**: 0 = No, 1 = Yes  

---

## Diagnosis Information
- **Diagnosis**: 0 = No, 1 = Yes  

---

## Confidential Information
- **DoctorInCharge**: "XXXConfid" for all patients

---

## Citation
```bibtex
@misc{rabie_el_kharoua_2024,
  title={Alzheimer's Disease Dataset},
  url={https://www.kaggle.com/dsv/8668279},
  DOI={10.34740/KAGGLE/DSV/8668279},
  publisher={Kaggle},
  author={Rabie El Kharoua},
  year={2024}
}
```
""")


st.header("Dataset Preview")

# --- Load data ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# Change this to your real path / uploader
DATA_PATH = "./data/raw/alzheimers.csv"
df = load_data(DATA_PATH)

# Quick summaries
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

st.divider()

# Data Preview
st.subheader("Data")
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    n_rows = st.selectbox("Rows to display", [5, 10, 20, 50, 100], index=1)

st.dataframe(
    df.head(n_rows),
    width="stretch",
    hide_index=True
)

st.divider()

# column overview
st.subheader("Columns")

col_overview = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in df.dtypes],
    "non_null": df.notna().sum().values,
    "nulls": df.isna().sum().values,
    "null_%": (df.isna().mean().values * 100).round(2),
    "unique": df.nunique(dropna=True).values
}).sort_values("null_%", ascending=False)

st.dataframe(col_overview, width="stretch", hide_index=True)

st.divider()

# descriptive statistics
st.subheader("Summary Statistics")

num_cols = df.select_dtypes(include="number").columns.tolist()
if num_cols:
    st.dataframe(df[num_cols].describe().T, width="stretch")
else:
    st.info("No numeric columns found to summarize.")

st.divider()

st.header("Preprocessing")

st.markdown("The PatientID and doctorInCharge columns are dropped for exploratory data analysis.")
