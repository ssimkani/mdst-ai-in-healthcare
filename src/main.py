import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.write("# MDST W26 'AI in Healthcare' Demo")
st.caption("By Will McKanna and Seena Simkani")
st.write("### Analyzing data from a synthetic dataset of Alzheimer's patients. A simple Random Forest model is used to predict if an individual has Alzheimer's given a set of cognitive diagnostics/assessments.")

st.divider()

# Project Description
st.header("Project Description")
st.markdown(
  """
  This project explores the intersection of Machine Learning interpretability
  and healthcare analytics by examining various datasets. This project teaches
  core data science skills, including data cleaning/pre-processing, Exploratory
  Data Analysis (EDA), and machine learning. Through building predictive models,
  members will investigate the extent to which patient factors influence model
  outcomes using SHAP (SHapley Additive exPlanations), Principal Component Analysis
  (PCA), and MAPIE, among other methods. This project aims to develop an understanding
  of statistical patterns, model interpretability, and the broader implications of deploying
  predictive algorithms in clinical settings.
  """
)

st.divider()

# Emphasis
st.header("Model Interpretability and Uncertainty Emphasis")
st.markdown(
  """
In high-risk settings such as healthcare, we should be able to interpret the
mechanisms of ML/AI models. Given their predictive nature, we need to be incredibly
cautious when utilizing AI, as there are big implications for its misuse in healthcare.
Our project aims to create platforms where model outcomes can be interpreted and understood,
and create confidence in their use. Members will gain a better understanding of the behavior
and outcomes of “black box” models and how to interpret their internal mechanisms, leading
to safer usage.
  """
)

st.divider()

# Navigation
st.header("Navigation")

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.subheader("Data")
    st.caption(
        "Explore the synthetic Alzheimer's dataset"
    )
    st.page_link("pages/about_data.py", label="About Data")

with nav2:
    st.subheader("Model")
    st.caption(
        "How Random Forest works"
    )
    st.page_link("pages/about_model.py", label="About Model")

with nav3:
    st.subheader("Analysis")
    st.caption(
        "Exploratory Data Analysis (EDA), model interpretability tools, and metrics"
    )
    st.page_link("pages/data_analysis.py", label="Data")
    st.page_link("pages/model_analysis.py", label="Model")

with nav4:
    st.subheader("Predictions")
    st.caption(
        "Enter patient features to get a prediction (0 or 1), predicted probability, and uncertainty-aware "
        "outputs (prediction sets)"
    )
    st.page_link("pages/predictions.py", label="Predictions")
