import streamlit as st
import pickle
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictions", layout="wide")

# Load SHAP
shap_values = joblib.load("./data/shap/shap_values.joblib")

# Load Models
with open("./models/cal_rf.pkl", "rb") as f:
    model = pickle.load(f)

# model for making class predictions or sending to human for review
with open("./models/bcc_rf.pkl", "rb") as f:
    bcc = pickle.load(f)

with open("./models/mapie_rf.pkl", "rb") as f:
    mapie = pickle.load(f)

# Load SHAP Explainer
explainer = joblib.load("./data/shap/explainer.joblib")

# Load Standardizer
scaler = joblib.load("./data/standardization/standard_scaler.joblib")
cols = joblib.load("./data/standardization/columns.joblib")

st.title("Predictions")
st.markdown("### A Random Forest Classifier that predicts whether a patient has Alzheimer's or not by using:\n##### * Class Predictions\n##### * Probability Predictions\n##### * Conformal Predictions for Uncertainty Quantification\n##### * SHAP Values for Interpretability")
st.divider()

st.markdown("### This model uses data from cognitive measurements/assessments")
st.divider()

st.markdown("### Enter inputs")
st.divider()

with st.form("Patient Inputs"):
    c1, c2 = st.columns(2)

    with c1:
        mmse = st.slider("Mini Mental State Examination Score (0–30)", min_value=0, max_value=30, value=24)
        func = st.slider("Functional Assessment Score (0–10)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
        adl = st.slider("Active Daily Living Score (0–10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

    with c2:
        mem = st.selectbox("Memory Complaints", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        beh = st.selectbox("Behavioral Problems", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Run prediction")

if submitted:
    # Build a single-row dataframe with raw (unstandardized) user inputs
    raw = pd.DataFrame([{
        "ADL": float(adl),
        "MMSE": float(mmse),
        "FunctionalAssessment": float(func),
        "MemoryComplaints": int(mem),
        "BehavioralProblems": int(beh),
    }])

    # Standardize only continuous columns
    X = raw.copy()
    X[cols] = scaler.transform(X[cols])
    
    # Predictions
    
    # Probability Prediction
    prob = model.predict_proba(X)[0]
    
    # Class and Conformal Prediction
    y_pred, y_pred_set = mapie.predict_set(X)
    pred = int(y_pred[0])

    # Formatting output
    pclass = "Alzheimer's" if pred == 1 else "No Alzheimer's"

    # Conformal prediction set
    pred_set = [
        "Alzheimer's" if i == 1 else "No Alzheimer's"
        for i, included in enumerate(y_pred_set[0])
        if included
    ]

    # Results
    st.markdown(f"### Predicted Class:\n##### {pclass}")
    
    # send to human for review under certain conditions
    if bcc.predict(X)[0] == 1:
      st.markdown("###### It is advised that this patient case be sent for review.")
    
    st.divider()

    st.markdown(f"### Conformal Prediction Set (with a confidence level of 90%):\n ##### {pred_set}")
    st.markdown("##### Interpretation: With a 90% confidence, we can say that the true label lies within the given prediction set.")
    
    set_size = y_pred_set.sum(axis=1)
    if set_size == 1:
        st.markdown(f"##### The model is 90% confident that the patient has {pclass}.")
    else:
        st.markdown("###### The model is uncertain about the patient's diagnosis and it is advised that this case be sent for review.")
        
    st.divider()
    
    # Probability Prediction
    st.markdown(f"### Predicted Probability:\n##### {prob[1]:.2f}")
    st.markdown(f"##### Interpretation: The patient has a predicted probability of having Alzheimer's of {prob[1]:.2f}.")
    st.divider()

    # SHAP
    st.markdown("### SHapley Additive exPlanations (SHAP)")
    st.markdown("##### SHAP is a tool used for interpreting black box models.")


    shap_values = explainer(X)

    st.write()
    exp = shap.Explanation(
      values=shap_values.values[0, :, 1],  
      base_values=shap_values.base_values[0, 1], 
      data=X.iloc[0],
      feature_names=X.columns
    )

    shap.plots.waterfall(exp, max_display=5)
    plt.xlabel("Change in predicted probability of Alzheimers (SHAP values)")
    st.pyplot(plt.gcf(), clear_figure=True, width="stretch")

    st.markdown(f"##### The standardized inputs are on the left for each feature. The base (expected) probability is {shap_values.base_values[0, 1]:.3f}.")
    st.markdown(
      f"""#### Interpretation for this Patient:
      - Active Daily Living Assessment: 
      Affects the predicted probability of Alzheimer's by {shap_values.values[0, :, 1][0]:.2f} relative to the model’s baseline probability.

      - Mini Mental State Examination Score:
      Affects the predicted probability of Alzheimer's by {shap_values.values[0, :, 1][1]:.2f} relative to the model’s baseline probability.
      
      - Functional Assessment Score:
      Affects the predicted probability of Alzheimer's by {shap_values.values[0, :, 1][2]:.2f} relative to the model’s baseline probability.
      
      - Memory Complaints:
      Affects the predicted probability of Alzheimer's by {shap_values.values[0, :, 1][3]:.2f} relative to the model’s baseline probability.
      
      - Behavioral Problems:
      Affects the predicted probability of Alzheimer's by {shap_values.values[0, :, 1][4]:.2f} relative to the model’s baseline probability.
      """)
