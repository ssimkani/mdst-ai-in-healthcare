import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import joblib

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)
from mapie.utils import train_conformalize_test_split
from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import classification_coverage_score

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("Model Visualizations")
st.markdown("### A Random Forest Classifier model trained on the Alzheimer's dataset")

# Preview of data used for model
st.markdown("### Data Preview")
st.caption("The dataset used for model training, calibration, and testing.")
df = pd.read_csv('./data/clean/alzheimers.csv')
st.dataframe(df.head(50), width="stretch")

# load models with pickle
with open("./models/cal_rf.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("./models/mapie_rf.pkl", "rb") as f:
    mapie = pickle.load(f)

# load all data partititions
X_train = pd.read_csv("./data/partitions/X_train.csv")
X_cal = pd.read_csv("./data/partitions/X_cal.csv")
X_test = pd.read_csv("./data/partitions/X_test.csv")
y_train = pd.read_csv("./data/partitions/y_train.csv")
y_cal = pd.read_csv("./data/partitions/y_cal.csv")
y_test = pd.read_csv("./data/partitions/y_test.csv")

# Create series from dataframes
y_train = y_train["Diagnosis"]
y_cal = y_cal["Diagnosis"]
y_test = y_test["Diagnosis"]


# Get predictions and probabilities

st.markdown("### Conformal Predictions using MAPIE and a confidence level of 90%")
y_pred, y_pred_set = mapie.predict_set(X_test)
coverage_score = classification_coverage_score(y_test, y_pred_set)
st.markdown(
    "#### For a confidence level of 0.90, "
    "the target coverage is 0.90, "
    f"and the effective coverage is {coverage_score[0]:.3f}."
)

# Set sizes
set_sizes = y_pred_set.sum(axis=1)
st.markdown(f"#### Average set size: {set_sizes.mean():.2f}")
st.markdown(f"#### Singleton rate: {(set_sizes == 1).mean():.2f}")
st.markdown(f"#### Ambiguous rate: {(set_sizes > 1).mean():.2f}")
st.markdown(f"#### Empty rate: {(set_sizes == 0).mean():.2f}")

st.divider()

# plot conformal predictions and probabilities
st.markdown("#### Probability vs Conformal Predictions")
st.markdown("##### A very small amount of sets have no elements and the probabilities vary and they range from 0.2-0.8 roughly. The singleton sets have high and low probabilities associated with them which makes sense. Low probabilities are likely {0} and high probabilities are most likely {1}.")

probs = model.predict_proba(X_test)[:, 1]
plt.scatter(probs, set_sizes, alpha=0.3)
plt.xlabel("P(y=1 | x)")
plt.ylabel("Set Size")
plt.yticks([1, 2])
plt.title("Probability vs Conformal Predictions")
st.pyplot(plt.gcf(), clear_figure=True, width="stretch")
st.divider()

# Confusion Matrices
st.markdown("### Confusion Matrices")
st.markdown("#### Confusion Matrix with a threshold of 0.7 predicted probability")

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.7).astype(int)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true")
plt.title("Confusion Matrix (threshold=0.7)")
st.pyplot(plt.gcf(), clear_figure=True, width="stretch")

st.divider()

st.markdown("#### Normal Confusion Matrix")
y_pred = model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true")
plt.title("Confusion Matrix")
st.pyplot(plt.gcf(), clear_figure=True, width="stretch")

st.markdown("#### For roughly 10% of patients, our model predicted that they did have Alzheimer's when the didn't and for roughly 1% of patients, our model predicted that they didn't have Alzhiemer's when they did.")
st.divider()
st.markdown("#### From these confusion matrices, we can see that the model is very accurate when predicting someone doesn't have Alzheimer's than predicting that someone does. The model handles false positives better than it handles false negatives.")
st.divider()

# SHAP Explanations

# load shap
shap_values = joblib.load("./data/shap/shap_values.joblib")

st.markdown("### SHAP Explanations")
st.markdown("#### Summary Plot")

shap.summary_plot(
    shap_values[..., 1],
    X_test,
    feature_names=X_test.columns,
    show=False,
)
plt.xlabel("Change in predicted probability of Alzheimers (SHAP values)")
st.pyplot(plt.gcf(), clear_figure=True, width="stretch")

st.divider()


st.markdown("#### Violin plot")

shap.plots.violin(shap_values[..., 1], plot_type="layered_violin", show=False)
plt.xlabel("Change in predicted probability of Alzheimers (SHAP values)")
st.pyplot(plt.gcf(), clear_figure=True, width="stretch")

st.divider()

st.markdown("### Interpreting Model Plots")

st.markdown(
    """
    <div style="font-size:18px; line-height:1.7;">

    <h4>Functional Assessment</h4>
    Higher functional assessment scores result in a  
    <b>lower predicted probability</b>, indicating reduced predicted risk of Alzheimer’s.
    Conversely, lower assessment scores contribute to a 
    <b>higher model score</b>, increasing predicted risk.

    <br><br>

    <h4>Activities of Daily Living (ADL) Assessment</h4>
    This feature shows a pattern similar to the Functional Assessment.
    Lower ADL scores are associated with higher predicted risk, while
    higher ADL scores contribute negatively to the model’s Alzheimer’s score.

    <br><br>

    <h4>Mini-Mental State Examination (MMSE)</h4>
    Both high and low MMSE values contribute to the model’s predictions,
    though the magnitude of contribution is relatively small.
    Lower MMSE scores tend to increase the model’s predicted risk.
    As shown in the SHAP summary plot, higher MMSE scores generally have
    a <b>negative contribution</b>, pushing predictions away from Alzheimer’s.

    <br><br>

    <h4>Memory Complaints and Behavioral Problems</h4>
    The absence of memory complaints or behavioral problems contributes
    slightly <b>negatively</b> to the model’s score.
    In contrast, the presence of these issues has a
    <b>large positive influence</b> on the predicted risk of Alzheimer’s.
    Notably, the dataset contains more individuals without these issues
    than with them, which is reflected in the distribution of SHAP values.

    </div>
    """,
    unsafe_allow_html=True
)

