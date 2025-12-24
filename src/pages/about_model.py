import streamlit as st
from PIL import Image

st.set_page_config(page_title="About Model", layout="wide")

st.title("About Model")

# import image
img = Image.open("./src/images/rf_classifier.jpg")
st.image(img, caption="[Architecture of a Random Forest Classifier](https://medium.com/analytics-vidhya/random-forest-classifier-and-its-hyperparameters-8467bec755f6)", width="stretch")

st.markdown("""
Random Forest is a popular supervised machine learning algorithm used for both
classification and regression tasks (in this case its classification). It works by combining many decision trees
to produce more accurate and stable predictions.
""")

st.header("What Is Random Forest?")
st.markdown("""
A Random Forest is an ensemble model made up of many decision trees.
Each tree is trained on a random subset of the data and features.
The final prediction is made by:
- Majority voting (classification)
- Averaging predictions (regression)
""")

st.divider()

st.header("How It Works")
st.markdown("""
1. Random samples of the training data are taken.
2. Each decision tree is trained on a random subset of features.
3. Trees make their predictions independent of each other.
4. Predictions are combined for the final output.

Randomness helps reduce overfitting and improves generalization.
""")

st.divider()

st.header("Random Forest vs Decision Trees")
st.markdown("""
- Decision Trees learn a single set of rules and can easily overfit.
- **Random Forests** combine many trees trained with randomness, making them
  more robust and accurate.
""")

st.divider()

st.header("Advantages")
st.markdown("""
- Works well out of the box
- Handles classification and regression
- Reduces overfitting
- Easy to understand
""")

st.divider()

st.header("Disadvantages")
st.markdown("""
- Slower predictions with many trees  
- Uses more memory
- Not ideal for explaining relationships between variables  
""")

st.divider()

st.caption("""
Source:
Donges, N. (Updated by Whitfield, B.). *Random Forest: A Complete Guide for Machine Learning*.  
https://builtin.com/data-science/random-forest-algorithm
""")