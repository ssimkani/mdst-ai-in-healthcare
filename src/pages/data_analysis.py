import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title='Data Analysis', layout='wide')
st.title('Data Analysis')

# For EDA
df = pd.read_csv('./data/interim/alzheimers.csv')

# Preview
with st.expander('Preview Data', expanded=False):
    st.dataframe(df.head(50), width="stretch")

def show_fig(fig=None):
    if fig is None:
        fig = plt.gcf()
    st.pyplot(fig, clear_figure=True, width="stretch")

# Data Viz
st.markdown("## Seena")

# correlation matrix with only cognitive assessments
matrix = df[['MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL']].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(matrix, dtype=bool))

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    matrix,
    mask=mask,
    cmap="coolwarm",
    center=0,
    annot=False,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax
)

# Description
st.markdown('##### From our correlation matrix, we found that all cognitive assessments are correlated with the Diagnosis with a threshold of 0.2. We started our analysis here.')

ax.set_title("Correlation Matrix of Cognitive Assessments", fontsize=14)
show_fig(fig)
st.divider()

st.markdown('## Cognitive Assessment Analysis')

st.markdown('##### MMSE is uniformly distributed.')
# MMSE distribution (pretty uniform with a dip in the middle)
plt.hist(df['MMSE'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Mini Mental State Examination')
show_fig()

# ECDF
sns.ecdfplot(df["MMSE"])
show_fig()
st.divider()

# MMSE vs Diagnosis boxplot
st.markdown("##### Patients diagnosed with Alzheimer's have lower median MMSE scores by about 6 points than those who weren't diagnosed.")

sns.boxplot(x='Diagnosis', y='MMSE', data=df)
plt.xlabel('MMSE')
plt.ylabel('Diagnosis')
plt.title('MMSE vs Diagnosis')
show_fig()

# violin plot
sns.violinplot(x='Diagnosis', y='MMSE', data=df, inner='quartile')
plt.xlabel('MMSE')
plt.ylabel('Diagnosis')
plt.title('MMSE vs Diagnosis')
show_fig()

st.dataframe(df.groupby('Diagnosis')['MMSE'].median())

st.divider()

# FunctionalAssessment Distribution (uniform)
st.markdown('##### Uniform distribution of FunctionalAssessment.')

plt.hist(df['FunctionalAssessment'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('FunctionalAssessment')
show_fig()

sns.ecdfplot(df["FunctionalAssessment"])
show_fig()
st.divider()

# FunctionalAssessment vs Diagnosis boxplot
st.markdown("##### 50% of patients diagnosed with Alzheimer's scored less than 4 on the FunctionalAssessment scale while 50% of patients who were not diagnosed with Alzheimer's scored above 6")

sns.boxplot(x='Diagnosis', y='FunctionalAssessment', data=df)
plt.xlabel('FunctionalAssessment')
plt.ylabel('Diagnosis')
plt.title('FunctionalAssessment vs Diagnosis')
show_fig()


sns.violinplot(data=df, x="Diagnosis", y="FunctionalAssessment", inner="quartile")
show_fig()

st.dataframe(df.groupby('Diagnosis')['FunctionalAssessment'].median())
st.divider()

# MemoryComplaints countplot
st.markdown('##### More data points have no memory complaints than those with memory complaints\n0 = no memory complaints\n1 = memory complaints')

sns.countplot(x='MemoryComplaints', data=df)
plt.xlabel('MemoryComplaints')
plt.ylabel('Proportion')
plt.title('MemoryComplaints')
show_fig()

st.markdown('##### Those in the no diagnosis group have a less memory complaints than those in the diagnosis group.')

# crosstab heatmap with Diagnosis
cross = pd.crosstab(df["Diagnosis"], df["MemoryComplaints"], normalize="index")
sns.heatmap(cross, annot=True, fmt=".2f", cmap="Blues")
show_fig()
st.divider()

# BehavioralProblems countplot
st.markdown('##### More data points have no behavioral complaints than those with behavioral complaints')

sns.countplot(x='BehavioralProblems', data=df)
plt.xlabel('BehavioralProblems')
plt.ylabel('Proportion')
plt.title('BehavioralProblems')
show_fig()

# crosstab heatmap
st.markdown('##### Higher prevalence of behavioral problems in the diagnosis group than in the no diagnosis group')

cross = pd.crosstab(df["Diagnosis"], df["BehavioralProblems"], normalize="index")
sns.heatmap(cross, annot=True, fmt=".2f", cmap="Blues")
show_fig()
st.divider()

# Activities of Daily Living Score distribution
st.markdown('##### Uniform distribution.')
plt.hist(df['ADL'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('ADL')
show_fig()

# ECDF
sns.ecdfplot(df["ADL"])
show_fig()
st.divider()

# ADL vs Diagnosis boxplot
st.markdown('##### The median of the ADL score is higher in the no diagnosis group.')

sns.boxplot(x='Diagnosis', y='ADL', data=df)
plt.xlabel('ADL')
plt.ylabel('Diagnosis')
plt.title('ADL vs Diagnosis')
show_fig()

st.dataframe(df.groupby('Diagnosis')['ADL'].median())

# violin plot
st.markdown("##### 50% of patients diagnosed with Alzheimer's scored less than 4 on the ADL scale while 50% of patients who were not diagnosed with Alzheimer's scored above 6.")

sns.violinplot(x='Diagnosis', y='ADL', data=df, inner='quartile')
plt.xlabel('ADL')
plt.ylabel('Diagnosis')
plt.title('ADL vs Diagnosis')
show_fig()


st.markdown("##### All cognitive assessments seem relevant to determining Alzheimer's diagnosis.")
st.divider()

st.markdown('## Demographic Analysis')


# Age distribution
st.markdown('##### Higher prevalence of patients older than 85.')

plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Age')
show_fig()

sns.ecdfplot(df["Age"])
show_fig()

# Violin Plot
st.markdown("##### Very similar violin plots with a very similar median of 75 years. While alzheimer's increases with age, the data contains mostly older patients making this metric less relevant.")

sns.violinplot(x='Diagnosis', y='Age', data=df, inner='quartile')
plt.xlabel('Age')
plt.ylabel('Diagnosis')
plt.title('Age vs Diagnosis')
show_fig()


st.divider()

# Gender distribution (uniform)
st.markdown('##### Uniform distribution of Gender.')
sns.countplot(x='Gender', data=df)
plt.title('Gender')
show_fig()

# countplot with Diagnosis
sns.countplot(x='Gender', hue='Diagnosis', data=df)
plt.xlabel('Gender')
plt.ylabel('Proportion')
plt.title('Gender vs Diagnosis')
show_fig()

st.markdown('##### Very similar to each other.')
st.divider()
# Ethnicity distribution
st.markdown('##### More Caucasian data points the others.')

sns.countplot(x='Ethnicity', data=df)
plt.title('Ethnicity')
show_fig()

st.markdown('* 0: Caucasian  \n* 1: African American  \n* 2: Asian  \n* 3: Other')


# countplot with Diagnosis
st.markdown("##### There seems to be overall less patients in each race in the diagnosis group, but there is no information about certain races having higher diagnosis of alzheimer's due to the left leaning distribution.")
sns.countplot(x='Ethnicity', hue='Diagnosis', data=df)
plt.title('Ethnicity vs Diagnosis')
show_fig()


st.markdown('##### Not much information about demographics in predicting diagnosis')
st.divider()

st.markdown('## Lifestyle Factors')

# BMI Distribution
st.markdown('##### Uniform distrubition.')
plt.hist(df['BMI'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('BMI')
show_fig()

# BMI violin plot
sns.violinplot(x='Diagnosis', y='BMI', data=df, inner='quartile')
plt.title('BMI vs Diagnosis')
show_fig()

st.divider()

# Smoking Distribution
sns.countplot(x='Smoking', data=df)
plt.title('Smoking')
show_fig()

st.markdown("##### More data points with patients who don't smoke.")

# crosstab heatmap
cross = pd.crosstab(df["Diagnosis"], df["Smoking"], normalize="index")
sns.heatmap(cross, annot=True, fmt=".2f", cmap="Blues")
show_fig()


# --- cell ---
# AlcoholConsumption Distribution
st.markdown('##### Uniform distribution for alcohol consumption.')
plt.hist(df['AlcoholConsumption'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('AlcoholConsumption')
show_fig()

# AlcoholConsumption violin plot

sns.violinplot(x='Diagnosis', y='AlcoholConsumption', data=df, inner='quartile')
plt.title('AlcoholConsumption vs Diagnosis')
show_fig()

st.markdown(" ##### No useful information from smoking, BMI, or alcohol consumption.")

st.divider()

# Distribution of Diagnosis
st.markdown("##### More data for patients who were not diagnosed with Alzheimer's than those that were. This might result in a model that can predict an absence of Alzheimer's better than it can predict the presence of it.")
sns.countplot(x='Diagnosis', data=df)
show_fig()

# Will's EDA
st.markdown('## Will')

# --- cell ---
# Engineer "AgeCat" feature that turns a numerical age into a categorical one
def assign_age(x):
    if x >= 60 and x < 70:
        return "Old"
    elif x >= 70 and x < 80:
        return "Older"
    else:
        return "Oldest"
    
df['AgeCat'] = df['Age'].apply(assign_age)

# --- cell ---
# Sort patients by age group and check prevalence of Alzheimer's
st.dataframe(df.groupby('AgeCat')['Diagnosis'].value_counts(normalize=True))

st.markdown("##### Here, we can see that diagnosis frequency across age groups is relatively similar. We got a decent idea of this beforehand, but this gives a more concrete image of how age alone does not seem to reliably distinguish those with Alzheimer's and those without.")


st.divider()

# Find proportion of patients of each sex
print(f"Male: {float((df['Gender'] == 0).sum())/len(df)*100}%")
print(f"Female: {float((df['Gender'] == 1).sum())/len(df)*100}%")

# Group by gender and Alzheimer's status
st.dataframe(df.groupby(['Gender', 'Diagnosis']).size().unstack())
# diagnosis_percentages = diagnosis_data.div(diagnosis_data.sum(axis = 1), axis = 0) * 100

# Print out the percentages for each gender
# for g in diagnosis_percentages.index:
#     print(f"Gender {g} Alzheimer's Proportion: {diagnosis_percentages.loc[g, 1]}%")

# Print out the overall population Alzheimer's proportion
# print(f"Overall Alzheimer's Proportion: {diagnosis_percentages[1].mean()}%")

st.markdown("##### The dataset is nearly perfectly split 50/50, though we can note that the males (Gender 0) have a slightly higher prevalence of Alzheimer's than females (Gender 1), though not by such a degree that we would want to use gender as a feature in our model.")

st.divider()

# Determine the prevalence of family Alzheimer's history
df['FamilyHistoryAlzheimers'].value_counts(normalize=True)

# Group by family history and Alzheimer's status
st.dataframe(df.groupby(['FamilyHistoryAlzheimers', 'Diagnosis']).size().unstack())
# diagnosis_percentages = diagnosis_data.div(diagnosis_data.sum(axis = 1), axis = 0) * 100

# Print out the percentages for each family history group
# for f in diagnosis_percentages.index:
#     print(f"Family History {f} Alzheimer's Proportion: {diagnosis_percentages.loc[f, 1]:.2f}%")

# Print out the overall population family history relevance
# print(f"Overall Population Family History Relevance: {df['FamilyHistoryAlzheimers'].sum()/len(df) * 100:.2f}%")

st.markdown("##### Really interesting thing to note here: There was a higher prevalence of Alzheimer's in the group who did NOT have a family history by a somewhat sizeable margin. Not sure if this is just me, but this acts against what I would expect the result to be. Maybe we can highlight this in our slides and show how our expectations may not hold up when we inspect what the data actually show.")

st.divider()

# Determine the prevalence of forgetfulness
df['Forgetfulness'].value_counts(normalize=True)

# Group by forgetfulness and Alzheimer's status
st.dataframe(df.groupby(['Forgetfulness', 'Diagnosis']).size().unstack())
# diagnosis_percentages = diagnosis_data.div(diagnosis_data.sum(axis = 1), axis = 0) * 100

# Print out the percentages for each forgetfulness group
# for f in diagnosis_percentages.index:
#     print(f"Forgetfulness {f} Alzheimer's Proportion: {diagnosis_percentages.loc[f, 1]:.2f}%")

# Print out the overall population forgetfulness relevance
# print(f"Overall Population Forgetfulness Relevance: {df['Forgetfulness'].sum()/len(df) * 100:.2f}%")

st.markdown("##### Here, it looks like forgetfulness is not a useful predictor of Alzheimer's whatsoever; the proportions are almost the exact same.")

st.divider()

# Determine the prevalence of depression
df['Depression'].value_counts(normalize=True)

# Group by depression and Alzheimer's status
st.dataframe(df.groupby(['Depression', 'Diagnosis']).size().unstack())
# diagnosis_percentages = diagnosis_data.div(diagnosis_data.sum(axis = 1), axis = 0) * 100

# Print out the percentages for each depression group
# for f in diagnosis_percentages.index:
#     print(f"Depression {f} Alzheimer's Proportion: {diagnosis_percentages.loc[f, 1]:.2f}%")

# Print out the overall population depression relevance
# print(f"Overall Population Depression Relevance: {df['Depression'].sum()/len(df) * 100:.2f}%")

st.markdown('##### Again, we find another variable that does not play whatsoever.')

st.divider()

# Determine the prevalence of personality changes
df['PersonalityChanges'].value_counts(normalize=True)

# Group by personality changes and Alzheimer's status
st.dataframe(df.groupby(['PersonalityChanges', 'Diagnosis']).size().unstack())
# diagnosis_percentages = diagnosis_data.div(diagnosis_data.sum(axis = 1), axis = 0) * 100

# Print out the percentages for each personality change group
# for f in diagnosis_percentages.index:
#     print(f"Personality Changes {f} Alzheimer's Proportion: {diagnosis_percentages.loc[f, 1]:.2f}%")

# Print out the overall population personality change relevance
# print(f"Overall Population Personality Change Relevance: {df['PersonalityChanges'].sum()/len(df) * 100:.2f}%")

st.markdown("##### Interesting takeaway: Alzheimer's is more prevalent in patients who did NOT experience personality changes than those who did. This goes against expectations and may be another interesting finding to highlight.")

st.divider()

st.markdown('## Principal Component Analysis')

# Create input and target dataframes/arrays
# We drop AgeCat here because it is derived from Age, which is already included
X_data = df.drop(columns = ['Diagnosis', 'AgeCat'])
y_data = df['Diagnosis']

# --- cell ---
# Encode categorical variables
X_data = pd.get_dummies(X_data, drop_first = True)

# Standardize the features
pca_scaler = StandardScaler()
X_scaled = pca_scaler.fit_transform(X_data)

# Run the PCA algorithm
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Determine how many PCs we need based on variance explanation threshold
# We do this by plotting a cumulative density plot of explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("CEV Plot for PCs on Alzheimer's Data")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
show_fig()

st.markdown('##### It appears that the first 25 or so principal components would be sufficient to explain ~80% of the variation in the data.\nThis would be useful if we were focused particularly on reducing the dimensionality of the data, though I doubt that we\nwill put such an emphasis on model efficiency that we would do such a thing.\n\nHowever, we can touch on the fact that dimensionality reduction techniques, which may be appealing to executives at large\nhospitals or healthcare companies, lose accuracy by design. This focus on efficiency can compromise best practice, and\nin the case of a very complex and hard-to-define disease (in terms of causes), obscuring patient data further can\nbe risky.')

# --- cell ---
# Plot the first two Principal Components, stratified by diagnosis status
plt.figure(figsize=(6, 5))
for label in y_data.unique():
    plt.scatter(
        X_pca[y_data == label, 0],
        X_pca[y_data == label, 1],
        label=label,
        alpha=0.6
    )

# Label the plot and print it
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title("PCA Projection Colored by Diagnosis")
show_fig()

st.markdown("##### We can see a slight difference in the clustering of orange and blue (diagnosis vs. no diagnosis) data points, meaning that while these two principal components cannot fully distinguish these two groups (or really get all that close, even), plotting the points on these two axes does do something to create a distinction between patients with and without Alzheimer's.")

# Define the loadings for each feature on each Principal Component
loadings = pd.DataFrame(
    pca.components_.T,
    index=X_data.columns,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)

# Show the top-contributing features for PC1
st.dataframe(loadings["PC1"].sort_values(ascending=False).head(10), width="stretch")

# --- cell ---
# Show the top-contributing features for PC2
st.dataframe(loadings["PC2"].sort_values(ascending=False).head(10), width="stretch")

st.markdown('##### Here\'s what we can take away from this:\nA combination of the following factors:\n* Smoking (0.38)\n* Cardiovascular Disease (0.34)\n* Memory Complaints (0.27)\n* Difficulty Completing Tasks (0.26)\n\nis our best (although relatively weak) predictor of Alzheimer\'s diagnosis.\n\nA combination of these factors:\n* BMI (0.33)\n* CholesterolHDL (0.24)\n* Smoking (0.20)\n* Behavioral Problems (0.20)\n\nis our second-best predictor of Alzheimer\'s disease being diagnosed in our data.\n\nWhat this means is that these factors, which are a mix of lifestyle choices, comorbid disease,\nand what are expected symptoms of Alzheimer\'s Disease, all co-correlate with a diagnosis to some degree\nWHEN TAKEN IN TANDEM. We need to consider their co-influence.\n\nWe cannot isolate a single main predictor that far-and-away explains Alzheimer\'s risk. No\nobvious line can be drawn to say, "these people will have the disease and these people won\'t,"\nwhich is to be expected from a multi-factor ailment like this.\n\nThese findings can inform our decisions on what factors we want to feed into our model and what\nfactors we want to look into further in our EDA process.')

st.markdown("##### For our model, we decided to use all cognitive assessments as they were the most promising features in our visualizations. Here is the updated dataset for use in our model:")

# create dataframe with just cognitive tests and target
model_df = df[['ADL', 'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'Diagnosis']]

# Standardize ADL, MMSE, FunctionalAssessment
scale = preprocessing.StandardScaler()
cols = ['ADL', 'MMSE', 'FunctionalAssessment']
model_df[cols] = scale.fit_transform(model_df[cols])

# final dataset used for model
st.dataframe(model_df.head(50), width="stretch")
