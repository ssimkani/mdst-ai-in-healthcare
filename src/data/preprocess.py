import pandas as pd

df = pd.read_csv('data/raw/alzheimers.csv')

# Drop doctorInCharge, PatientID columns
df.drop(columns=['DoctorInCharge', 'PatientID'], inplace=True)

df.drop_duplicates(inplace=True)

df.to_csv('data/interim/alzheimers.csv', index=False)
