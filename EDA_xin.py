import pandas as pd

# load data
basic_survey_df = pd.read_csv('BasicSurvey.csv', low_memory=False)
fitbit_activity_df = pd.read_csv('FitbitActivity.csv')
fitbit_sleep_df = pd.read_csv('FitbitSleep.csv')

# BasicSurvey
basic_survey_cleaned = basic_survey_df[['egoid', 'gender_1']].drop_duplicates()

# FitbitActivity
activity_columns = ['egoid', 'datadate', 'meanrate', 'steps', 'sedentaryminutes',
                    'lightlyactiveminutes', 'fairlyactiveminutes', 'veryactiveminutes',
                    'lowrangecal', 'fatburncal', 'cardiocal']
fitbit_activity_cleaned = fitbit_activity_df[activity_columns]

# implent NUll values
fitbit_activity_cleaned['lightlyactiveminutes'] = fitbit_activity_cleaned.groupby('egoid')['lightlyactiveminutes'] \
    .transform(lambda x: x.fillna(x.mean()))
fitbit_activity_cleaned['sedentaryminutes'] = fitbit_activity_cleaned.groupby('egoid')['sedentaryminutes'] \
    .transform(lambda x: x.fillna(x.mean()))
fitbit_activity_cleaned['fairlyactiveminutes'] = fitbit_activity_cleaned['fairlyactiveminutes'].fillna(0)
fitbit_activity_cleaned['veryactiveminutes'] = fitbit_activity_cleaned['veryactiveminutes'].fillna(0)

# calculate totalCalorie
fitbit_activity_cleaned['totalCalorie'] = fitbit_activity_cleaned[['lowrangecal', 'fatburncal', 'cardiocal']].sum(axis=1)

# delete duplicated
fitbit_activity_cleaned = fitbit_activity_cleaned.sort_values('datadate').drop_duplicates(['egoid', 'datadate'])

# FitbitSleep
sleep_columns = ['egoid', 'dataDate', 'bedtimedur', 'minstofallasleep', 'minsasleep', 'minsawake', 'Efficiency']
fitbit_sleep_cleaned = fitbit_sleep_df[sleep_columns].rename(columns={'dataDate': 'datadate'})

# combine
merged_data = fitbit_activity_cleaned.merge(fitbit_sleep_cleaned, on=['egoid', 'datadate'], how='inner')
final_data = basic_survey_cleaned.merge(merged_data, on='egoid', how='inner')

# save as a new
final_data.to_csv('ProcessedData.csv', index=False)

print("saved as 'ProcessedData.csv'")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the processed data
data = pd.read_csv("ProcessedData.csv")

# Step 1: Data Overview
print("Data Info:")
print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

# Step 2: Data Visualization

# (a) Distribution of key variables
sns.histplot(data['Efficiency'], kde=True)
plt.title('Efficiency Distribution')
plt.show()

sns.histplot(data['minsasleep'], kde=True)
plt.title('Minutes Asleep Distribution')
plt.show()

sns.histplot(data['totalCalorie'], kde=True)
plt.title('Mean Heart Rate Distribution')
plt.show()

from sklearn.preprocessing import StandardScaler

# normalization
numeric_columns = data.select_dtypes(include=['number']).columns
scaler = StandardScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)

# compute cov matrix
normalized_cov_matrix = normalized_data.cov()

# plot
plt.figure(figsize=(8, 6))
sns.heatmap(normalized_cov_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=numeric_columns, yticklabels=numeric_columns, annot_kws={"size": 8})
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.title('Normalized Covariance Matrix', fontsize=15)
plt.show()

# compute the relation of Efficiency with other attributes
efficiency_corr = normalized_data.corr()['Efficiency']

# sort/order
efficiency_corr_sorted = efficiency_corr.abs().sort_values(ascending=False)

# show top five attributes
top_correlated_attributes = efficiency_corr_sorted[1:6]
print("Top attributes affecting Efficiency based on correlation:")
print(top_correlated_attributes)

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# normalization
scaler = MinMaxScaler()
data[['totalCalorie', 'Efficiency']] = scaler.fit_transform(data[['totalCalorie', 'Efficiency']])

# pick one sample
example_ids = data['egoid'].unique()[35:36]
filtered_data = data[data['egoid'].isin(example_ids)]

# plot scatter
for egoid in example_ids:
    egoid_data = filtered_data[filtered_data['egoid'] == egoid]

    plt.figure(figsize=(5, 4))
    plt.scatter(egoid_data['totalCalorie'], egoid_data['Efficiency'], alpha=0.6, color='blue')

    plt.title(f'Scatter Plot of TotalCalorie vs Efficiency for EgoID {egoid}')
    plt.xlabel('TotalCalorie (Normalized)')
    plt.ylabel('Efficiency (Normalized)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import numpy as np

data = pd.read_csv("ProcessedData.csv")

# delete null rows
data_cleaned = data.dropna()

# binning some attributes
columns_to_bin = [
    "meanrate", "steps", "sedentaryminutes",
    "lowrangecal", "fatburncal", "cardiocal", "totalCalorie",
    "bedtimedur", "minstofallasleep", "minsasleep", "minsawake", "Efficiency"
]

# binning
for col in columns_to_bin:
    # define
    bins = np.linspace(data_cleaned[col].min(), data_cleaned[col].max(), 10)
    labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    data_cleaned[f'{col}_binned'] = pd.cut(data_cleaned[col], bins=bins, labels=labels, include_lowest=True)
    data_cleaned[col] = data_cleaned[f'{col}_binned'].astype(float)

data_cleaned = data_cleaned.drop(columns=[f'{col}_binned' for col in columns_to_bin])

# save
data_cleaned.to_csv("cleanedData.csv", index=False)

print("saved 'cleanedData.csv'")
