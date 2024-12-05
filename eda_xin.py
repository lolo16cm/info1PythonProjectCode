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

