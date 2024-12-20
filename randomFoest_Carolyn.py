import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Read the CSV file
df = pd.read_csv('cleanedData.csv')

# Ignore gender_1, 'egoid' and 'datadate' features
merged_df = df.drop(['gender_1','egoid', 'datadate'], axis=1)

# Set threshold to 0.9
threshold = 0.9

# Binarize sleep quality based on 'Efficiency'
# 1= good sleep, 0 is poor sleep
merged_df['sleep_quality'] = np.where(merged_df['Efficiency'] >= threshold, 1, 0)

# Select features (all except 'Efficiency' and 'sleep_quality') and target variable
features = merged_df.columns.drop(['sleep_quality', 'Efficiency'])
X = merged_df[features]
y = merged_df['sleep_quality']

# Check for missing values
print("Missing values in features:\n", X.isnull().sum())



X = merged_df[features]
y = merged_df['sleep_quality']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

# Get feature importances
importances = rf_model.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# visualization the feature importance
# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm'
)
plt.title('RandomForestClassifier Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()






# Default Interactive Prediction: Test data
test_data_1 = {
    'meanrate': 65.5,
    'steps': 7500,
    'sedentaryminutes': 300,
    'lightlyactiveminutes': 120,
    'fairlyactiveminutes': 45,
    'veryactiveminutes': 30,
    'lowrangecal': 1500,
    'fatburncal': 400,
    'cardiocal': 300,
    'totalCalorie': 2500,
    'bedtimedur': 8.0,
    'minstofallasleep': 15,
    'minsasleep': 450,
    'minsawake': 30,
}

test_data_2 = {
    'meanrate': 60,
    'steps': 0,
    'sedentaryminutes': 1440,
    'lightlyactiveminutes': 0,
    'fairlyactiveminutes': 0,
    'veryactiveminutes': 0,
    'lowrangecal': 1000,
    'fatburncal': 0,
    'cardiocal': 0,
    'totalCalorie': 1000,
    'bedtimedur': 5.0,
    'minstofallasleep': 30,
    'minsasleep': 300,
    'minsawake': 60,
}

# Convert to DataFrame
input_df_1 = pd.DataFrame([test_data_1])

# Run prediction
prediction_1 = rf_model.predict(input_df_1)
print(f"Test Case 1 Prediction: {'Good' if prediction_1[0] == 1 else 'Poor'}")

import pandas as pd

# Iteractive Interactive Prediction
#user can input the data into the prediction to get the prediction result

# List of factors, declare heere for easy readable
factors = [
    'meanrate', 'steps', 'sedentaryminutes', 'lightlyactiveminutes',
    'fairlyactiveminutes', 'veryactiveminutes', 'lowrangecal', 'fatburncal',
    'cardiocal', 'totalCalorie', 'bedtimedur', 'minstofallasleep',
    'minsasleep', 'minsawake'
]

# Initialize an empty dictionary to store user input
user_input_data = {}

# Collect input from the user and store in the dictionary
print("Enter the values for your test case:")
for feature in factors:
    while True:
        try:
            value = float(input(f"Enter value for {feature}: "))
            user_input_data[feature] = value
            break
        except ValueError:
            print(f"Invalid input for {feature}. Please enter a numeric value.")

# Display the collected dictionary
print("\nCollected Data:")
print(user_input_data)

# Optionally convert the dictionary to a DataFrame for prediction
user_input_df = pd.DataFrame([user_input_data])

# Assuming `rf_model` is a trained model, run the prediction
try:
    prediction = rf_model.predict(user_input_df)
    print(f"\nPredicted Sleep Quality: {'Good' if prediction[0] == 1 else 'Poor'}")
except Exception as e:
    print(f"Error during prediction: {e}")



