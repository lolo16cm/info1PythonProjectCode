import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Read the CSV file
df = pd.read_csv('cleanedData.csv')

# Ignoring 'gender_1', 'egoid', and 'datadate' features
merged_df = df.drop(['gender_1', 'egoid', 'datadate'], axis=1)

# Setting the threshold to 0.9
threshold = 0.9

# Binarize sleep quality based on 'Efficiency'
# 1 = good sleep, 0 = poor sleep
merged_df['sleep_quality'] = np.where(merged_df['Efficiency'] >= threshold, 1, 0)

# Select features (all except 'Efficiency' and 'sleep_quality') and target variable
features = merged_df.columns.drop(['sleep_quality', 'Efficiency'])
X = merged_df[features]
y = merged_df['sleep_quality']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize the Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

# Get feature importance (coefficients from Logistic Regression)
# importances = lr_model.coef_[0]

# Create a DataFrame for visualization
# feature_importance_df = pd.DataFrame({
#     'Feature': features,
#     'Importance': np.abs(importances)  # Use absolute values for ranking
# }).sort_values(by='Importance', ascending=False)

# # Visualization of feature importance
# plt.figure(figsize=(10, 6))
# sns.barplot(
#     x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm'
# )
# plt.title('Feature Importance (Logistic Regression)')
# plt.xlabel('Importance Score')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()

# Default Interactive Prediction
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

# Convert to DataFrame for predictions
input_df_1 = pd.DataFrame([test_data_1])
input_df_2 = pd.DataFrame([test_data_2])

# Run predictions
prediction_1 = lr_model.predict(input_df_1)
print(f"Test Case 1 Prediction: {'Good' if prediction_1[0] == 1 else 'Poor'}")

prediction_2 = lr_model.predict(input_df_2)
print(f"Test Case 2 Prediction: {'Good' if prediction_2[0] == 1 else 'Poor'}")

import pandas as pd

# List of factors
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

# Convert the collected data into a DataFrame for prediction
user_input_df = pd.DataFrame([user_input_data])

# Assuming `lr_model` is a trained Logistic Regression model
try:
    # Run the prediction
    prediction = lr_model.predict(user_input_df)
    print(f"\nPredicted Sleep Quality: {'Good' if prediction[0] == 1 else 'Poor'}")
except Exception as e:
    print(f"Error during prediction: {e}")
