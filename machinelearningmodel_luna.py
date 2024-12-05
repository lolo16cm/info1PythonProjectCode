import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('cleanedData.csv')

# Feature selection
features = [
    'meanrate', 'steps', 'sedentaryminutes', 'lightlyactiveminutes',
    'fairlyactiveminutes', 'veryactiveminutes', 'lowrangecal', 'fatburncal',
    'cardiocal', 'totalCalorie', 'bedtimedur', 'minstofallasleep', 'minsasleep',
    'minsawake'
]
target = 'Efficiency'

X = data[features]

# Create a binary classification target: 1 for good sleep, 0 for poor sleep
threshold =   0.9
# Defining the threshold for good vs. poor sleep
y = (data['Efficiency'] >= threshold).astype(int)  # 1 for good sleep, 0 for poor sleep

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Build the Predictive Model

# Using Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Evaluate the Model

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Model Tuning and Optimization

# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Initialize the RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Define the parameter grid for tuning with reduced search space
param_grid = {
    'n_estimators': [50, 100],               # Reduced number of estimators
    'max_depth': [10, 20],                    # Reduced depth
    'min_samples_split': [2, 5],              # Reduced sample splits
}

# Initialize GridSearchCV with parallel processing and verbose output
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the model with the best parameters
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and the best model
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
# Retrain with Optimal Hyperparameters
# After finding the optimal parameters, retrain the model using the best combination of hyperparameters.

best_model.fit(X_train_scaled, y_train)
import joblib
from sklearn.preprocessing import StandardScaler

# Save the final trained model for later use in the Interactive Prediction System.
joblib.dump(best_model, 'sleep_quality_model.pkl')

# Save the scaler after scaling the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# Test the model：

import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
best_model = joblib.load('sleep_quality_model.pkl')
scaler = joblib.load('scaler.pkl')


# Create new input data (single sample)
new_data = {
    'meanrate': 72,                  # Average heart rate
    'steps': 7000,                   # Steps taken
    'sedentaryminutes': 500,         # Sedentary minutes
    'lightlyactiveminutes': 100,     # Lightly active minutes
    'fairlyactiveminutes': 50,       # Fairly active minutes
    'veryactiveminutes': 30,         # Very active minutes
    'lowrangecal': 1700,             # Low-range calorie burn
    'fatburncal': 350,               # Fat-burn calories
    'cardiocal': 250,                # Cardio calories
    'totalCalorie': 2400,            # Total calories burned
    'bedtimedur': 450,               # Bedtime duration (minutes)
    'minstofallasleep': 20,          # Minutes to fall asleep
    'minsasleep': 430,               # Minutes asleep
    'minsawake': 15,                 # Minutes awake
}


# Convert to DataFrame and scale the input data
new_data_df = pd.DataFrame([new_data])
new_data_scaled = scaler.transform(new_data_df)


# Make prediction
prediction = best_model.predict(new_data_scaled)

# Output the result
if prediction[0] == 1:
    print("Prediction: Good Sleep Quality")
else:
    print("Prediction: Poor Sleep Quality")
