import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score


# Load the dataset
df = pd.read_csv('cleanedData.csv')

# Drop irrelevant features
df = df.drop(['gender_1', 'egoid', 'datadate'], axis=1)

# Create the target variable: sleep_quality
df['sleep_quality'] = np.where(df['Efficiency'] >= 0.9, 1, 0)

# Select features and target
X = df.drop(['Efficiency', 'sleep_quality'], axis=1)
y = df['sleep_quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features for neural network input
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Input layer
    Dropout(0.2),  # Dropout for regularization
    Dense(32, activation='relu'),  # Hidden layer
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Loss function for binary classification
              metrics=['accuracy'])


# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=20,  # Number of epochs
    batch_size=32,  # Batch size
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)


# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions
y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compute ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.2f}")


import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import numpy as np
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

# Convert the dictionary to a DataFrame
user_input_df = pd.DataFrame([user_input_data])

# Scale the input data using the same scaler as in training
scaler = StandardScaler()
user_input_scaled = scaler.fit_transform(user_input_df)

# Assuming `model` is the trained neural network model, make a prediction
try:
    prediction_proba = model.predict(user_input_scaled).flatten()
    prediction = (prediction_proba > 0.5).astype(int)
    print(f"\nPredicted Sleep Quality: {'Good' if prediction[0] == 1 else 'Poor'}")
except Exception as e:
    print(f"Error during prediction: {e}")

    '''Enter the values for your test case:
Enter value for meanrate: 70
Enter value for steps: 8500
Enter value for sedentaryminutes: 200
Enter value for lightlyactiveminutes: 100
Enter value for fairlyactiveminutes: 50
Enter value for veryactiveminutes: 20
Enter value for lowrangecal: 1800
Enter value for fatburncal: 300
Enter value for cardiocal: 200
Enter value for totalCalorie: 2300
Enter value for bedtimedur: 7.5
Enter value for minstofallasleep: 10
Enter value for minsasleep: 420
Enter value for minsawake: 15'''
