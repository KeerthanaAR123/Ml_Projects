import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report 

# Load the dataset

df = pd.read_csv('heart_disease_data.csv')
# Suppress warnings for a clean output (we fix causes below)
warnings.filterwarnings("ignore")

# Minimal clean info
print(f"Dataset shape: {df.shape}")
print("Target distribution:")
print(df['target'].value_counts())

# Features / target
x = df.drop(columns='target', axis=1)
y = df['target']
print(f"Feature sample:\n{x.head()}")
print(f"Target sample:\n{y.head()}")

#splitting the train ,test data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(f"Total/features/train/test shapes: {x.shape} / {x_train.shape} / {x_test.shape}")

# Scale features to help convergence and stable behavior
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model training (use a solver suitable for small datasets)
model = LogisticRegression(solver='liblinear', max_iter=200)
model.fit(x_train_scaled, y_train)

# Model evaluation
x_train_prediction = model.predict(x_train_scaled)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print(f"Accuracy on training data: {training_data_accuracy:.4f}")
x_test_prediction = model.predict(x_test_scaled)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(f"Accuracy on test data: {test_data_accuracy:.4f}")

#building a predictive system

input_data = (37,0,3,105,200,0,1,100,1,5.3,0,1,1)

# Build a single-row DataFrame with the same column names used for training
input_df = pd.DataFrame([input_data], columns=x.columns)

# Scale the input with the same scaler and predict

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
print(prediction)
if(prediction[0]==0):
    print("The person does not have a heart disease")
else:
    print("The person has heart disease")
