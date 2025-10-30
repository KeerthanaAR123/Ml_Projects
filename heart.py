import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('heart_disease_data.csv')

# Suppress warnings for a clean output
warnings.filterwarnings("ignore")

# Basic information about dataset
print(f"Dataset shape: {df.shape}")
print("Target distribution:")
print(df['target'].value_counts())

# Separate features and target
x = df.drop(columns='target', axis=1)
y = df['target']
print(f"Feature sample:\n{x.head()}")
print(f"Target sample:\n{y.head()}")

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(f"Total/features/train/test shapes: {x.shape} / {x_train.shape} / {x_test.shape}")

# Standardize features for stable model training
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train logistic regression model
model = LogisticRegression(solver='liblinear', max_iter=200)
model.fit(x_train_scaled, y_train)

# Evaluate model accuracy
x_train_prediction = model.predict(x_train_scaled)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print(f"Accuracy on training data: {training_data_accuracy:.4f}")

x_test_prediction = model.predict(x_test_scaled)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(f"Accuracy on test data: {test_data_accuracy:.4f}")

# Build a predictive system for a new input
input_data = (37,0,3,105,200,0,1,100,1,5.3,0,1,1)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data], columns=x.columns)

# Scale the input and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
print(prediction)

if prediction[0] == 0:
    print("The person does not have heart disease")
else:
    print("The person has heart disease")

# Visualization 1: Target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df, palette='coolwarm')
plt.title('Distribution of Heart Disease (0 = No, 1 = Yes)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# Visualization 2: Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Reds')
plt.title('Feature Correlation Heatmap')
plt.show()

# Visualization 3: Age vs Cholesterol by target
plt.figure(figsize=(7,5))
sns.scatterplot(x='age', y='chol', hue='target', data=df, palette='Set1')
plt.title('Age vs Cholesterol Level')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()

# Visualization 4: Boxplot for resting blood pressure
plt.figure(figsize=(7,5))
sns.boxplot(x='target', y='trestbps', data=df, palette='cool')
plt.title('Resting Blood Pressure vs Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Resting Blood Pressure')
plt.show()

# Visualization 5: Pairplot for selected features
sns.pairplot(df[['age','chol','thalach','trestbps','target']], hue='target', palette='husl')
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()

# Confusion matrix for test data
cm = confusion_matrix(y_test, x_test_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease','Disease'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Test Data")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, x_test_prediction))
