# %%
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

#The 2 Lines below allows you to record metrics
import mlflow
mlflow.autolog()

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()


# Load the dataset
# file_path = "path to your dataset"  # Replace with your actual dataset path
data  = pd.read_csv(args.trainingdata)

# Preprocess the data
# Separate features and target
X = data.drop(columns=["Activity", "subject"])  # Drop target and subject columns
y = data["Activity"]

# Encode the target variable (if it's categorical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=40)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=40)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# %%



