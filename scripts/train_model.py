import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('data/model_comparison.csv')

# Prepare features and target variable
X = data.drop('target', axis=1)  # Assuming 'target' is the column with labels
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, 'models/fraud_detection_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')