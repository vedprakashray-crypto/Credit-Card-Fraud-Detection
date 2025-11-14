import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def main():
    # Paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    x_path = os.path.join(root, '..', 'X_preprocessed.npy')
    y_path = os.path.join(root, '..', 'y_preprocessed.npy')
    models_dir = os.path.join(root, 'models')

    # Check data files
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Missing preprocessed data files. Expected:\n  {x_path}\n  {y_path}")
        print("Make sure X_preprocessed.npy and y_preprocessed.npy exist in the project root.")
        return

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    # Split (keeps this small and reproducible)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a simple model (RandomForest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save model and scaler
    model_path = os.path.join(models_dir, 'fraud_detection_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")


if __name__ == '__main__':
    main()
