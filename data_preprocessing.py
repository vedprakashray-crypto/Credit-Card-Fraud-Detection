import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Generate synthetic credit card transaction data
def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic credit card transaction data.
    In a real scenario, you would load data from a CSV file or database.
    """
    np.random.seed(42)

    # Create features similar to real credit card datasets
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),  # Time in seconds (2 days)
        'V1': np.random.normal(0, 1, n_samples),
        'V2': np.random.normal(0, 1, n_samples),
        'V3': np.random.normal(0, 1, n_samples),
        'V4': np.random.normal(0, 1, n_samples),
        'V5': np.random.normal(0, 1, n_samples),
        'V6': np.random.normal(0, 1, n_samples),
        'V7': np.random.normal(0, 1, n_samples),
        'V8': np.random.normal(0, 1, n_samples),
        'V9': np.random.normal(0, 1, n_samples),
        'V10': np.random.normal(0, 1, n_samples),
        'V11': np.random.normal(0, 1, n_samples),
        'V12': np.random.normal(0, 1, n_samples),
        'V13': np.random.normal(0, 1, n_samples),
        'V14': np.random.normal(0, 1, n_samples),
        'V15': np.random.normal(0, 1, n_samples),
        'V16': np.random.normal(0, 1, n_samples),
        'V17': np.random.normal(0, 1, n_samples),
        'V18': np.random.normal(0, 1, n_samples),
        'V19': np.random.normal(0, 1, n_samples),
        'V20': np.random.normal(0, 1, n_samples),
        'V21': np.random.normal(0, 1, n_samples),
        'V22': np.random.normal(0, 1, n_samples),
        'V23': np.random.normal(0, 1, n_samples),
        'V24': np.random.normal(0, 1, n_samples),
        'V25': np.random.normal(0, 1, n_samples),
        'V26': np.random.normal(0, 1, n_samples),
        'V27': np.random.normal(0, 1, n_samples),
        'V28': np.random.normal(0, 1, n_samples),
        'Amount': np.random.exponential(100, n_samples),  # Transaction amounts
    }

    # Create target variable (Class): 0 for normal, 1 for fraud
    # Make fraud rare (about 0.6% of transactions)
    fraud_prob = 0.006
    data['Class'] = np.random.choice([0, 1], n_samples, p=[1-fraud_prob, fraud_prob])

    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """
    Preprocess the data: handle missing values, scale features, handle imbalance.
    """
    print("Preprocessing data...")

    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle imbalanced data using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print(f"Original dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Original class distribution: {y.value_counts()}")
    print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")

    return X_resampled, y_resampled, scaler

if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_data()
    print("Generated synthetic data:")
    print(df.head())
    print(f"Dataset shape: {df.shape}")

    # Preprocess data
    X, y, scaler = preprocess_data(df)

    # Save preprocessed data and scaler
    np.save('X_preprocessed.npy', X)
    np.save('y_preprocessed.npy', y)

    import joblib
    joblib.dump(scaler, 'scaler.pkl')

    print("Preprocessed data and scaler saved.")
