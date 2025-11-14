import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def train_models(X, y):
    """
    Train multiple models for fraud detection and compare their performance.
    """
    print("Training multiple models for fraud detection...")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'y_test': y_test
        }

        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    return results

def select_best_model(results):
    """
    Select the best model based on F1-score (important for imbalanced datasets).
    """
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    print(f"\nBest model selected: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")
    return best_model, best_model_name

if __name__ == "__main__":
    # Load preprocessed data
    X = np.load('X_preprocessed.npy')
    y = np.load('y_preprocessed.npy')

    print(f"Loaded data shape: X={X.shape}, y={y.shape}")

    # Train multiple models
    results = train_models(X, y)

    # Select the best model
    best_model, best_model_name = select_best_model(results)

    # Save the best model
    joblib.dump(best_model, 'fraud_detection_model.pkl')
    print(f"Best model ({best_model_name}) saved as 'fraud_detection_model.pkl'")

    # Save model comparison results
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1'] for m in results]
    })
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("Model comparison saved as 'model_comparison.csv'")
