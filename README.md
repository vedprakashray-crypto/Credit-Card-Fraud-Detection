# Credit Card Fraud Detection Project

## Overview
This project demonstrates a complete machine learning pipeline for detecting credit card fraud. It's designed for beginners and includes feature engineering, model training, and a modern web UI.

## Technologies Used

### Python
Python is a high-level programming language that's easy to learn and widely used in data science and machine learning. It has a simple syntax and a vast ecosystem of libraries.

### Pandas
Pandas is a powerful library for data manipulation and analysis. It provides data structures like DataFrames that make it easy to work with tabular data.

### NumPy
NumPy is the fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.

### Scikit-learn
Scikit-learn is a machine learning library for Python. It provides simple and efficient tools for data mining and data analysis, including algorithms for classification, regression, clustering, etc.

### Streamlit
Streamlit is an open-source app framework for Machine Learning and Data Science projects. It turns data scripts into shareable web apps in minutes.

### CSS
CSS (Cascading Style Sheets) is used to describe the presentation of a document written in HTML. Here, we use it to create a modern glass morphism effect.

## How the Project Works

### 1. Data Collection
We use a synthetic credit card transaction dataset that simulates real-world transactions. Each transaction has features like amount, time, and various anonymized features (V1-V28).

### 2. Data Preprocessing and Feature Engineering
- **Handling Missing Values**: Check for and handle any missing data.
- **Feature Scaling**: Normalize features to have similar scales using StandardScaler.
- **Handling Imbalanced Data**: Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes since fraud cases are rare.

### 3. Model Training
We train a Logistic Regression model on the preprocessed data. Logistic Regression is a simple yet effective algorithm for binary classification problems like fraud detection.

### 4. Model Evaluation
We evaluate the model using metrics like accuracy, precision, recall, and F1-score. For imbalanced datasets, recall is particularly important as we want to catch as many fraud cases as possible.

### 5. Web Application
The Streamlit app provides a user-friendly interface where users can:
- View model performance metrics
- Input transaction details for prediction
- See the prediction result with a modern, glass morphism UI

## Project Structure
```
credit-card-fraud-detection/
├── README.md
├── data_preprocessing.py
├── model_training.py
├── app.py
├── fraud_detection_model.pkl
└── scaler.pkl
```

## Running the Project

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn streamlit imbalanced-learn
   ```

2. Run data preprocessing:
   ```
   python data_preprocessing.py
   ```

3. Train the model:
   ```
   python model_training.py
   ```

4. Launch the web app:
   ```
   streamlit run app.py
   ```

## Glass Morphism UI
Glass morphism is a modern design trend that creates a "frosted glass" effect. It uses:
- Semi-transparent backgrounds
- Backdrop filters for blur effects
- Subtle borders and shadows
- Rounded corners

This creates a sleek, modern look that feels futuristic and elegant.
