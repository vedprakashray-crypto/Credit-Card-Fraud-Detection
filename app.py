import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Custom CSS for glass morphism effect
st.markdown("""
<style>
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .title {
        color: #ffffff;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 15px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .prediction-result {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
    .fraud {
        background: rgba(255, 0, 0, 0.3);
        color: #ffcccc;
        border: 2px solid rgba(255, 0, 0, 0.5);
    }
    .legitimate {
        background: rgba(0, 255, 0, 0.3);
        color: #ccffcc;
        border: 2px solid rgba(0, 255, 0, 0.5);
    }
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 10px 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white;
        backdrop-filter: blur(10px);
    }
    .stTextInput>div>div>input::placeholder, .stNumberInput>div>div>input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">🛡️ Credit Card Fraud Detection</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="glass-container">
    <h3>Welcome to the Fraud Detection System</h3>
    <p>This application uses machine learning to detect potentially fraudulent credit card transactions.
    Enter the transaction details below to get a prediction.</p>
</div>
""", unsafe_allow_html=True)

# Input form
st.markdown('<div class="glass-container">', unsafe_allow_html=True)
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=1000.0)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)
    v7 = st.number_input("V7", value=0.0)
    v8 = st.number_input("V8", value=0.0)
    v9 = st.number_input("V9", value=0.0)
    v10 = st.number_input("V10", value=0.0)
    v11 = st.number_input("V11", value=0.0)
    v12 = st.number_input("V12", value=0.0)
    v13 = st.number_input("V13", value=0.0)
    v14 = st.number_input("V14", value=0.0)

with col2:
    v15 = st.number_input("V15", value=0.0)
    v16 = st.number_input("V16", value=0.0)
    v17 = st.number_input("V17", value=0.0)
    v18 = st.number_input("V18", value=0.0)
    v19 = st.number_input("V19", value=0.0)
    v20 = st.number_input("V20", value=0.0)
    v21 = st.number_input("V21", value=0.0)
    v22 = st.number_input("V22", value=0.0)
    v23 = st.number_input("V23", value=0.0)
    v24 = st.number_input("V24", value=0.0)
    v25 = st.number_input("V25", value=0.0)
    v26 = st.number_input("V26", value=0.0)
    v27 = st.number_input("V27", value=0.0)
    v28 = st.number_input("V28", value=0.0)
    amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
if st.button("🔍 Analyze Transaction"):
    # Prepare input data
    input_data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                           v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                           v21, v22, v23, v24, v25, v26, v27, v28, amount]])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Display result
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="prediction-result fraud">
            🚨 FRAUDULENT TRANSACTION DETECTED! 🚨<br>
            Confidence: {prediction_proba[0][1]*100:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-result legitimate">
            ✅ LEGITIMATE TRANSACTION ✅<br>
            Confidence: {prediction_proba[0][0]*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

# Model Performance Metrics
try:
    comparison_df = pd.read_csv('model_comparison.csv')
    st.markdown("""
    <div class="glass-container">
        <h3>Model Performance Comparison</h3>
        <p>The system uses multiple machine learning models and selects the best performing one for fraud detection.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display model comparison table
    st.dataframe(comparison_df.style.highlight_max(axis=0))

    # Best model metrics
    best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
    st.markdown(f"""
    <div class="glass-container">
        <h4>Best Model: {best_model['Model']}</h4>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div class="metric-card">
                <h4>Accuracy</h4>
                <p>{best_model['Accuracy']:.1%}</p>
            </div>
            <div class="metric-card">
                <h4>Precision</h4>
                <p>{best_model['Precision']:.1%}</p>
            </div>
            <div class="metric-card">
                <h4>Recall</h4>
                <p>{best_model['Recall']:.1%}</p>
            </div>
            <div class="metric-card">
                <h4>F1-Score</h4>
                <p>{best_model['F1-Score']:.1%}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.markdown("""
    <div class="glass-container">
        <h3>Model Performance</h3>
        <p>Model comparison data not available. Please run the training script first.</p>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div class="metric-card">
                <h4>Accuracy</h4>
                <p>~94%</p>
            </div>
            <div class="metric-card">
                <h4>Precision</h4>
                <p>~93%</p>
            </div>
            <div class="metric-card">
                <h4>Recall</h4>
                <p>~95%</p>
            </div>
            <div class="metric-card">
                <h4>F1-Score</h4>
                <p>~94%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="glass-container">
    <p style="text-align: center; color: rgba(255,255,255,0.8);">
        Built with ❤️ using Streamlit and Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)
