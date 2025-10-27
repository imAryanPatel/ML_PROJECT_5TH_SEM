import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/best_random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("Model files not found! Please run train_model.py first.")
        return None, None, None

model, scaler, label_encoder = load_models()

# Title
st.title("🌾 Crop Recommendation System")
st.subheader("Get the best crop recommendation based on soil and climate conditions")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This application uses Machine Learning to recommend the most suitable crop "
    "based on soil nutrients and environmental conditions."
)

st.sidebar.header("Model Info")
if model:
    st.sidebar.success("Model: Random Forest Classifier")
    st.sidebar.info("Features: 7 input parameters")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Soil Nutrients")
    nitrogen = st.number_input("Nitrogen (N) ratio", 0.0, 150.0, 50.0, step=1.0)
    phosphorus = st.number_input("Phosphorus (P) ratio", 0.0, 150.0, 50.0, step=1.0)
    potassium = st.number_input("Potassium (K) ratio", 0.0, 210.0, 50.0, step=1.0)
    ph = st.slider("pH value", 3.0, 10.0, 6.5, step=0.1)

with col2:
    st.subheader("Climate Conditions")
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, step=0.5)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, step=1.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0, step=5.0)

# Prediction section
st.divider()
predict_button = st.button("🌱 Get Crop Recommendation")

if predict_button:
    if model and scaler and label_encoder:
        # Prepare input data
        input_data = pd.DataFrame({
            'Nitrogen': [nitrogen],
            'phosphorus': [phosphorus],
            'potassium': [potassium],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction_encoded = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Decode prediction
        predicted_crop = label_encoder.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba) * 100

        # Display result
        st.success(f"### Recommended Crop: **{predicted_crop.upper()}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Show top 3 predictions
        st.subheader("Top 3 Recommendations")
        top_3_idx = np.argsort(prediction_proba[0])[-3:][::-1]

        cols = st.columns(3)
        for i, idx in enumerate(top_3_idx):
            crop_name = label_encoder.inverse_transform([idx])[0]
            crop_confidence = prediction_proba[0][idx] * 100
            with cols[i]:
                st.metric(label=f"#{i+1} {crop_name.title()}", value=f"{crop_confidence:.1f}%")

        # Input summary
        st.subheader("Input Summary")
        summary_df = pd.DataFrame({
            'Parameter': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
            'Value': [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
            'Unit': ['ratio', 'ratio', 'ratio', '°C', '%', '', 'mm']
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.error("Models not loaded. Please ensure model files exist in the 'models' directory.")

