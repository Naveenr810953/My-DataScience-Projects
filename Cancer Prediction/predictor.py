# Import necessary libraries
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import warnings         
          
# Suppress warnings for a cleaner output  
warnings.filterwarnings('ignore')

# --- 1. Load Data and Train the Model (Backend) ---   
# This part is the same as before,  it prepares the model.
# We wrap it in a function to cache it, so it doesn't retrain on every interaction.
@st.cache_data
def train_model():
    # Load the Wisconsin Breast Cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LogisticRegression(max_iter=10000) # Increased max_iter for convergence
    model.fit(X_train, y_train)
    
    return model, data.feature_names, data # Return feature names and full data for slider ranges

# Train the model and get feature names
model, feature_names, data = train_model()


# --- 2. Create the Streamlit Web App (Frontend) ---

# Set the title and a small description for the app
st.set_page_config(page_title="Cancer Prediction Tool", layout="wide")
st.title("🔬 Breast Cancer Prediction Tool")
st.write("""
This app uses a Machine Learning model to predict whether a breast tumor is **Malignant** (cancerous) or **Benign** (not cancerous). 
Please use the sliders on the left to input the feature values from a medical report.
""")
st.write("---")

# --- Sidebar for User Input ---
st.sidebar.header("Input Tumor Features")

# Function to get user input from sliders
def get_user_input():
    inputs = {}
    for feature in feature_names:
        # Get min, max, and mean for realistic slider values
        min_val = float(data.data[:, list(feature_names).index(feature)].min())
        max_val = float(data.data[:, list(feature_names).index(feature)].max())
        mean_val = float(data.data[:, list(feature_names).index(feature)].mean())
        
        # Create a slider in the sidebar for each feature
        inputs[feature] = st.sidebar.slider(
            label=f"Enter {feature}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val # Default to the average value
        )
    return inputs

# Get user input
user_inputs = get_user_input()
input_data = np.array(list(user_inputs.values())).reshape(1, -1)

# --- Main Page for Displaying Results ---

st.header("Prediction")
st.write("Click the button below to see the model's prediction based on your inputs.")

# Create a button to trigger the prediction
if st.button("Predict Now", key='predict_button'):
    # Make the prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Prediction Result:")
    if prediction[0] == 0:
        st.error(f"**Malignant (Cancerous)** 🟥")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
    else:
        st.success(f"**Benign (Not Cancerous)** ✅")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
        
    st.write("---")
    st.info("""
    **Disclaimer:** ⚠️ This is an educational tool and not a substitute for professional medical advice. 
    Consult a qualified healthcare provider for any health concerns.
    """)
