import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('heart_disease_model.h5')

# Streamlit UI settings
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Function for prediction
def predict_heart_disease(inputs):
    scaler = StandardScaler()
    inputs_scaled = scaler.fit_transform(np.array(inputs).reshape(1, -1))
    prediction = model.predict(inputs_scaled)
    return prediction[0][0]

# Sidebar Menu with options
st.sidebar.title("Menu")
option = st.sidebar.selectbox("Choose an Option", ["Home", "About", "Heart Disease Prediction"])

# Home page
if option == "Home":
    st.title("Welcome to Health360")
    st.write("""
        **Health360** is an AI-powered platform that helps you predict heart disease risk.
        Enter your health data and get predictions on the likelihood of heart disease.
    """)

# About page
elif option == "About":
    st.title("About Health360")
    st.write("""
        **Health360** uses machine learning models to analyze health data and predict the risk of heart disease.
        Developed by **Gurjap Singh** (Age: 17), a passionate AI enthusiast and developer.

        [Visit LinkedIn Profile](https://www.linkedin.com/in/gurjapsingh/)
    """)

# Heart Disease Prediction page
elif option == "Heart Disease Prediction":
    st.title("Heart Disease Risk Prediction")

    # Input fields for user data
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cholesterol = st.slider("Cholesterol Level", 100, 300, 200)
    blood_pressure = st.slider("Blood Pressure", 50, 200, 120)
    max_heart_rate = st.slider("Max Heart Rate", 50, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

    # Mapping categorical inputs
    sex = 1 if sex == "Male" else 0
    exercise_angina = 1 if exercise_angina == "Yes" else 0

    # Prepare input for prediction
    input_data = [age, sex, cholesterol, blood_pressure, max_heart_rate, exercise_angina]

    if st.button("Predict"):
        prediction = predict_heart_disease(input_data)

        # Displaying result with special effects and colors
        if prediction > 0.5:
            st.markdown(
                """
                <div style="background-color: red; padding: 10px; border-radius: 10px; color: white;">
                    <h2><span style="font-size:30px;">❤️ High Risk of Heart Disease</span></h2>
                </div>
                """, 
                unsafe_allow_html=True
            )

            st.subheader("Recommendations & Prevention:")
            st.write("""
                - **Consult a healthcare provider** for further testing and consultation.
                - **Regular Exercise**: Aim for at least 150 minutes of moderate aerobic activity per week.
                - **Healthy Diet**: Focus on a diet rich in fruits, vegetables, and whole grains. Limit saturated fats and salt.
                - **Quit Smoking**: Smoking is a major risk factor for heart disease. Seek support to quit.
                - **Manage Stress**: Practice stress-relieving techniques such as yoga or meditation.
                - **Control Blood Pressure**: Regularly monitor and manage your blood pressure.
            """)
            st.image("https://www.seekpng.com/png/full/22-224170_heart-attack-warning-sign-heart-attack-icon-red.png", width=150)

        else:
            st.markdown(
                """
                <div style="background-color: green; padding: 10px; border-radius: 10px; color: white;">
                    <h2><span style="font-size:30px;">💚 Low Risk of Heart Disease</span></h2>
                </div>
                """, 
                unsafe_allow_html=True
            )

            st.subheader("Prevention Tips:")
            st.write("""
                - **Maintain a healthy lifestyle** by eating nutritious food and exercising regularly.
                - **Stay active**: Engage in physical activities like walking, swimming, or biking.
                - **Monitor your health**: Regular checkups can catch any early signs of health issues.
                - **Avoid excessive alcohol consumption** and smoking.
                - **Sleep well**: Ensure you get 7-9 hours of quality sleep each night.
                - **Manage stress**: Practice relaxation techniques like deep breathing, meditation, or yoga.
            """)
            st.image("https://www.seekpng.com/png/full/20-204157_healthy-heart-healthy-heart-icon.png", width=150)

# Run the app
if __name__ == "__main__":
    st.write("Health360 Web App")
