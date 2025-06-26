import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/salary_model.pkl')

st.title("💰 Salary Prediction App")

# Input fields
experience = st.slider("Years of Experience", 0, 30, 1)
education = st.selectbox("Education Level", ['Bachelors', 'Masters', 'PhD'])
interview_score = st.slider("Interview Score (0-10)", 0, 10, 5)
skills = st.multiselect("Skills", ['Python', 'Excel', 'Java', 'SQL'])

# Encode inputs
input_dict = {
    'Experience': experience,
    'Interview_Score': interview_score,
    'Education_Masters': 1 if education == 'Masters' else 0,
    'Education_PhD': 1 if education == 'PhD' else 0,
    'Skills_Python': int('Python' in skills),
    'Skills_Excel': int('Excel' in skills),
    'Skills_Java': int('Java' in skills),
    'Skills_SQL': int('SQL' in skills)
}
input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ₹{prediction:,.2f}")
