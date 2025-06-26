import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/salary_model.pkl')

st.image("https://cdn-icons-png.flaticon.com/512/1077/1077114.png", width=100)
st.title("💼 Salary Prediction App")
st.markdown("### Get a fair and unbiased salary estimate based on your profile")

# Neat layout using columns
col1, col2 = st.columns(2)

with col1:
    experience = st.slider("Years of Experience", 0, 30, 1)
    interview_score = st.slider("Interview Score (0–10)", 0, 10, 5)

with col2:
    education = st.selectbox("Education Level", ['Bachelors', 'Masters', 'PhD'])
    skills = st.multiselect("Skills", ['Python', 'Excel', 'Java', 'SQL'])

st.markdown("---")

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

# Predict and display result
if st.button("💰 Predict Salary"):
    prediction = model.predict(input_df)[0]
    st.markdown(f"""
    <div style="background-color:#e8f5e9;padding:20px;border-radius:10px">
        <h3 style="color:#2e7d32;">Predicted Salary:</h3>
        <h1 style="color:#1b5e20;">₹ {prediction:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [Fayez Ahmad](https://github.com/FAYEZ087)")
