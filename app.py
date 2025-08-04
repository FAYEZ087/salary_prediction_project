import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('models/salary_model.pkl')

# ------------------ Page Configuration ------------------ #
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

# ------------------ Sidebar ------------------ #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1077/1077114.png", width=100)
    st.title("💼 Salary Predictor")
    st.markdown("""
    This app predicts salaries based on:
    - Experience
    - Education Level
    - Interview Score
    - Technical Skills
    """)

# ------------------ Main Heading ------------------ #
st.markdown("<h2 style='text-align:center;'>🎯 Estimate Your Market Salary</h2>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Input Form ------------------ #
col1, col2 = st.columns(2)

with col1:
    experience = st.slider("Years of Experience", 0, 30, 1)
    interview_score = st.slider("Interview Score (0–10)", 0, 10, 5)

with col2:
    education = st.selectbox("Education Level", ['Bachelors', 'Masters', 'PhD'])
    skills = st.multiselect("Skills", ['Python', 'Excel', 'Java', 'SQL'])

# ------------------ Data Encoding ------------------ #
input_dict = {
    'Experience': experience,
    'Interview_Score': interview_score,
    'Education_Masters': int(education == 'Masters'),
    'Education_PhD': int(education == 'PhD'),
    'Skills_Python': int('Python' in skills),
    'Skills_Excel': int('Excel' in skills),
    'Skills_Java': int('Java' in skills),
    'Skills_SQL': int('SQL' in skills),
}

input_df = pd.DataFrame([input_dict])

# ------------------ Prediction Button ------------------ #
if st.button("💰 Predict Salary"):
    try:
        prediction = model.predict(input_df)[0]
        st.markdown(f"""
        <div style="background-color:#e8f5e9;padding:20px;border-radius:10px">
            <h3 style="color:#2e7d32;">Predicted Salary:</h3>
            <h1 style="color:#1b5e20;">₹ {prediction:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error("⚠️ Error during prediction. Please check your input or model.")

# ------------------ Footer ------------------ #
st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ by <a href='https://github.com/FAYEZ087'>Fayez Ahmad</a></div>", unsafe_allow_html=True)
