import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Edu-Leap Attrition Prediction",
    page_icon="🎓",
    layout="wide"
)

# --- File Paths ---
MODEL_PATH = 'attrition_model.joblib'
DB_PATH = 'edu_leap.db'
CSV_PATH = 'student_master_data.csv'

# --- One-Time Setup: Create DB from CSV if it doesn't exist ---
# This runs once and is not cached with Streamlit's decorators
try:
    if not os.path.exists(DB_PATH):
        st.toast("First time setup: Creating database from CSV...", icon="⚙️")
        df = pd.read_csv(CSV_PATH)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('students', conn, if_exists='replace', index=False)
        conn.close()
        st.toast("Database ready!", icon="✅")
except FileNotFoundError:
    st.error(f"Fatal Error: The source data file '{CSV_PATH}' was not found. Please upload it to your GitHub repository.")
    st.stop() # Stop the app if the CSV is missing
except Exception as e:
    st.error(f"An error occurred during database creation: {e}")
    st.stop() # Stop the app on other errors

# --- Data and Model Loading Functions (Cached) ---

@st.cache_resource
def load_model():
    """Loads the trained machine learning model from file."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    """Connects to the SQLite DB, loads the full student dataset, and closes the connection."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM students", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to read data from the database: {e}")
        return None

# --- Main Application Logic ---

# Load the model and data using the cached functions
model = load_model()
student_df = load_data()

# Check if everything loaded correctly before building the UI
if model is None:
    st.error(f"Fatal Error: Model file '{MODEL_PATH}' not found. Please upload it to your GitHub repository.")
elif student_df is None:
    st.error("Could not load student data from the database. Please check the logs.")
else:
    # --- If all successful, run the main app ---
    st.title("🎓 Edu-Leap: AI-Powered Student Attrition Platform")
    st.markdown("This platform uses machine learning to predict student dropout risk and provide actionable insights for administrators.")

    # Sidebar for Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard Overview", "Single Student Prediction", "At-Risk Students Report"])

    # Page 1: Dashboard Overview
    if page == "Dashboard Overview":
        st.header("Institutional Health Dashboard")
        
        total_students = len(student_df)
        model_features = model.feature_names_in_
        predictions = model.predict(student_df[model_features])
        num_at_risk = np.sum(predictions)
        attrition_rate = (num_at_risk / total_students) * 100 if total_students > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", f"{total_students}")
        col2.metric("Predicted At-Risk Students", f"{num_at_risk}")
        col3.metric("Predicted Attrition Rate", f"{attrition_rate:.2f}%")
        
        st.markdown("---")
        
        st.subheader("Attrition by Department")
        at_risk_df = student_df.copy()
        at_risk_df['is_at_risk'] = predictions
        at_risk_by_dept = at_risk_df[at_risk_df['is_at_risk'] == 1]['Department'].value_counts()
        
        st.bar_chart(at_risk_by_dept)

    # Page 2: Single Student Prediction
    elif page == "Single Student Prediction":
        st.header("Predict Attrition for a Single Student")
        st.markdown("Enter the student's details below to get a risk prediction.")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", 17, 25, 20)
                tenth_perc = st.number_input("10th Percentage", 0.0, 100.0, 85.0, format="%.2f")
                extracurricular_count = st.slider("Number of Extracurricular Activities", 0, 10, 2)
            
            with col2:
                city_tier = st.selectbox("City Tier", options=student_df['City_Tier'].unique())
                twelfth_perc = st.number_input("12th Percentage", 0.0, 100.0, 80.0, format="%.2f")
                fee_status = st.selectbox("Fee Payment Status", options=student_df['Fee_Payment_Status'].unique())

            with col3:
                state = st.selectbox("State", options=student_df['State'].unique())
                entrance_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 75.0, format="%.2f")
                scholarship = st.selectbox("Scholarship Recipient", options=student_df['Scholarship_Recipient'].unique())
            
            course_name = st.selectbox("Course Name", options=student_df['Course_Name'].unique())
            department = student_df[student_df['Course_Name'] == course_name]['Department'].iloc[0]
            st.info(f"Selected Department: **{department}**")

            submitted = st.form_submit_button("🔮 Predict Risk")

        if submitted:
            input_features = model.feature_names_in_
            input_data = pd.DataFrame({
                'Age': [age], 'City_Tier': [city_tier], 'State': [state],
                '10th_Percentage': [tenth_perc], '12th_Percentage': [twelfth_perc],
                'Entrance_Exam_Score': [entrance_score], 'Course_Name': [course_name],
                'Department': [department], 'Fee_Payment_Status': [fee_status],
                'Scholarship_Recipient': [scholarship],
                'Extracurricular_Activity_Count': [extracurricular_count],
                'Avg_Attendance': [75], 'Final_CGPA': [8.0]
            })
            input_data = input_data[input_features]

            prediction_proba = model.predict_proba(input_data)[0][1]
            risk_score = prediction_proba * 100

            if risk_score > 60:
                st.error(f"High Risk: There is a {risk_score:.2f}% probability this student will drop out.", icon="🚨")
            elif risk_score > 30:
                st.warning(f"Medium Risk: There is a {risk_score:.2f}% probability this student will drop out.", icon="⚠️")
            else:
                st.success(f"Low Risk: There is a {risk_score:.2f}% probability this student will drop out.", icon="✅")

    # Page 3: At-Risk Students Report
    elif page == "At-Risk Students Report":
        st.header("Report of At-Risk Students")
        
        model_features = model.feature_names_in_
        risk_probabilities = model.predict_proba(student_df[model_features])[:, 1]
        report_df = student_df.copy()
        report_df['Risk_Probability_%'] = (risk_probabilities * 100).round(2)
        
        risk_threshold = st.slider("Select Risk Threshold (%)", 0, 100, 60)
        at_risk_students = report_df[report_df['Risk_Probability_%'] > risk_threshold].sort_values(by='Risk_Probability_%', ascending=False)

        st.write(f"Found {len(at_risk_students)} students above the {risk_threshold}% risk threshold.")
        
        st.dataframe(at_risk_students[['StudentID', 'Course_Name', 'Final_CGPA', 'Avg_Attendance', 'Fee_Payment_Status', 'Risk_Probability_%']])

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(at_risk_students)
        st.download_button(
            label="📥 Download Report as CSV",
            data=csv,
            file_name=f"at_risk_report_{risk_threshold}.csv",
            mime="text/csv",
        )
