import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3

# --- Configuration ---
st.set_page_config(
    page_title="Edu-Leap Attrition Prediction",
    page_icon="🎓",
    layout="wide"
)

# --- Load Model and Data ---
MODEL_PATH = 'attrition_model.joblib'
DB_PATH = 'edu_leap.db'

@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_model():
    """Loads the trained machine learning model from file."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{MODEL_PATH}'. Please ensure it's in the same directory.")
        return None

@st.cache_data(ttl=600)
def load_data_from_db():
    """Connects to the SQLite DB and loads the full student dataset."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM students", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return pd.DataFrame()

# Load the necessary files
model = load_model()
student_df = load_data_from_db()

# --- Main App UI ---
if model is None or student_df.empty:
    st.warning("Application cannot start because the model or data file is missing.")
else:
    st.title("🎓 Edu-Leap: AI-Powered Student Attrition Platform")
    st.markdown("This platform uses machine learning to predict student dropout risk and provide actionable insights for administrators.")

    # --- Sidebar for Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard Overview", "Single Student Prediction", "At-Risk Students Report"])

    # --- Page 1: Dashboard Overview ---
    if page == "Dashboard Overview":
        st.header("Institutional Health Dashboard")
        
        # Calculate KPIs
        total_students = len(student_df)
        # Use the model to predict on the entire dataset
        predictions = model.predict(student_df)
        num_at_risk = np.sum(predictions)
        attrition_rate = (num_at_risk / total_students) * 100 if total_students > 0 else 0

        # Display KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", f"{total_students}")
        col2.metric("Predicted At-Risk Students", f"{num_at_risk}")
        col3.metric("Predicted Attrition Rate", f"{attrition_rate:.2f}%")
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("Attrition by Department")
        at_risk_df = student_df.copy()
        at_risk_df['is_at_risk'] = predictions
        at_risk_by_dept = at_risk_df[at_risk_df['is_at_risk'] == 1]['Department'].value_counts()
        
        st.bar_chart(at_risk_by_dept)

    # --- Page 2: Single Student Prediction ---
    elif page == "Single Student Prediction":
        st.header("Predict Attrition for a Single Student")
        st.markdown("Enter the student's details below to get a risk prediction.")

        # Create input fields for all model features
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 17, 25, 20)
            tenth_perc = st.number_input("10th Percentage", 0.0, 100.0, 85.0)
            extracurricular_count = st.slider("Number of Extracurricular Activities", 0, 10, 2)
        
        with col2:
            city_tier = st.selectbox("City Tier", options=student_df['City_Tier'].unique())
            twelfth_perc = st.number_input("12th Percentage", 0.0, 100.0, 80.0)
            fee_status = st.selectbox("Fee Payment Status", options=student_df['Fee_Payment_Status'].unique())

        with col3:
            state = st.selectbox("State", options=student_df['State'].unique())
            entrance_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 75.0)
            scholarship = st.selectbox("Scholarship Recipient", options=student_df['Scholarship_Recipient'].unique())
        
        course_name = st.selectbox("Course Name", options=student_df['Course_Name'].unique())
        department = student_df[student_df['Course_Name'] == course_name]['Department'].iloc[0]
        st.text(f"Selected Department: {department}")

        if st.button("🔮 Predict Risk", type="primary"):
            # Create a DataFrame from the inputs
            input_data = pd.DataFrame({
                'Age': [age], 'City_Tier': [city_tier], 'State': [state],
                '10th_Percentage': [tenth_perc], '12th_Percentage': [twelfth_perc],
                'Entrance_Exam_Score': [entrance_score], 'Course_Name': [course_name],
                'Department': [department], 'Fee_Payment_Status': [fee_status],
                'Scholarship_Recipient': [scholarship],
                'Extracurricular_Activity_Count': [extracurricular_count],
                # Add dummy values for columns not in input form but needed by model
                'Avg_Attendance': [75], 'Final_CGPA': [8.0] 
            })

            # Get prediction probability
            prediction_proba = model.predict_proba(input_data)[0][1]
            risk_score = prediction_proba * 100

            if risk_score > 60:
                st.error(f"High Risk: There is a {risk_score:.2f}% probability this student will drop out.", icon="🚨")
            elif risk_score > 30:
                st.warning(f"Medium Risk: There is a {risk_score:.2f}% probability this student will drop out.", icon="⚠️")
            else:
                st.success(f"Low Risk: There is a {risk_score:.2f}% probability this student will drop out.", icon="✅")

    # --- Page 3: At-Risk Students Report ---
    elif page == "At-Risk Students Report":
        st.header("Report of At-Risk Students")
        
        # Run predictions on the full dataset
        risk_probabilities = model.predict_proba(student_df)[:, 1]
        report_df = student_df.copy()
        report_df['Risk_Probability_%'] = (risk_probabilities * 100).round(2)
        
        # Filter for at-risk students
        risk_threshold = st.slider("Select Risk Threshold (%)", 0, 100, 60)
        at_risk_students = report_df[report_df['Risk_Probability_%'] > risk_threshold].sort_values(by='Risk_Probability_%', ascending=False)

        st.write(f"Found {len(at_risk_students)} students above the {risk_threshold}% risk threshold.")
        
        # Display the report table
        st.dataframe(at_risk_students[['StudentID', 'Course_Name', 'Final_CGPA', 'Avg_Attendance', 'Fee_Payment_Status', 'Risk_Probability_%']])

        # Add a download button for the report
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df_to_csv(at_risk_students)
        st.download_button(
            label="📥 Download Report as CSV",
            data=csv,
            file_name=f"at_risk_report_{risk_threshold}.csv",
            mime="text/csv",
        )
