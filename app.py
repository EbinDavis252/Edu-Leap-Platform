import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Edu-Leap Decision Platform",
    page_icon="🚀",
    layout="wide"
)

# --- File Paths & Constants ---
MODEL_PATH = 'attrition_model.joblib'

# --- Data and Model Loading Functions (Cached) ---

@st.cache_resource
def load_model():
    """Loads the trained machine learning model from file."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Fatal Error: Model file '{MODEL_PATH}' not found. Please ensure it is in the GitHub repository.")
        return None

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Reads data from the user's uploaded file and adds 'Joining_Year'."""
    try:
        df = pd.read_csv(uploaded_file)
        if 'Joining_Year' not in df.columns:
            np.random.seed(42)
            years = [2021, 2022, 2023, 2024]
            df['Joining_Year'] = np.random.choice(years, size=len(df))
        return df
    except Exception as e:
        st.error(f"An error occurred while loading or processing the data: {e}")
        return None

# --- AI & Financial Logic Functions ---

def get_recommendations(student_data, model):
    """Generates intervention recommendations based on student data."""
    recommendations = []
    # Define simple rules based on key risk factors
    if student_data['Fee_Payment_Status'].iloc[0] == 'Defaulted':
        recommendations.append("🚨 **High Priority: Financial Counseling.** Connect the student with the financial aid office immediately to discuss payment plans or emergency aid.")
    elif student_data['Fee_Payment_Status'].iloc[0] == 'Delayed':
        recommendations.append("⚠️ **Financial Follow-up:** Send a reminder about fee deadlines and offer information on financial support services.")

    if student_data['Avg_Attendance'].iloc[0] < 65:
        recommendations.append("📚 **Academic Mentorship:** Low attendance is a strong indicator of disengagement. Assign a faculty or peer mentor to check in with the student.")
    
    if student_data['Final_CGPA'].iloc[0] < 6.0:
        recommendations.append("📖 **Tutoring Services:** Proactively offer and enroll the student in subject-specific tutoring or academic skills workshops.")

    if not recommendations:
        recommendations.append("✅ **Monitor Standard Progress:** No immediate high-risk factors detected based on primary rules. Continue standard monitoring.")
        
    return recommendations

# --- Sidebar for Data Upload ---
st.sidebar.title("Setup")
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file with student data.",
    type="csv"
)

# --- Main Application Logic ---
model = load_model()

if model is None:
    st.stop()

if uploaded_file is not None:
    student_df = load_and_prepare_data(uploaded_file)
    
    if student_df is not None:
        st.title("🚀 Edu-Leap: AI Decision Platform")
        
        st.sidebar.header("2. Navigate Dashboards")
        # Added new pages
        page = st.sidebar.radio("Go to", [
            "Dashboard Overview", 
            "Historical Trends", 
            "Risk Prediction & Recommendations", 
            "Financial 'What-If' Simulator",
            "At-Risk Students Report"
        ])

        # --- All Pages ---
        
        if page == "Dashboard Overview":
            st.header("Institutional Health Dashboard")
            # ... (Code for this page remains the same)
            total_students = len(student_df)
            model_features = model.feature_names_in_
            if not all(feature in student_df.columns for feature in model_features):
                st.error("The uploaded CSV is missing one or more required columns for prediction.")
            else:
                df_for_prediction = student_df[model_features]
                predictions = model.predict(df_for_prediction)
                num_at_risk = np.sum(predictions)
                attrition_rate = (num_at_risk / total_students) * 100 if total_students > 0 else 0
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Students", f"{total_students}")
                col2.metric("Predicted At-Risk Students", f"{num_at_risk}")
                col3.metric("Predicted Attrition Rate", f"{attrition_rate:.2f}%")

        elif page == "Historical Trends":
            st.header("Historical Trend Analysis")
            # ... (Code for this page remains the same)
            trends = student_df.groupby('Joining_Year').agg(total_students=('StudentID', 'count'), dropout_count=('Is_Dropout', 'sum'), avg_attendance=('Avg_Attendance', 'mean'), avg_cgpa=('Final_CGPA', 'mean')).reset_index()
            trends['attrition_rate'] = (trends['dropout_count'] / trends['total_students']) * 100
            st.subheader("Year-over-Year Data")
            st.dataframe(trends.sort_values(by='Joining_Year'))
            st.subheader("Attrition Rate Over Time")
            st.line_chart(trends.set_index('Joining_Year')['attrition_rate'])

        elif page == "Risk Prediction & Recommendations":
            st.header("🔍 Risk Prediction & AI Recommendations")
            st.markdown("Select a student by their ID to assess their dropout risk and receive tailored intervention strategies.")
            
            # Select a student from the list
            student_id_to_check = st.selectbox("Select Student ID", options=student_df['StudentID'].unique())
            
            if student_id_to_check:
                student_data = student_df[student_df['StudentID'] == student_id_to_check]
                st.write("#### Student Profile")
                st.write(student_data[['Course_Name', 'Final_CGPA', 'Avg_Attendance', 'Fee_Payment_Status']])

                # Predict risk
                model_features = model.feature_names_in_
                input_data = student_data[model_features]
                prediction_proba = model.predict_proba(input_data)[0][1]
                risk_score = prediction_proba * 100

                st.write("#### Risk Assessment")
                if risk_score > 60:
                    st.error(f"**High Risk:** {risk_score:.2f}% probability of dropout.", icon="🚨")
                elif risk_score > 30:
                    st.warning(f"**Medium Risk:** {risk_score:.2f}% probability of dropout.", icon="⚠️")
                else:
                    st.success(f"**Low Risk:** {risk_score:.2f}% probability of dropout.", icon="✅")

                # Get and display recommendations
                st.write("#### Recommended Interventions")
                recommendations = get_recommendations(student_data, model)
                for rec in recommendations:
                    st.markdown(f"- {rec}")

        elif page == "Financial 'What-If' Simulator":
            st.header("💰 Financial 'What-If' Simulator")
            st.markdown("Model the financial impact of your intervention strategies.")

            st.subheader("1. Set Your Baseline Assumptions")
            col1, col2 = st.columns(2)
            with col1:
                avg_fee = st.number_input("Average Annual Fee per Student (₹)", min_value=10000, value=150000, step=5000)
            with col2:
                # Calculate current predicted dropouts
                model_features = model.feature_names_in_
                predictions = model.predict(student_df[model_features])
                num_at_risk = np.sum(predictions)
                st.metric("Predicted Dropouts This Year", value=f"{num_at_risk}")

            current_revenue_loss = num_at_risk * avg_fee
            st.error(f"**Current Expected Annual Revenue Loss: ₹{current_revenue_loss:,.2f}**")

            st.markdown("---")
            st.subheader("2. Design Your Intervention Program")
            col1, col2 = st.columns(2)
            with col1:
                intervention_cost = st.number_input("Total Cost of Intervention Program (₹)", min_value=0, value=500000, step=25000)
            with col2:
                retention_improvement = st.slider("Expected Improvement in Retention (%)", min_value=0, max_value=100, value=20, step=1)

            # Calculate the impact
            students_retained = int(num_at_risk * (retention_improvement / 100))
            revenue_saved = students_retained * avg_fee
            net_impact = revenue_saved - intervention_cost
            roi = (net_impact / intervention_cost) * 100 if intervention_cost > 0 else 0

            st.markdown("---")
            st.subheader("3. See the Financial Projection")
            st.success(f"**Projected Revenue Saved: ₹{revenue_saved:,.2f}** (by retaining {students_retained} students)")
            
            if net_impact >= 0:
                st.success(f"**Projected Net Financial Impact: +₹{net_impact:,.2f}**")
                st.success(f"**Return on Investment (ROI): {roi:.2f}%**")
            else:
                st.error(f"**Projected Net Financial Impact: -₹{abs(net_impact):,.2f}**")
                st.error(f"**Return on Investment (ROI): {roi:.2f}%**")

        elif page == "At-Risk Students Report":
            st.header("📄 At-Risk Students Report")
            # ... (Code for this page remains the same)
            model_features = model.feature_names_in_
            risk_probabilities = model.predict_proba(student_df[model_features])[:, 1]
            report_df = student_df.copy()
            report_df['Risk_Probability_%'] = (risk_probabilities * 100).round(2)
            risk_threshold = st.slider("Select Risk Threshold (%)", 0, 100, 60)
            at_risk_students = report_df[report_df['Risk_Probability_%'] > risk_threshold].sort_values(by='Risk_Probability_%', ascending=False)
            st.write(f"Found {len(at_risk_students)} students above the {risk_threshold}% risk threshold.")
            display_cols_ideal = ['StudentID', 'Course_Name', 'Final_CGPA', 'Avg_Attendance', 'Fee_Payment_Status', 'Joining_Year', 'Risk_Probability_%']
            cols_to_display = [col for col in display_cols_ideal if col in at_risk_students.columns]
            st.dataframe(at_risk_students[cols_to_display])

else:
    st.title("🎓 Welcome to Edu-Leap")
    st.info("Please upload a student data CSV file using the sidebar to begin analysis.")
    st.image("https://i.imgur.com/2w4sS2A.png", caption="Upload your data to get started.")

