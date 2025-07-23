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

# --- Initialize Session State ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'users' not in st.session_state:
    st.session_state['users'] = {"admin": "admin123", "guest": "guest"}
if 'student_notes' not in st.session_state:
    st.session_state['student_notes'] = {}
if 'registration_success' not in st.session_state:
    st.session_state['registration_success'] = False

# --- File Paths ---
MODEL_PATH = 'attrition_model.joblib'

# --- Data and Model Loading ---

@st.cache_resource
def load_model():
    """Loads the ML model."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Fatal Error: Model file '{MODEL_PATH}' not found. Please ensure it is in the GitHub repository.")
        return None

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Reads data from the user's uploaded file and prepares it."""
    try:
        df = pd.read_csv(uploaded_file)
        if 'Joining_Year' not in df.columns:
            np.random.seed(42)
            years = [2021, 2022, 2023, 2024]
            df['Joining_Year'] = np.random.choice(years, size=len(df))
        return df
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None

# --- Authentication & UI Functions ---

def check_login(username, password):
    if username in st.session_state['users'] and st.session_state['users'][username] == password:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.rerun()
    else:
        st.error("Incorrect username or password")

def register_user(username, password):
    if not username or not password:
        st.warning("Username and password cannot be empty.")
        return
    if username in st.session_state['users']:
        st.error("Username already exists.")
    else:
        st.session_state['users'][username] = password
        st.session_state['registration_success'] = True
        st.rerun()

def logout():
    st.session_state['logged_in'] = False
    st.session_state.pop('username', None)
    st.rerun()

# --- Main Application ---

if not st.session_state['logged_in']:
    st.title("🎓 Welcome to Edu-Leap")
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.form_submit_button("Login"):
                check_login(username, password)
        st.info("Default users: `admin`/`admin123` or `guest`/`guest`")
    with register_tab:
        st.header("Register New User")
        if st.session_state.get('registration_success'):
            st.success("Registration successful! Please switch to the Login tab to continue.")
            if st.button("Register another user?"):
                st.session_state['registration_success'] = False
                st.rerun()
        else:
            with st.form("register_form"):
                new_username = st.text_input("Choose a Username", key="reg_user")
                new_password = st.text_input("Choose a Password", type="password", key="reg_pass")
                if st.form_submit_button("Register"):
                    register_user(new_username, new_password)
else:
    model = load_model()
    if model is None:
        st.stop()
    
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    st.sidebar.header("1. Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with student data.", type="csv")

    if uploaded_file is not None:
        student_df = load_and_prepare_data(uploaded_file)
        if student_df is not None:
            st.title("🚀 Edu-Leap: AI Decision Platform")
            st.sidebar.header("2. Navigate Dashboards")
            page = st.sidebar.radio("Go to", [
                "Dashboard Overview", 
                "Historical Trends", 
                "Risk Prediction", 
                "Financial 'What-If' Simulator",
                "At-Risk Students Report & Actions"
            ])

            if page == "Risk Prediction":
                st.header("🔍 Manual Risk Prediction")
                st.markdown("Enter a student's details manually to assess their dropout risk and receive tailored intervention strategies.")
                
                with st.form("prediction_form"):
                    st.subheader("Student Details")
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
                    
                    avg_attendance = st.slider("Assumed Average Attendance (%)", 0, 100, 75)
                    final_cgpa = st.slider("Assumed Final CGPA", 0.0, 10.0, 8.0)
                    
                    submitted = st.form_submit_button("🔮 Predict Risk & Get Recommendations")

                if submitted:
                    model_features = model.feature_names_in_
                    input_data_df = pd.DataFrame({
                        'Age': [age], 'City_Tier': [city_tier], 'State': [state],
                        '10th_Percentage': [tenth_perc], '12th_Percentage': [twelfth_perc],
                        'Entrance_Exam_Score': [entrance_score], 'Course_Name': [course_name],
                        'Department': [department], 'Fee_Payment_Status': [fee_status],
                        'Scholarship_Recipient': [scholarship],
                        'Extracurricular_Activity_Count': [extracurricular_count],
                        'Avg_Attendance': [avg_attendance], 'Final_CGPA': [final_cgpa]
                    })
                    input_data_df = input_data_df[model_features]

                    prediction_proba = model.predict_proba(input_data_df)[0][1]
                    risk_score = prediction_proba * 100

                    st.markdown("---")
                    st.subheader("Results")
                    
                    st.write("#### Risk Assessment")
                    if risk_score > 60:
                        st.error(f"**High Risk:** {risk_score:.2f}% probability of dropout.", icon="🚨")
                    elif risk_score > 30:
                        st.warning(f"**Medium Risk:** {risk_score:.2f}% probability of dropout.", icon="⚠️")
                    else:
                        st.success(f"**Low Risk:** {risk_score:.2f}% probability of dropout.", icon="✅")

            # Other pages remain the same
            elif page == "Dashboard Overview":
                st.header("Institutional Health Dashboard")
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
                    st.markdown("---")
                    st.subheader("Attrition by Department")
                    at_risk_df = student_df.copy()
                    at_risk_df['is_at_risk'] = predictions
                    at_risk_by_dept = at_risk_df[at_risk_df['is_at_risk'] == 1]['Department'].value_counts()
                    st.bar_chart(at_risk_by_dept)

            elif page == "Historical Trends":
                st.header("Historical Trend Analysis")
                st.markdown("Analyze how key metrics have evolved over different student cohorts.")
                trends = student_df.groupby('Joining_Year').agg(total_students=('StudentID', 'count'), dropout_count=('Is_Dropout', 'sum'), avg_attendance=('Avg_Attendance', 'mean'), avg_cgpa=('Final_CGPA', 'mean')).reset_index()
                trends['attrition_rate'] = (trends['dropout_count'] / trends['total_students']) * 100
                st.subheader("Year-over-Year Data")
                st.dataframe(trends.sort_values(by='Joining_Year'))
                st.markdown("---")
                st.subheader("Attrition Rate Over Time")
                st.line_chart(trends.set_index('Joining_Year')['attrition_rate'])
                st.subheader("Academic Performance Trends")
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(trends.set_index('Joining_Year')['avg_cgpa'], color="#FF4B4B")
                with col2:
                    st.line_chart(trends.set_index('Joining_Year')['avg_attendance'], color="#0068C9")

            elif page == "Financial 'What-If' Simulator":
                st.header("💰 Financial 'What-If' Simulator")
                st.markdown("Model the financial impact of your intervention strategies.")
                st.subheader("1. Set Your Baseline Assumptions")
                col1, col2 = st.columns(2)
                with col1:
                    avg_fee = st.number_input("Average Annual Fee per Student (₹)", min_value=10000, value=150000, step=5000)
                with col2:
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

            elif page == "At-Risk Students Report & Actions":
                st.header("📝 At-Risk Students Report & Action Tracker")
                model_features = model.feature_names_in_
                risk_probabilities = model.predict_proba(student_df[model_features])[:, 1]
                report_df = student_df.copy()
                report_df['Risk_Probability_%'] = (risk_probabilities * 100).round(2)
                risk_threshold = st.slider("Select Risk Threshold (%)", 0, 100, 60)
                at_risk_students = report_df[report_df['Risk_Probability_%'] > risk_threshold].sort_values(by='Risk_Probability_%', ascending=False)
                st.write(f"Found {len(at_risk_students)} students above the {risk_threshold}% risk threshold.")
                for index, row in at_risk_students.iterrows():
                    student_id = row['StudentID']
                    with st.expander(f"**ID: {student_id}** | Course: {row['Course_Name']} | Risk: {row['Risk_Probability_%']:.2f}%"):
                        st.write(f"**Key Details:** CGPA: {row['Final_CGPA']}, Attendance: {row['Avg_Attendance']}%")
                        note_key = f"note_{student_id}"
                        current_note = st.session_state['student_notes'].get(student_id, "")
                        note_text = st.text_area("Add or Edit Note:", value=current_note, key=note_key)
                        if st.button("Save Note", key=f"save_{student_id}"):
                            st.session_state['student_notes'][student_id] = note_text
                            st.success(f"Note saved for student {student_id}.")
    else:
        st.info("Awaiting data file. Please upload a CSV to activate the dashboards.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        logout()
