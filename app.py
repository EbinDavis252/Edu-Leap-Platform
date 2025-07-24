import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import base64
from fpdf import FPDF
import hashlib

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Edu-Leap Decision Platform",
    page_icon="🚀",
    layout="wide"
)

# --- STYLISH ENHANCEMENTS (Custom CSS) ---
st.markdown("""
<style>
    /* Keyframes for animation (if you want to re-enable it) */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Main app and sidebar background with a professional, static gradient */
    .stApp, [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(-45deg, #e0eafc, #cfdef3); /* Lighter, more professional blue gradient */
        background-size: 400% 400%;
        /* animation: gradient 15s ease infinite; */ /* Animation disabled for a cleaner look */
    }

    /* Make the sidebar's inner content container transparent and add a visible border */
    [data-testid="stSidebar"] > div:first-child {
        border-right: 2px solid rgba(0, 0, 0, 0.1);
    }
    [data-testid="stSidebar"] > div:first-child > div:first-child {
        background-color: transparent;
    }
    
    /* Frosted glass effect for all content containers (main page) */
    .st-emotion-cache-r421ms, .st-emotion-cache-1r6slb0, .st-emotion-cache-1d3wzry, .st-emotion-cache-1v0mbdj, .st-emotion-cache-17xrh1x, .st-emotion-cache-1t42gct {
        background-color: rgba(255, 255, 255, 0.95); /* More opaque for better readability */
        backdrop-filter: blur(10px);
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.1); /* Softer shadow */
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Center the login/register forms */
    .st-emotion-cache-uf99v8 {
        display: flex;
        justify-content: center;
    }
    
    /* Login and Register tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.95);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 0.5rem;
        background-color: #4A90E2; /* A strong, professional blue */
        color: white;
        border: none;
        padding: 10px 24px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        color: white;
    }
    
    /* Headers and Titles on main page */
    h1, h2, h3 {
        color: #1E293B; /* Dark slate color for text */
    }

    /* Ensure text inside sidebar is readable on the gradient */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-emotion-cache-10trblm {
        color: #1E293B; /* Change sidebar text to dark for better contrast */
    }

</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'users' not in st.session_state:
    st.session_state['users'] = {"admin": "admin123", "guest": "guest"}
if 'student_notes' not in st.session_state:
    st.session_state['student_notes'] = {}
if 'registration_success' not in st.session_state:
    st.session_state['registration_success'] = False
if 'student_df' not in st.session_state:
    st.session_state['student_df'] = None
if 'course_to_dept_map' not in st.session_state:
    st.session_state['course_to_dept_map'] = {}


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
        st.error(f"Fatal Error: Model file '{MODEL_PATH}' not found. Please ensure it is in the project directory.")
        return None

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Reads data from the user's uploaded file and prepares it."""
    try:
        df = pd.read_csv(uploaded_file)
        if 'StudentID' not in df.columns:
            st.error("Fatal Error: 'StudentID' column is missing from the uploaded file. This column is required for the application to function.")
            return None

        # Ensure required columns exist for synthetic data generation if needed
        if 'Joining_Year' not in df.columns:
            np.random.seed(42)
            years = [2021, 2022, 2023, 2024]
            df['Joining_Year'] = np.random.choice(years, size=len(df))
        if 'Is_Dropout' not in df.columns:
            st.warning("Warning: 'Is_Dropout' column not found. Historical trend analysis will be limited.")
            df['Is_Dropout'] = 0 # Add a dummy column to prevent errors
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
    st.session_state['student_df'] = None # Clear data on logout
    st.session_state.pop('username', None)
    st.rerun()

# --- PDF Reporting Function ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Edu-Leap Executive Summary', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(metrics, top_at_risk):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Key Performance Metrics', 0, 1)
    pdf.set_font('Arial', '', 12)
    for key, value in metrics.items():
        pdf.cell(0, 10, f"{key}: {value}", 0, 1)
    
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Top 10 At-Risk Students', 0, 1)
    pdf.set_font('Arial', 'B', 10)
    
    # Create table header
    pdf.cell(30, 10, 'Student ID', 1)
    pdf.cell(90, 10, 'Course', 1)
    pdf.cell(40, 10, 'Risk Probability (%)', 1)
    pdf.ln()

    pdf.set_font('Arial', '', 10)
    # Create table rows
    for index, row in top_at_risk.head(10).iterrows():
        student_id = str(row['StudentID'])
        # This encoding handles potential unicode characters that FPDF doesn't support
        course_name = str(row['Course_Name']).encode('latin-1', 'replace').decode('latin-1')
        risk_prob = f"{row['Risk_Probability_%']:.2f}"
        
        pdf.cell(30, 10, student_id, 1)
        pdf.cell(90, 10, course_name, 1)
        pdf.cell(40, 10, risk_prob, 1)
        pdf.ln()
        
    return pdf.output()


# --- Main Application ---

if not st.session_state['logged_in']:
    _, col2, _ = st.columns([1,2,1])
    with col2:
        st.title("🎓 Welcome to Edu-Leap")
        
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            with st.container(border=False):
                st.header("Login")
                with st.form("login_form"):
                    username = st.text_input("Username", key="login_user")
                    password = st.text_input("Password", type="password", key="login_pass")
                    if st.form_submit_button("Login"):
                        check_login(username, password)
                st.info("Default users: `admin`/`admin123` or `guest`/`guest`")
                
        with register_tab:
            with st.container(border=False):
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

    # --- Data Processing and Prediction (runs once after upload) ---
    if uploaded_file is not None and st.session_state.student_df is None:
        df = load_and_prepare_data(uploaded_file)
        if df is not None:
            model_features = model.feature_names_in_
            if not all(feature in df.columns for feature in model_features):
                st.error("The uploaded CSV is missing one or more required columns for prediction. Please check your file.")
                st.session_state.student_df = "error" # Mark as error to prevent further processing
            else:
                df_for_prediction = df[model_features]
                predictions = model.predict(df_for_prediction)
                risk_probabilities = model.predict_proba(df_for_prediction)[:, 1]
                
                df['is_at_risk'] = predictions
                df['Risk_Probability_%'] = (risk_probabilities * 100)
                st.session_state.student_df = df # Store the processed dataframe in session state
                
                st.session_state.course_to_dept_map = df.drop_duplicates(subset=['Course_Name']).set_index('Course_Name')['Department'].to_dict()
        else:
            st.session_state.student_df = "error"

    # --- Main App Body (displays after data is loaded) ---
    if isinstance(st.session_state.student_df, pd.DataFrame):
        student_df = st.session_state.student_df
        
        st.sidebar.header("2. Navigate Dashboards")
        page = st.sidebar.radio("Go to", [
            "Dashboard Overview", 
            "Historical Trends", 
            "Risk Prediction", 
            "Student Profile Deep Dive",
            "Comparative Analytics",
            "Financial 'What-If' Simulator",
            "At-Risk Students Report & Actions",
            "Communication Module"
        ])
        
        st.sidebar.divider()
        if st.sidebar.button("Logout"):
            logout()

        # --- Page: Dashboard Overview ---
        if page == "Dashboard Overview":
            st.header("📊 Dashboard Overview")
            with st.container():
                total_students = len(student_df)
                num_at_risk = student_df['is_at_risk'].sum()
                attrition_rate = (num_at_risk / total_students) * 100 if total_students > 0 else 0
                
                st.subheader("Key Performance Indicators (KPIs)")
                col1, col2, col3 = st.columns(3)
                col1.metric("👥 Total Students", f"{total_students}")
                col2.metric("❗ Predicted At-Risk", f"{num_at_risk}")
                col3.metric("📉 Predicted Attrition Rate", f"{attrition_rate:.2f}%")
                st.info("These metrics provide a high-level snapshot of the student body's current risk profile based on the predictive model.", icon="🧠")
                
                st.divider()

                st.subheader("Attrition by Department")
                col1, col2 = st.columns([2, 1])
                with col1:
                    at_risk_by_dept = student_df[student_df['is_at_risk'] == 1]['Department'].value_counts()
                    st.bar_chart(at_risk_by_dept)
                with col2:
                    st.info(
                        """
                        **Analysis:**
                        This chart shows the absolute number of at-risk students in each department.
                        
                        **Actionable Insight:**
                        Focus intervention resources on departments with the highest bars, as they represent the largest groups of students needing support.
                        """, icon="🔬"
                    )

                st.divider()
                st.subheader("Download Report")
                metrics_for_pdf = {
                    "Total Students": total_students,
                    "Predicted At-Risk Students": num_at_risk,
                    "Predicted Attrition Rate (%)": f"{attrition_rate:.2f}"
                }
                top_at_risk_df = student_df[student_df['is_at_risk'] == 1].sort_values(by='Risk_Probability_%', ascending=False)
                
                pdf_data = generate_pdf(metrics_for_pdf, top_at_risk_df)
                st.download_button(
                    label="📥 Download Executive Summary (PDF)",
                    data=pdf_data,
                    file_name="Edu-Leap_Executive_Summary.pdf",
                    mime="application/pdf"
                )

        # --- Page: Historical Trends ---
        elif page == "Historical Trends":
            st.header("📈 Historical Trend Analysis")
            with st.container():
                st.markdown("Analyze how key metrics have evolved over different student cohorts.")
                trends = student_df.groupby('Joining_Year').agg(total_students=('StudentID', 'count'), dropout_count=('Is_Dropout', 'sum'), avg_attendance=('Avg_Attendance', 'mean'), avg_cgpa=('Final_CGPA', 'mean')).reset_index()
                trends['attrition_rate'] = (trends['dropout_count'] / trends['total_students']) * 100
                st.subheader("Year-over-Year Data")
                st.dataframe(trends.sort_values(by='Joining_Year'))
                
                st.divider()

                st.subheader("Attrition Rate Over Time")
                col1, col2 = st.columns([2,1])
                with col1:
                    st.line_chart(trends.set_index('Joining_Year')['attrition_rate'])
                with col2:
                    st.info(
                        """
                        **Analysis:**
                        This chart tracks the percentage of students who dropped out from each joining cohort based on historical data.
                        
                        **Actionable Insight:**
                        An upward trend signals a growing retention problem. A downward trend suggests that past retention strategies may be working.
                        """, icon="🔬"
                    )
                
                st.divider()

                st.subheader("Academic Performance Trends")
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(trends.set_index('Joining_Year')['avg_cgpa'], color="#FF4B4B")
                    st.info("Tracks the average final CGPA for each cohort. A declining trend could indicate issues with academic rigor or student preparedness.", icon="💡")
                with col2:
                    st.line_chart(trends.set_index('Joining_Year')['avg_attendance'], color="#0068C9")
                    st.info("Tracks the average attendance. A drop in attendance across cohorts can be a leading indicator of disengagement.", icon="�")

        # --- Page: Risk Prediction ---
        elif page == "Risk Prediction":
            st.header("🔍 Manual Risk Prediction")
            with st.container():
                st.markdown("Enter a student's details manually to assess their dropout risk.")
                with st.form("prediction_form"):
                    st.subheader("Student Details")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        age = st.number_input("Age", 17, 25, 20)
                        tenth_perc = st.number_input("10th Percentage", 0.0, 100.0, 85.0, format="%.2f")
                        extracurricular_count = st.slider("Number of Extracurricular Activities", 0, 10, 2)
                    with col2:
                        city_tier = st.selectbox("City Tier", options=sorted(student_df['City_Tier'].unique()))
                        twelfth_perc = st.number_input("12th Percentage", 0.0, 100.0, 80.0, format="%.2f")
                        fee_status = st.selectbox("Fee Payment Status", options=sorted(student_df['Fee_Payment_Status'].unique()))
                    with col3:
                        state = st.selectbox("State", options=sorted(student_df['State'].unique()))
                        entrance_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 75.0, format="%.2f")
                        scholarship = st.selectbox("Scholarship Recipient", options=sorted(student_df['Scholarship_Recipient'].unique()))
                    
                    course_to_dept_map = st.session_state.course_to_dept_map
                    available_courses = sorted(course_to_dept_map.keys())
                    
                    course_name = st.selectbox("Course Name", options=available_courses, help="This list is populated from your uploaded CSV file.")
                    
                    department = None 
                    if course_name:
                        department = course_to_dept_map.get(course_name)
                        st.info(f"Selected Department: **{department}**")
                    else:
                        st.warning("No courses available to select. Please check your uploaded data file.")

                    avg_attendance = st.slider("Assumed Average Attendance (%)", 0, 100, 75)
                    final_cgpa = st.slider("Assumed Final CGPA", 0.0, 10.0, 8.0, step=0.1)
                    
                    submitted = st.form_submit_button("🔮 Predict Risk")
                
                if submitted:
                    if not department:
                        st.error("Cannot predict risk. No course was selected or available in the data.")
                    else:
                        model_features = model.feature_names_in_
                        input_data_df = pd.DataFrame({'Age': [age], 'City_Tier': [city_tier], 'State': [state], '10th_Percentage': [tenth_perc], '12th_Percentage': [twelfth_perc], 'Entrance_Exam_Score': [entrance_score], 'Course_Name': [course_name], 'Department': [department], 'Fee_Payment_Status': [fee_status], 'Scholarship_Recipient': [scholarship], 'Extracurricular_Activity_Count': [extracurricular_count], 'Avg_Attendance': [avg_attendance], 'Final_CGPA': [final_cgpa]})
                        input_data_df = input_data_df[model_features]
                        prediction_proba = model.predict_proba(input_data_df)[0][1]
                        risk_score = prediction_proba * 100
                        st.subheader("Results")
                        if risk_score > 60:
                            st.error(f"**High Risk:** {risk_score:.2f}% probability of dropout.", icon="🚨")
                        elif risk_score > 30:
                            st.warning(f"**Medium Risk:** {risk_score:.2f}% probability of dropout.", icon="⚠️")
                        else:
                            st.success(f"**Low Risk:** {risk_score:.2f}% probability of dropout.", icon="✅")

        # --- Page: Student Profile Deep Dive ---
        elif page == "Student Profile Deep Dive":
            st.header("👤 Student Profile Deep Dive")
            with st.container():
                st.markdown("Select a student to view their complete profile and performance trends.")
                student_id_to_view = st.selectbox("Select Student ID", options=sorted(student_df['StudentID'].unique()))
                if student_id_to_view:
                    student_data = student_df[student_df['StudentID'] == student_id_to_view].iloc[0]
                    st.subheader(f"Profile for Student ID: {student_id_to_view}")
                    
                    risk_level = student_data['Risk_Probability_%']
                    if risk_level > 60:
                        st.error(f"**High Risk Student:** {risk_level:.2f}% probability of dropout.", icon="🚨")
                    elif risk_level > 30:
                        st.warning(f"**Medium Risk Student:** {risk_level:.2f}% probability of dropout.", icon="⚠️")
                    else:
                        st.success(f"**Low Risk Student:** {risk_level:.2f}% probability of dropout.", icon="✅")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Course", student_data['Course_Name'])
                        st.metric("12th Percentage", f"{student_data['12th_Percentage']}%")
                        st.metric("Fee Status", student_data['Fee_Payment_Status'])
                    with col2:
                        st.metric("Department", student_data['Department'])
                        st.metric("Entrance Score", f"{student_data['Entrance_Exam_Score']}")
                        st.metric("Scholarship", student_data['Scholarship_Recipient'])
                    
                    st.divider()
                    st.subheader("Simulated Performance Trends")
                    
                    seed_value = int(hashlib.md5(str(student_id_to_view).encode()).hexdigest(), 16) % (10**8)
                    np.random.seed(seed_value)

                    semesters = [f"Sem {i}" for i in range(1, 7)]
                    sgpa_trend = np.random.normal(loc=student_data['Final_CGPA'], scale=0.5, size=6)
                    sgpa_trend = np.clip(sgpa_trend, 0, 10)
                    attendance_trend = np.random.normal(loc=student_data['Avg_Attendance'], scale=5, size=6)
                    attendance_trend = np.clip(attendance_trend, 40, 100)
                    trend_df = pd.DataFrame({'Semester': semesters, 'SGPA': sgpa_trend, 'Attendance (%)': attendance_trend}).set_index('Semester')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**SGPA Trend**")
                        st.line_chart(trend_df['SGPA'])
                    with col2:
                        st.write("**Attendance Trend**")
                        st.line_chart(trend_df['Attendance (%)'])
                    
                    st.info("These charts show a *simulated* semester-wise trajectory based on the student's overall profile. A consistent downward trend in either chart is a strong indicator of struggle.", icon="💡")
                    
                    st.divider()
                    st.subheader("Action Log & Notes")
                    current_note = st.session_state['student_notes'].get(student_id_to_view, "No notes yet.")
                    st.text_area("Notes for this student (edit in 'Actions' tab):", value=current_note, height=150, disabled=True)
        
        # --- Page: Comparative Analytics (REVISED) ---
        elif page == "Comparative Analytics":
            st.header("🔬 Filtered Segment Analysis")
            with st.container():
                st.markdown("Use the filters below to isolate and analyze a specific segment of the student population.")
                
                st.subheader("Filter Controls")
                col1, col2 = st.columns(2)
                with col1:
                    dept = st.selectbox("Department", options=["All"] + sorted(student_df['Department'].unique()), key="dept_filter")
                    scholar = st.selectbox("Scholarship Recipient", options=["All"] + sorted(student_df['Scholarship_Recipient'].unique()), key="scholar_filter")
                
                with col2:
                    tier = st.selectbox("City Tier", options=["All"] + sorted(student_df['City_Tier'].unique()), key="tier_filter")
                    fee = st.selectbox("Fee Payment Status", options=["All"] + sorted(student_df['Fee_Payment_Status'].unique()), key="fee_filter")

                # Filtering logic
                filtered_df = student_df.copy()
                if dept != "All":
                    filtered_df = filtered_df[filtered_df['Department'] == dept]
                if tier != "All":
                    filtered_df = filtered_df[filtered_df['City_Tier'] == tier]
                if scholar != "All":
                    filtered_df = filtered_df[filtered_df['Scholarship_Recipient'] == scholar]
                if fee != "All":
                    filtered_df = filtered_df[filtered_df['Fee_Payment_Status'] == fee]

                st.divider()
                st.subheader("Analysis of Filtered Group")

                if filtered_df.empty:
                    st.warning("No students match the selected criteria. Please adjust your filters.")
                else:
                    # Calculate metrics for the filtered group
                    total_students = len(filtered_df)
                    num_at_risk = filtered_df['is_at_risk'].sum()
                    num_not_at_risk = total_students - num_at_risk
                    attrition_rate = (num_at_risk / total_students * 100) if total_students > 0 else 0
                    avg_cgpa = filtered_df['Final_CGPA'].mean()
                    avg_attendance = filtered_df['Avg_Attendance'].mean()

                    # Display metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("👥 Total Students", f"{total_students}")
                    m2.metric("❗ At-Risk Students", f"{num_at_risk}")
                    m3.metric("📉 Attrition Rate", f"{attrition_rate:.2f}%")
                    m4.metric("🎓 Avg. Final CGPA", f"{avg_cgpa:.2f}")
                    
                    st.divider()

                    # Visualization for the filtered group
                    st.write("**Risk Profile of this Segment**")
                    risk_composition = pd.DataFrame({
                        'Category': ['At-Risk', 'Not At-Risk'],
                        'Number of Students': [num_at_risk, num_not_at_risk]
                    }).set_index('Category')

                    st.bar_chart(risk_composition, color="#FF4B4B")
                    
                    st.info(
                        """
                        **How to Interpret this View:**
                        - The metrics above provide a snapshot of the student group you have selected with the filters.
                        - The bar chart visualizes the number of students within this segment who are predicted to be at-risk versus those who are not.
                        - Use these filters to drill down into specific cohorts (e.g., non-scholarship students from Tier 2 cities in the Engineering department) to identify hidden pockets of high risk.
                        """, icon="💡"
                    )


        # --- Page: Financial 'What-If' Simulator ---
        elif page == "Financial 'What-If' Simulator":
            st.header("💸 Financial 'What-If' Simulator")
            with st.container():
                st.markdown("Estimate the financial impact of student attrition and the potential savings from interventions.")
                
                avg_tuition = st.number_input("Enter Average Annual Tuition Fee per Student ($)", min_value=1000, max_value=100000, value=20000, step=1000)
                
                at_risk_df = student_df[student_df['is_at_risk'] == 1]
                num_at_risk = len(at_risk_df)
                potential_loss = num_at_risk * avg_tuition

                st.subheader("Current Financial Exposure")
                st.metric("Number of At-Risk Students", num_at_risk)
                st.metric("Potential Revenue Loss from Attrition", f"${potential_loss:,.2f}")

                st.divider()
                st.subheader("Intervention Savings Calculator")
                
                reduction_perc = st.slider("Target Attrition Reduction (%)", 0, 100, 10)
                
                students_retained = int(num_at_risk * (reduction_perc / 100))
                revenue_saved = students_retained * avg_tuition

                st.success(f"**By reducing attrition among the at-risk group by {reduction_perc}%, you could potentially retain {students_retained} students, saving an estimated ${revenue_saved:,.2f} in tuition fees.**", icon="✅")
                st.info("This is a simplified model. It doesn't account for the costs of intervention programs, but it effectively highlights the financial benefits of student retention.", icon="💡")

        # --- Page: At-Risk Students Report & Actions ---
        elif page == "At-Risk Students Report & Actions":
            st.header("📝 At-Risk Students Report & Actions")
            with st.container():
                st.markdown("View, filter, and take action on the list of students identified as being at risk.")
                at_risk_df = student_df[student_df['is_at_risk'] == 1].sort_values('Risk_Probability_%', ascending=False)
                
                st.dataframe(at_risk_df[['StudentID', 'Course_Name', 'Department', 'Risk_Probability_%', 'Avg_Attendance', 'Final_CGPA', 'Fee_Payment_Status']])

                st.divider()
                st.subheader("Log an Action or Note")
                
                student_id_action = st.selectbox("Select Student ID to Log a Note", options=at_risk_df['StudentID'].unique())
                
                if student_id_action:
                    with st.form(f"note_form_{student_id_action}"):
                        current_note = st.session_state.student_notes.get(student_id_action, "")
                        new_note = st.text_area("Enter note or action taken:", value=current_note, height=150, key=f"note_{student_id_action}")
                        
                        submitted = st.form_submit_button("Save Note")
                        if submitted:
                            st.session_state.student_notes[student_id_action] = new_note
                            st.success(f"Note saved for Student ID {student_id_action}!")

        # --- Page: Communication Module ---
        elif page == "Communication Module":
            st.header("✉️ Communication Module")
            with st.container():
                st.markdown("Generate pre-written communication templates for at-risk students.")
                
                at_risk_students_list = student_df[student_df['is_at_risk'] == 1]['StudentID'].unique()
                student_id_comm = st.selectbox("Select At-Risk Student ID", options=at_risk_students_list)
                
                if student_id_comm:
                    student_data = student_df[student_df['StudentID'] == student_id_comm].iloc[0]
                    
                    template_type = st.selectbox("Choose Template Type", ["Initial Outreach to Student", "Follow-up to Student", "Alert to Academic Advisor"])
                    
                    st.divider()
                    st.subheader("Generated Message")

                    message = ""
                    if template_type == "Initial Outreach to Student":
                        message = f"""
Subject: Checking In: Support and Resources Available

Dear Student {student_id_comm},

I hope this message finds you well.

I'm reaching out from the student success office to check in and see how your semester is going in the {student_data['Course_Name']} program. We want to ensure every student has the support they need to succeed.

We have a wide range of resources available, from academic tutoring to wellness counseling. If you're facing any challenges or just want to chat about your goals, please don't hesitate to schedule a meeting with us.

Best regards,
The Student Success Team
"""
                    elif template_type == "Follow-up to Student":
                        message = f"""
Subject: Following Up

Dear Student {student_id_comm},

Just wanted to follow up on my previous message. Please let us know if there is anything we can do to support you this semester.

Your success is our priority, and we're here to help you navigate any hurdles you might encounter.

Best regards,
The Student Success Team
"""
                    elif template_type == "Alert to Academic Advisor":
                        message = f"""
Subject: Proactive Alert: Student {student_id_comm}

Dear Advisor,

This is a proactive alert regarding Student ID {student_id_comm} in the {student_data['Department']} department.

Our early-warning system has indicated that this student may benefit from some additional support at this time. Key indicators include an average attendance of {student_data['Avg_Attendance']}% and a CGPA of {student_data['Final_CGPA']}.

Could you please consider reaching out to them to offer guidance and support?

Thank you,
Edu-Leap System
"""
                    st.text_area("Copy this template:", message, height=300)

    elif st.session_state.student_df == "error":
        st.error("There was an error processing the uploaded file. Please ensure it has the correct columns and format.")
    else:
        st.info("Awaiting data file. Please upload a CSV in the sidebar to activate the dashboards.")
�
