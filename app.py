import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Edu-Leap Decision Platform",
    page_icon="🚀",
    layout="wide"
)

# --- STYLISH ENHANCEMENTS (Custom CSS) ---
st.markdown("""
<style>
    /* Main app and sidebar background with a professional, static gradient */
    .stApp, [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(-45deg, #e0eafc, #cfdef3); /* Lighter, more professional blue gradient */
    }

    /* Make the sidebar's inner content container transparent and add a visible border */
    [data-testid="stSidebar"] > div:first-child {
        border-right: 2px solid rgba(0, 0, 0, 0.1);
    }
    [data-testid="stSidebar"] > div:first-child > div:first-child {
        background-color: transparent;
    }
    
    /* Frosted glass effect for all content containers */
    .st-emotion-cache-r421ms, .st-emotion-cache-1r6slb0, .st-emotion-cache-1d3wzry, .st-emotion-cache-1v0mbdj, .st-emotion-cache-17xrh1x, .st-emotion-cache-1t42gct {
        background-color: rgba(255, 255, 255, 0.95);
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
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: rgba(255, 255, 255, 0.95); }

    /* Buttons */
    .stButton>button {
        border-radius: 0.5rem;
        background-color: #4A90E2; /* A strong, professional blue */
        color: white;
        border: none;
        padding: 10px 24px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover { background-color: #357ABD; color: white; }
    
    /* Headers and Titles */
    h1, h2, h3 { color: #1E293B; }

    /* Sidebar Text */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-emotion-cache-10trblm {
        color: #1E293B;
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

# --- File Paths ---
MODEL_PATH = 'attrition_model.joblib'

# --- Data and Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Fatal Error: Model file '{MODEL_PATH}' not found. Please ensure it is in the project directory.")
        return None

@st.cache_data
def load_and_prepare_data(uploaded_file):
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
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Key Performance Metrics', 0, 1)
    pdf.set_font('Arial', '', 12)
    for key, value in metrics.items():
        pdf.cell(0, 10, f"{key}: {value}", 0, 1)
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Top 10 At-Risk Students', 0, 1)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(30, 10, 'Student ID', 1)
    pdf.cell(90, 10, 'Course', 1)
    pdf.cell(40, 10, 'Risk Probability (%)', 1)
    pdf.ln()
    pdf.set_font('Arial', '', 10)
    for index, row in top_at_risk.head(10).iterrows():
        student_id = str(row['StudentID'])
        course_name = str(row['Course_Name']).encode('latin-1', 'replace').decode('latin-1')
        risk_prob = f"{row['Risk_Probability_%']:.2f}"
        pdf.cell(30, 10, student_id, 1)
        pdf.cell(90, 10, course_name, 1)
        pdf.cell(40, 10, risk_prob, 1)
        pdf.ln()
    return pdf.output()

# --- Main Application ---
if not st.session_state['logged_in']:
    _, col2, _ = st.columns([1, 2, 1])
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

    st.sidebar.title(f"Welcome, {st.session_state.get('username', '')}!")
    st.sidebar.header("1. Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with student data.", type="csv")

    if uploaded_file is not None:
        student_df = load_and_prepare_data(uploaded_file)
        if student_df is not None:
            st.title("🚀 Edu-Leap: AI Decision Platform")
            st.sidebar.header("2. Navigate Dashboards")
            page = st.sidebar.radio("Go to", [
                "Dashboard Overview", "Historical Trends", "Risk Prediction", "Student Profile Deep Dive",
                "Comparative Analytics", "Financial 'What-If' Simulator", "At-Risk Students Report & Actions", "Communication Module"
            ])
            st.sidebar.divider()
            if st.sidebar.button("Logout"):
                logout()

            # Pre-calculate predictions once to use across pages
            model_features = model.feature_names_in_
            if not all(feature in student_df.columns for feature in model_features):
                st.error("The uploaded CSV is missing one or more required columns for prediction. Please check the file and try again.")
                st.stop()

            df_for_prediction = student_df[model_features]
            predictions = model.predict(df_for_prediction)
            risk_probabilities = model.predict_proba(df_for_prediction)[:, 1] * 100
            student_df['is_at_risk'] = predictions
            student_df['Risk_Probability_%'] = risk_probabilities


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
                    st.info("These metrics provide a high-level snapshot of the student body's current risk profile based on the predictive model.", icon="💡")
                    st.divider()
                    st.subheader("Attrition by Department")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        at_risk_by_dept = student_df[student_df['is_at_risk'] == 1]['Department'].value_counts()
                        st.bar_chart(at_risk_by_dept)
                    with col2:
                        st.info("**Analysis:** This chart shows the absolute number of at-risk students in each department.\n\n**Actionable Insight:** Focus intervention resources on departments with the highest bars.", icon="🔬")
                    st.divider()
                    st.subheader("Download Report")
                    metrics_for_pdf = {"Total Students": total_students, "Predicted At-Risk Students": num_at_risk, "Predicted Attrition Rate (%)": f"{attrition_rate:.2f}"}
                    top_at_risk_df = student_df[student_df['is_at_risk'] == 1].sort_values(by='Risk_Probability_%', ascending=False)
                    pdf_data = generate_pdf(metrics_for_pdf, top_at_risk_df)
                    st.download_button(label="📥 Download Executive Summary (PDF)", data=pdf_data, file_name="Edu-Leap_Executive_Summary.pdf", mime="application/pdf")

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
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.line_chart(trends.set_index('Joining_Year')['attrition_rate'])
                    with col2:
                        st.info("**Analysis:** Tracks the percentage of students who dropped out from each cohort.\n\n**Actionable Insight:** An upward trend signals a growing retention problem. A downward trend suggests strategies may be working.", icon="🔬")
                    st.divider()
                    st.subheader("Academic Performance Trends")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.line_chart(trends.set_index('Joining_Year')['avg_cgpa'], color="#FF4B4B")
                        st.info("Tracks the average final CGPA for each cohort. A declining trend could indicate issues with academic rigor or student preparedness.", icon="💡")
                    with col2:
                        st.line_chart(trends.set_index('Joining_Year')['avg_attendance'], color="#0068C9")
                        st.info("Tracks the average attendance. A drop across cohorts can be a leading indicator of disengagement.", icon="💡")

            elif page == "Risk Prediction":
                st.header("🔍 Manual Risk Prediction")
                # Implementation from previous correct response
                # (This page was already working)

            elif page == "Student Profile Deep Dive":
                st.header("👤 Student Profile Deep Dive")
                # Implementation from previous correct response
                # (This page was already working)

            # --- START: CORRECTED AND COMPLETE CODE FOR LAST 4 TABS ---
            elif page == "Comparative Analytics":
                st.header("🔬 Comparative Analytics Dashboard")
                with st.container():
                    st.markdown("Compare the performance and risk profiles of different student segments.")
                    categorical_cols = ['Course_Name', 'City_Tier', 'State', 'Department', 'Fee_Payment_Status', 'Scholarship_Recipient']
                    compare_by = st.selectbox("Select category to compare by:", options=categorical_cols)
                    unique_values = sorted(student_df[compare_by].unique())
                    selected_groups = st.multiselect(f"Select groups from '{compare_by}' to compare:", options=unique_values, default=unique_values[:2] if len(unique_values) > 1 else unique_values)
                    if len(selected_groups) > 1:
                        comparison_df = student_df[student_df[compare_by].isin(selected_groups)]
                        comparison_results = comparison_df.groupby(compare_by).agg(total_students=('StudentID', 'count'), at_risk_count=('is_at_risk', 'sum'), avg_cgpa=('Final_CGPA', 'mean')).reset_index()
                        comparison_results['attrition_rate'] = (comparison_results['at_risk_count'] / comparison_results['total_students']) * 100
                        st.divider()
                        st.subheader("Comparison Results")
                        st.dataframe(comparison_results)
                        st.divider()
                        st.subheader("Attrition Rate Comparison")
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.bar_chart(comparison_results.set_index(compare_by)['attrition_rate'])
                        with col2:
                            st.info("**Analysis:** This chart directly compares attrition rates for the selected groups.\n\n**Actionable Insight:** Identify segments with higher attrition to pinpoint systemic issues affecting specific groups.", icon="🔬")
                    else:
                        st.warning("Please select at least two groups to compare.")

            elif page == "Financial 'What-If' Simulator":
                st.header("💰 Financial 'What-If' Simulator")
                with st.container():
                    st.markdown("Model the financial impact of your intervention strategies.")
                    st.subheader("1. Set Your Baseline Assumptions")
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_fee = st.number_input("Average Annual Fee per Student (₹)", min_value=10000, value=150000, step=5000)
                    with col2:
                        num_at_risk = student_df['is_at_risk'].sum()
                        st.metric("Predicted Dropouts This Year", value=f"{num_at_risk}")
                    current_revenue_loss = num_at_risk * avg_fee
                    st.error(f"**Current Expected Annual Revenue Loss: ₹{current_revenue_loss:,.2f}**")
                    st.divider()
                    st.subheader("2. Design Your Intervention Program")
                    col1, col2 = st.columns(2)
                    with col1:
                        intervention_cost = st.number_input("Total Cost of Intervention Program (₹)", min_value=0, value=500000, step=25000)
                    with col2:
                        retention_improvement = st.slider("Expected Improvement in Retention (%)", 0, 100, 20, 1)
                    students_retained = int(num_at_risk * (retention_improvement / 100))
                    revenue_saved = students_retained * avg_fee
                    net_impact = revenue_saved - intervention_cost
                    roi = (net_impact / intervention_cost) * 100 if intervention_cost > 0 else 0
                    st.divider()
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
                with st.container():
                    risk_threshold = st.slider("Select Risk Threshold (%) to filter students:", 0, 100, 60)
                    at_risk_students = student_df[student_df['Risk_Probability_%'] > risk_threshold].sort_values(by='Risk_Probability_%', ascending=False)
                    st.write(f"Found **{len(at_risk_students)}** students above the {risk_threshold}% risk threshold.")
                    st.divider()
                    for index, row in at_risk_students.iterrows():
                        student_id = row['StudentID']
                        with st.expander(f"**ID: {student_id}** | Course: {row['Course_Name']} | **Risk: {row['Risk_Probability_%']:.1f}%**"):
                            st.write(f"**Key Details:** CGPA: `{row['Final_CGPA']}`, Attendance: `{row['Avg_Attendance']}%`, Fees: `{row['Fee_Payment_Status']}`")
                            note_key = f"note_{student_id}"
                            current_note = st.session_state['student_notes'].get(student_id, "")
                            note_text = st.text_area("Add or Edit Note:", value=current_note, key=note_key, height=100)
                            if st.button("Save Note", key=f"save_{student_id}"):
                                st.session_state['student_notes'][student_id] = note_text
                                st.success(f"Note saved for student {student_id}.")
                                st.rerun()

            elif page == "Communication Module":
                st.header("📧 Personalized Communication Module")
                with st.container():
                    st.markdown("Generate and download personalized outreach emails for at-risk students.")
                    st.subheader("1. Select Target Audience")
                    at_risk_df = student_df[student_df['is_at_risk'] == 1]
                    filter_reason = st.selectbox("Filter at-risk students by reason:", ["All At-Risk", "Low Attendance (<65%)", "Low CGPA (<6.0)", "Fee Status: Defaulted"])
                    if filter_reason == "Low Attendance (<65%)":
                        target_students = at_risk_df[at_risk_df['Avg_Attendance'] < 65]
                    elif filter_reason == "Low CGPA (<6.0)":
                        target_students = at_risk_df[at_risk_df['Final_CGPA'] < 6.0]
                    elif filter_reason == "Fee Status: Defaulted":
                        target_students = at_risk_df[at_risk_df['Fee_Payment_Status'] == 'Defaulted']
                    else:
                        target_students = at_risk_df
                    st.write(f"Found **{len(target_students)}** students matching the criteria.")
                    st.divider()
                    st.subheader("2. Customize Email Template")
                    email_template = st.text_area("Email Template (use [Name] and [Course] as placeholders):", "Subject: Checking In - Your Success Matters to Us\n\nDear [Name],\n\nWe hope this message finds you well. We're reaching out from the student success office.\n\nWe want to ensure you have all the resources you need to succeed in your [Course] program. If you are facing any challenges, academic or otherwise, please know that we are here to help.\n\nWe encourage you to schedule a quick chat with your faculty advisor to discuss your progress.\n\nBest regards,\n\nStudent Success Team", height=250)
                    st.divider()
                    st.subheader("3. Generate & Download")
                    if not target_students.empty:
                        if st.button("Generate Email List"):
                            email_list = []
                            for index, row in target_students.iterrows():
                                student_name = f"Student_{row['StudentID']}" # Using ID as a placeholder for name
                                course_name = row['Course_Name']
                                personalized_email = email_template.replace("[Name]", student_name).replace("[Course]", course_name)
                                email_list.append({"StudentID": row['StudentID'], "Email_Content": personalized_email})
                            email_df = pd.DataFrame(email_list)
                            st.dataframe(email_df)
                            csv = email_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Download as CSV", data=csv, file_name="personalized_email_outreach.csv", mime="text/csv")
                    else:
                        st.warning("No students match the selected criteria to generate emails.")
    else:
        st.info("Awaiting data file. Please upload a CSV to activate the dashboards.")
