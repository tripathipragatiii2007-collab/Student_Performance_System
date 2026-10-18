"""
app.py
------
Streamlit Web Application for the Student Performance Report System.

Features:
1. Authentication & Role-Based Access Control:
   - Separate Teacher & Student roles.
   - Secure PBKDF2 password hashing & verification.
   - Student registration with unique roll number and email validation.
   - Pre-seeded default teacher login.
2. Modern Dark Theme UI:
   - Fixed Top Navbar with user profile badge and Logout button.
   - Responsive Slide-out Hamburger Drawer.
3. Teacher Interface:
   - 📊 Dashboard & Analytics: Class KPIs, dark-themed charts, pending submission alert.
   - 📬 Review Submissions: Inspect student submissions, ML preview, approve/reject with remarks.
   - ➕ Add Student & Predict: Teacher manual entry & instant Decision Tree inference.
   - 📋 Student Records & Reports: Search, category filter, CSV export, scorecards.
   - 🧠 Model Evaluation & Insights: Accuracy, confusion matrix, feature importance, ethics.
4. Student Interface:
   - 🏠 Student Dashboard: Personal summary, status badge (Pending/Approved/Rejected), teacher feedback.
   - 👤 My Profile: View and edit permitted profile information.
   - 📝 Submit Academic Details: Form for attendance, study hours, coursework, exam marks, resubmissions.
   - 📑 My Performance Report: Personal verified report, class benchmarks, recommendations, text export.
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import (
    get_records_path,
    initialize_records,
    load_student_records,
    save_student_record,
    validate_student_inputs,
    generate_recommendations,
    initialize_users,
    load_users,
    register_student,
    authenticate_user,
    update_student_profile,
    validate_student_registration,
    validate_academic_inputs,
    submit_student_academic_details,
    get_student_submission,
    teacher_review_submission
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Report System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Modern Fixed Top Navbar, Hamburger Drawer, and Dark Theme
st.markdown("""
<style>
    /* Hide Default Streamlit Sidebar and Header */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Full-Width Main Layout with Top Padding for Fixed Navbar */
    .main {
        background-color: #0E1117 !important;
        color: #F8FAFC !important;
    }
    .block-container {
        max-width: 100% !important;
        padding-top: 78px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* Fixed Top Navbar */
    div[class*="st-key-top_navbar_bar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100vw !important;
        height: 64px !important;
        background-color: #0B0F19 !important;
        border-bottom: 1px solid #1E293B !important;
        z-index: 9999 !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45) !important;
        display: flex !important;
        align-items: center !important;
    }

    .navbar-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .navbar-status-badge {
        text-align: right;
        font-size: 0.88rem;
        color: #94A3B8;
    }
    .navbar-status-badge strong {
        color: #60A5FA;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #10B981;
    }
    .role-badge-teacher {
        background-color: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #6366F1;
        font-size: 0.8rem;
    }
    .role-badge-student {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #10B981;
        font-size: 0.8rem;
    }

    /* Hamburger Button */
    .st-key-btn_hamburger button {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        font-size: 1.3rem !important;
        line-height: 1 !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    .st-key-btn_hamburger button:hover {
        background-color: #2563EB !important;
        border-color: #60A5FA !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.5) !important;
    }

    /* Drawer Overlay */
    .drawer-overlay {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background-color: rgba(0, 0, 0, 0.72) !important;
        backdrop-filter: blur(4px) !important;
        -webkit-backdrop-filter: blur(4px) !important;
        z-index: 10000 !important;
        display: block !important;
    }

    /* Drawer Panel (Left Slide-out) */
    div[class*="st-key-drawer_panel"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 340px !important;
        max-width: 85vw !important;
        height: 100vh !important;
        background-color: #0F172A !important;
        border-right: 1px solid #334155 !important;
        z-index: 10001 !important;
        padding: 24px 18px !important;
        box-shadow: 10px 0 35px rgba(0, 0, 0, 0.8) !important;
        overflow-y: auto !important;
        animation: drawerSlide 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    @keyframes drawerSlide {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }

    .drawer-header h3 {
        margin: 0 !important;
        color: #F8FAFC !important;
        font-size: 1.25rem !important;
    }
    .drawer-divider {
        height: 1px;
        background: #1E293B;
        margin: 14px 0 16px 0;
    }
    .st-key-btn_close_drawer button {
        background-color: transparent !important;
        border: 1px solid #334155 !important;
        color: #94A3B8 !important;
        font-size: 1.1rem !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    .st-key-btn_close_drawer button:hover {
        background-color: #EF4444 !important;
        border-color: #EF4444 !important;
        color: #FFFFFF !important;
    }

    /* Inactive Drawer Button */
    div[class*="st-key-drawer_panel"] button[kind="secondary"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #CBD5E1 !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[class*="st-key-drawer_panel"] button[kind="secondary"]:hover {
        background-color: #334155 !important;
        border-color: #60A5FA !important;
        color: #FFFFFF !important;
        transform: translateX(4px) !important;
    }

    /* Active Highlighted Drawer Button */
    div[class*="st-key-drawer_panel"] button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: 1px solid #60A5FA !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
    }

    /* Cards & Elements in Dark Theme */
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.02rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60A5FA;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .badge-good {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid #10B981;
    }
    .badge-average {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid #F59E0B;
    }
    .badge-needs-improvement {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid #EF4444;
    }
    .report-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        color: #F8FAFC;
    }
    .profile-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .status-badge-approved {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #10B981;
        display: inline-block;
    }
    .status-badge-pending {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #F59E0B;
        display: inline-block;
    }
    .status-badge-rejected {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #EF4444;
        display: inline-block;
    }
    .remarks-box {
        background-color: #0F172A;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
        color: #E2E8F0;
    }
    .auth-card {
        max-width: 520px;
        margin: 40px auto;
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 32px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "student_model.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "model", "model_metrics.json")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "student_data.csv")

# ---------------------------------------------------------
# Helper: Load Model & Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Loads the serialized Decision Tree model bundle."""
    if not os.path.exists(MODEL_PATH):
        try:
            from model.train_model import train_and_evaluate
            train_and_evaluate()
        except Exception as e:
            st.error(f"Error training model: {e}")
            return None
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        return None

def load_metrics():
    """Loads precomputed evaluation metrics."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_badge_html(category: str) -> str:
    """Returns styled HTML badge for performance tier."""
    if category == "Good":
        return '<span class="badge-good">🌟 Good</span>'
    elif category == "Average":
        return '<span class="badge-average">⚡ Average</span>'
    elif category == "Needs Improvement":
        return '<span class="badge-needs-improvement">⚠️ Needs Improvement</span>'
    else:
        return f'<span class="status-badge-pending">{category}</span>'

def get_status_badge_html(status: str) -> str:
    """Returns styled HTML badge for submission status."""
    if status == "Approved":
        return '<span class="status-badge-approved">✅ Approved</span>'
    elif status == "Rejected":
        return '<span class="status-badge-rejected">❌ Needs Revision</span>'
    elif status == "Pending Review":
        return '<span class="status-badge-pending">⏳ Pending Review</span>'
    else:
        return '<span class="status-badge-pending">ℹ️ Not Submitted</span>'

def apply_dark_plot_style(fig, ax):
    """Applies clean dark theme styling to Matplotlib figures and axes."""
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    ax.tick_params(colors='#CBD5E1', labelsize=9)
    ax.xaxis.label.set_color('#E2E8F0')
    ax.yaxis.label.set_color('#E2E8F0')
    ax.title.set_color('#F8FAFC')
    for spine in ax.spines.values():
        spine.set_color('#475569')

# Initialize databases & model
initialize_records()
initialize_users()
model_bundle = load_trained_model()
metrics_info = load_metrics()

# ---------------------------------------------------------
# Session State & Authentication Setup
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "current_page" not in st.session_state:
    st.session_state["current_page"] = ""
if "drawer_open" not in st.session_state:
    st.session_state["drawer_open"] = False

# Navigation Modules per Role
TEACHER_NAV = [
    "📊 Dashboard & Analytics",
    "📬 Review Submissions",
    "➕ Add Student & Predict",
    "📋 Student Records & Reports",
    "🧠 Model Evaluation & Insights"
]

STUDENT_NAV = [
    "🏠 Student Dashboard",
    "👤 My Profile",
    "📝 Submit Academic Details",
    "📑 My Performance Report"
]

# ---------------------------------------------------------
# Authentication Screen (when not logged in)
# ---------------------------------------------------------
if not st.session_state["authenticated"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <h1 style="color: #F8FAFC; margin-bottom: 6px;">🎓 Student Performance Report System</h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">Diagnostic Machine Learning Platform for Educators and Students</p>
    </div>
    """, unsafe_allow_html=True)

    auth_tab1, auth_tab2 = st.tabs(["🔑 Sign In to Portal", "📝 Student Registration"])

    # TAB 1: LOGIN
    with auth_tab1:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.subheader("Sign In")

        login_role = st.radio(
            "Select Role:",
            ["👨‍🏫 Teacher / Faculty", "👨‍🎓 Student"],
            horizontal=True
        )
        selected_role = "teacher" if "Teacher" in login_role else "student"

        with st.form("login_form", clear_on_submit=False):
            if selected_role == "teacher":
                ident_label = "Email Address or Teacher ID"
                ident_placeholder = "e.g., teacher@school.edu"
            else:
                ident_label = "Roll Number or Registered Email"
                ident_placeholder = "e.g., STU2024_01 or rahul@student.edu"

            identifier = st.text_input(ident_label, placeholder=ident_placeholder).strip()
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            submit_login = st.form_submit_button("🚀 Sign In", use_container_width=True)

        if submit_login:
            if not identifier or not password:
                st.error("Please enter both identifier and password.")
            else:
                ok, user_data, msg = authenticate_user(
                    identifier=identifier,
                    password=password,
                    expected_role=selected_role
                )
                if ok:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data
                    st.session_state["role"] = user_data["role"]
                    # Set default landing page based on role
                    if user_data["role"] == "teacher":
                        st.session_state["current_page"] = TEACHER_NAV[0]
                    else:
                        st.session_state["current_page"] = STUDENT_NAV[0]
                    st.success(f"Welcome back, {user_data['name']}!")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        # Quick Demo Credential Helper
        st.markdown("---")
        with st.expander("ℹ️ Demo Login Credentials"):
            st.markdown("""
            - **Teacher Login:**
              - Email: `teacher@school.edu`
              - Password: `teacher123`
            - **Student Login:**
              - Register a new student account in the **Student Registration** tab, or use an existing registered Roll Number.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: REGISTRATION
    with auth_tab2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.subheader("Student Account Registration")
        st.caption("Create a verified student account to submit academic parameters and access performance scorecards.")

        with st.form("student_registration_form", clear_on_submit=False):
            reg_name = st.text_input("Full Name *", placeholder="e.g., Aryan Sharma").strip()
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                reg_roll = st.text_input("Roll Number / Student ID *", placeholder="e.g., 22CS105").strip().upper()
            with r_col2:
                reg_course = st.text_input("Course & Branch *", placeholder="e.g., B.Tech CSE").strip()

            reg_email = st.text_input("College Email Address *", placeholder="e.g., aryan@student.college.edu").strip()

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                reg_pass = st.text_input("Password (min 6 characters) *", type="password")
            with p_col2:
                reg_pass_confirm = st.text_input("Confirm Password *", type="password")

            submit_reg = st.form_submit_button("✅ Create Student Account", use_container_width=True)

        if submit_reg:
            is_valid, reg_errors = validate_student_registration(
                name=reg_name,
                roll_number=reg_roll,
                email=reg_email,
                course_branch=reg_course,
                password=reg_pass,
                confirm_password=reg_pass_confirm
            )

            if not is_valid:
                for err in reg_errors:
                    st.error(f"❌ {err}")
            else:
                ok, msg = register_student(
                    name=reg_name,
                    roll_number=reg_roll,
                    email=reg_email,
                    course_branch=reg_course,
                    password=reg_pass
                )
                if ok:
                    st.success(f"🎉 {msg}")
                    st.info("You can now switch to the **Sign In to Portal** tab to log in with your credentials.")
                else:
                    st.error(f"❌ {msg}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ---------------------------------------------------------
# Authenticated User Context
# ---------------------------------------------------------
current_user = st.session_state["user"]
user_role = st.session_state["role"]

# Ensure current_page belongs to user's authorized role
valid_pages = TEACHER_NAV if user_role == "teacher" else STUDENT_NAV
if st.session_state.get("current_page") not in valid_pages:
    st.session_state["current_page"] = valid_pages[0]

current_page = st.session_state["current_page"]

# Handle overlay click via query params
if "close_drawer" in st.query_params:
    st.session_state["drawer_open"] = False
    try:
        del st.query_params["close_drawer"]
    except Exception:
        pass
    st.rerun()

# ---------------------------------------------------------
# Fixed Top Navbar (with User Role & Logout)
# ---------------------------------------------------------
with st.container(key="top_navbar_bar"):
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([0.05, 0.50, 0.33, 0.12], vertical_alignment="center")
    with nav_col1:
        if st.button("☰", key="btn_hamburger", help="Toggle Navigation Menu"):
            st.session_state["drawer_open"] = not st.session_state["drawer_open"]
            st.rerun()
    with nav_col2:
        st.markdown(
            '<div class="navbar-title">'
            '<span>🎓</span> <span>Student Performance Report System</span>'
            '</div>',
            unsafe_allow_html=True
        )
    with nav_col3:
        clean_name = current_page.split(" ", 1)[-1]
        role_label = f'<span class="role-badge-teacher">Faculty</span>' if user_role == "teacher" else f'<span class="role-badge-student">Student</span>'
        st.markdown(
            f'<div class="navbar-status-badge">'
            f'{role_label} <strong>{current_user["name"]}</strong> | Active: <strong>{clean_name}</strong>'
            f'</div>',
            unsafe_allow_html=True
        )
    with nav_col4:
        if st.button("🚪 Logout", key="btn_logout", help="Sign out of account"):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state["role"] = None
            st.session_state["drawer_open"] = False
            st.rerun()

# ---------------------------------------------------------
# Left Slide-out Drawer & Overlay
# ---------------------------------------------------------
if st.session_state["drawer_open"]:
    # 1. Overlay (dims background; clicking closes drawer)
    st.markdown('<a href="?close_drawer=1" class="drawer-overlay" target="_self" title="Click anywhere to close"></a>', unsafe_allow_html=True)

    # 2. Left Drawer Panel
    with st.container(key="drawer_panel"):
        d_col_title, d_col_close = st.columns([0.8, 0.2], vertical_alignment="center")
        with d_col_title:
            st.markdown('<div class="drawer-header"><h3>🎓 Navigation</h3></div>', unsafe_allow_html=True)
        with d_col_close:
            if st.button("✕", key="btn_close_drawer", help="Close Menu"):
                st.session_state["drawer_open"] = False
                st.rerun()

        st.markdown('<div class="drawer-divider"></div>', unsafe_allow_html=True)
        st.caption(f"LOGGED IN AS: {user_role.upper()}")

        for idx, mod in enumerate(valid_pages):
            is_active = (current_page == mod)
            btn_key = f"drawer_btn_{idx}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(mod, key=btn_key, type=btn_type, use_container_width=True):
                st.session_state["current_page"] = mod
                st.session_state["drawer_open"] = False
                st.rerun()

# =========================================================
# ROUTING & PAGE HANDLERS (ROLE-BASED AUTHORIZATION)
# =========================================================
page = st.session_state["current_page"]

# Strict Authorization Guard: Students cannot access Teacher Pages
if user_role == "student" and page not in STUDENT_NAV:
    st.session_state["current_page"] = STUDENT_NAV[0]
    st.warning("⚠️ Access restricted. You do not have permission to view faculty-only pages.")
    st.rerun()

# =========================================================
# 👨‍🎓 STUDENT PAGES
# =========================================================

# ---------------------------------------------------------
# STUDENT PAGE 1: Student Dashboard
# ---------------------------------------------------------
if page == "🏠 Student Dashboard":
    st.markdown(f'<div class="main-title">🎓 Welcome, {current_user["name"]}!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Your personalized academic diagnostic and progress monitoring center.</div>', unsafe_allow_html=True)

    # 1. Profile Overview Banner
    st.markdown(f"""
    <div class="profile-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <h3 style="margin: 0; color: #F8FAFC;">{current_user['name']}</h3>
                <div style="color: #94A3B8; margin-top: 4px;">
                    Roll Number: <strong style="color: #60A5FA;">{current_user['roll_number']}</strong> &nbsp;|&nbsp;
                    Course: <strong style="color: #CBD5E1;">{current_user['course_branch']}</strong> &nbsp;|&nbsp;
                    Email: <strong style="color: #CBD5E1;">{current_user['email']}</strong>
                </div>
            </div>
            <div>
                <span class="role-badge-student" style="font-size: 0.95rem; padding: 6px 14px;">Student Account</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Check Submission Status
    submission = get_student_submission(current_user["roll_number"])
    status = submission.get("Submission_Status", "Not Submitted") if submission else "Not Submitted"

    st.subheader("📋 Academic Submission Status & Evaluation")

    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    with c1:
        st.markdown(f"**Current Status:** {get_status_badge_html(status)}", unsafe_allow_html=True)
    with c2:
        if submission and submission.get("Submission_Date"):
            st.markdown(f"**Submission Date:** `{submission['Submission_Date']}`")
        else:
            st.markdown("**Submission Date:** *Not submitted yet*")
    with c3:
        if status == "Approved":
            cat = submission.get("Performance_Category", "Average")
            st.markdown(f"**Performance Tier:** {get_badge_html(cat)}", unsafe_allow_html=True)

    # 3. Status Callouts & Content
    if status == "Not Submitted":
        st.info("ℹ️ You have not submitted your academic performance indicators yet. Please click below to submit your attendance, study hours, and exam scores for faculty review.")
        if st.button("📝 Submit Academic Details Now", type="primary"):
            st.session_state["current_page"] = "📝 Submit Academic Details"
            st.rerun()

    elif status == "Pending Review":
        st.warning("⏳ **Your academic details have been submitted and are currently in the Faculty Review Queue.** Your performance classification and tailored report will be unlocked once approved.")
        # Metric indicators preview
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Attendance", f"{submission['Attendance_Percentage']:.1f}%")
        with m_c2:
            st.metric("Study Hours", f"{submission['Study_Hours_Per_Day']:.1f} hrs/day")
        with m_c3:
            st.metric("Assignment Score", f"{submission['Assignment_Marks']:.1f}/100")
        with m_c4:
            st.metric("Exam Marks", f"{submission['Exam_Marks']:.1f}/100")

    elif status == "Rejected":
        st.error("❌ **Your submission was rejected by the course faculty and requires revision.**")
        teacher_remarks = submission.get("Teacher_Remarks", "Please review and revise your submitted parameters.")
        st.markdown(f"""
        <div class="remarks-box" style="border-left-color: #EF4444;">
            <strong>👨‍🏫 Faculty Feedback Remarks:</strong><br>
            <span style="color: #F87171;">{teacher_remarks}</span>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("✏️ Correct & Resubmit Academic Details", type="primary"):
            st.session_state["current_page"] = "📝 Submit Academic Details"
            st.rerun()

    elif status == "Approved":
        st.success("✅ **Your academic details are officially verified and approved by the faculty!**")
        teacher_remarks = submission.get("Teacher_Remarks", "Approved by Course Faculty.")
        st.markdown(f"""
        <div class="remarks-box">
            <strong>👨‍🏫 Faculty Review Remarks:</strong> {teacher_remarks}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Display KPIs
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Verified Attendance", f"{submission['Attendance_Percentage']:.1f}%")
        with m_c2:
            st.metric("Daily Study Hours", f"{submission['Study_Hours_Per_Day']:.1f} hrs")
        with m_c3:
            st.metric("Coursework Marks", f"{submission['Assignment_Marks']:.1f}/100")
        with m_c4:
            st.metric("Exam Marks", f"{submission['Exam_Marks']:.1f}/100")

        st.markdown("<br>", unsafe_allow_html=True)
        # Action Plan Teaser
        category = submission.get("Performance_Category", "Average")
        recs = generate_recommendations(
            attendance=submission["Attendance_Percentage"],
            study_hours=submission["Study_Hours_Per_Day"],
            assignment_marks=submission["Assignment_Marks"],
            exam_marks=submission["Exam_Marks"],
            prediction=category
        )
        st.subheader("💡 Key Personalized Recommendations")
        for r in recs[:2]:
            st.info(f"• **{r['area']}:** {r['message']}")

        st.write("")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("📑 View Full Performance Report & Benchmarks", use_container_width=True, type="primary"):
                st.session_state["current_page"] = "📑 My Performance Report"
                st.rerun()
        with act_col2:
            if st.button("✏️ Update Academic Details (New Review)", use_container_width=True):
                st.session_state["current_page"] = "📝 Submit Academic Details"
                st.rerun()

# ---------------------------------------------------------
# STUDENT PAGE 2: My Profile
# ---------------------------------------------------------
elif page == "👤 My Profile":
    st.markdown('<div class="main-title">👤 My Profile & Account Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">View and update your permitted student information.</div>', unsafe_allow_html=True)

    with st.form("student_profile_edit_form"):
        st.markdown("##### 🪪 Identity Information (Read-Only)")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.text_input("Roll Number / Student ID (Permanent)", value=current_user["roll_number"], disabled=True)
        with p_col2:
            st.text_input("Account Role", value=current_user["role"].capitalize(), disabled=True)

        st.markdown("##### ✏️ Permitted Information Updates")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            new_name = st.text_input("Full Name *", value=current_user["name"]).strip()
            new_email = st.text_input("College Email Address *", value=current_user["email"]).strip()
        with f_col2:
            new_course = st.text_input("Course & Branch *", value=current_user["course_branch"]).strip()
            st.text_input("Account Created On", value=current_user.get("created_at", "N/A"), disabled=True)

        save_profile = st.form_submit_button("💾 Save Profile Updates", use_container_width=True, type="primary")

    if save_profile:
        if not new_name or len(new_name) < 2:
            st.error("Student Name must be at least 2 characters long.")
        elif not new_email or "@" not in new_email:
            st.error("Please provide a valid email address.")
        elif not new_course:
            st.error("Course & Branch cannot be empty.")
        else:
            ok, msg = update_student_profile(
                roll_number=current_user["roll_number"],
                name=new_name,
                email=new_email,
                course_branch=new_course
            )
            if ok:
                current_user["name"] = new_name
                current_user["email"] = new_email
                current_user["course_branch"] = new_course
                st.session_state["user"] = current_user
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

# ---------------------------------------------------------
# STUDENT PAGE 3: Submit Academic Details
# ---------------------------------------------------------
elif page == "📝 Submit Academic Details":
    st.markdown('<div class="main-title">📝 Submit Academic Indicators for Faculty Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your current semester parameters. Submissions are verified by course faculty before ML classification is finalized.</div>', unsafe_allow_html=True)

    submission = get_student_submission(current_user["roll_number"])
    prev_status = submission.get("Submission_Status", "None") if submission else "None"

    # Banner notifications depending on prior state
    if prev_status == "Rejected":
        teacher_remarks = submission.get("Teacher_Remarks", "")
        st.error(f"⚠️ **Your previous submission was rejected by the faculty.**\n\n**Reason:** {teacher_remarks}\n\nPlease correct your parameters below and submit your revision.")
    elif prev_status == "Pending Review":
        st.warning("ℹ️ **You currently have a pending submission awaiting faculty review.** Updating the form below will replace your pending request.")
    elif prev_status == "Approved":
        st.info("ℹ️ **Your current record is Approved.** If you have updated examination scores or attendance for the term, you may submit new values for faculty re-evaluation.")

    # Defaults for form
    def_att = float(submission["Attendance_Percentage"]) if submission else 75.0
    def_study = float(submission["Study_Hours_Per_Day"]) if submission else 3.0
    def_assign = float(submission["Assignment_Marks"]) if submission else 70.0
    def_exam = float(submission["Exam_Marks"]) if submission else 65.0

    with st.form("student_academic_submission_form"):
        st.markdown("##### 📊 Academic Parameters")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            att_val = st.number_input(
                "Classroom Attendance Percentage (%) *",
                min_value=0.0,
                max_value=100.0,
                value=def_att,
                step=0.5,
                help="University norms require at least 75% for exam eligibility."
            )
            study_val = st.number_input(
                "Daily Self-Study Hours (hrs/day) *",
                min_value=0.0,
                max_value=16.0,
                value=def_study,
                step=0.25,
                help="Daily dedicated self-study hours outside class."
            )
        with s_col2:
            assign_val = st.number_input(
                "Internal Assignment Marks (/100) *",
                min_value=0.0,
                max_value=100.0,
                value=def_assign,
                step=1.0,
                help="Continuous evaluation / assignment aggregate out of 100."
            )
            exam_val = st.number_input(
                "Written Examination Marks (/100) *",
                min_value=0.0,
                max_value=100.0,
                value=def_exam,
                step=1.0,
                help="Mid-term or end-semester exam score out of 100."
            )

        submit_details = st.form_submit_button("📤 Submit Details for Faculty Review", use_container_width=True, type="primary")

    if submit_details:
        is_valid, val_errors = validate_academic_inputs(
            attendance=att_val,
            study_hours=study_val,
            assignment_marks=assign_val,
            exam_marks=exam_val
        )

        if not is_valid:
            for err in val_errors:
                st.error(f"❌ {err}")
        else:
            ok, msg = submit_student_academic_details(
                roll_number=current_user["roll_number"],
                name=current_user["name"],
                email=current_user["email"],
                course_branch=current_user["course_branch"],
                attendance=att_val,
                study_hours=study_val,
                assignment_marks=assign_val,
                exam_marks=exam_val
            )
            if ok:
                st.success(f"🎉 {msg}")
                st.info("Your details have been moved to status: **Pending Review**. You can track the approval on your Student Dashboard.")
                if st.button("Go to Student Dashboard"):
                    st.session_state["current_page"] = "🏠 Student Dashboard"
                    st.rerun()
            else:
                st.error(f"❌ {msg}")

# ---------------------------------------------------------
# STUDENT PAGE 4: My Performance Report
# ---------------------------------------------------------
elif page == "📑 My Performance Report":
    st.markdown('<div class="main-title">📑 Official Student Performance Scorecard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comprehensive performance breakdown, peer average comparison, and AI improvement plan.</div>', unsafe_allow_html=True)

    submission = get_student_submission(current_user["roll_number"])
    status = submission.get("Submission_Status", "None") if submission else "None"

    if status != "Approved":
        if status == "Pending Review":
            st.warning("⏳ **Your report is currently awaiting teacher approval.** Once the faculty validates your submission, your full report card, benchmarks, and recommendations will be available here.")
        elif status == "Rejected":
            st.error(f"❌ **Your submission was rejected by the faculty.**\n\nRemarks: {submission.get('Teacher_Remarks', 'Please resubmit.')}")
        else:
            st.info("ℹ️ You have not submitted your academic metrics yet. Please submit your academic details first.")

        if st.button("📝 Go to Academic Submission Form"):
            st.session_state["current_page"] = "📝 Submit Academic Details"
            st.rerun()
    else:
        df_records = load_student_records()
        approved_records = df_records[df_records["Submission_Status"] == "Approved"]

        st.markdown("""<div class="report-box">""", unsafe_allow_html=True)
        h_col1, h_col2 = st.columns([2, 1])
        with h_col1:
            st.markdown(f"### 🎓 Student Performance Report: {submission['Name']}")
            st.markdown(f"**Roll Number:** `{submission['Student_ID']}` &nbsp;|&nbsp; **Branch:** {current_user['course_branch']}")
            st.markdown(f"**Faculty Remarks:** *{submission.get('Teacher_Remarks', 'Approved by Faculty.')}*")
        with h_col2:
            st.markdown(f"**Overall Classification:**")
            st.markdown(get_badge_html(submission['Performance_Category']), unsafe_allow_html=True)

        st.markdown("---")

        # Class Averages
        class_att_avg = approved_records["Attendance_Percentage"].mean() if not approved_records.empty else 75.0
        class_study_avg = approved_records["Study_Hours_Per_Day"].mean() if not approved_records.empty else 4.0
        class_assign_avg = approved_records["Assignment_Marks"].mean() if not approved_records.empty else 70.0
        class_exam_avg = approved_records["Exam_Marks"].mean() if not approved_records.empty else 65.0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Attendance",
                f"{submission['Attendance_Percentage']:.1f}%",
                delta=f"{submission['Attendance_Percentage'] - class_att_avg:+.1f}% vs Class Avg"
            )
        with m2:
            st.metric(
                "Study Hours",
                f"{submission['Study_Hours_Per_Day']:.1f} hrs",
                delta=f"{submission['Study_Hours_Per_Day'] - class_study_avg:+.1f} hrs vs Class Avg"
            )
        with m3:
            st.metric(
                "Assignment Marks",
                f"{submission['Assignment_Marks']:.1f}/100",
                delta=f"{submission['Assignment_Marks'] - class_assign_avg:+.1f} vs Class Avg"
            )
        with m4:
            st.metric(
                "Exam Marks",
                f"{submission['Exam_Marks']:.1f}/100",
                delta=f"{submission['Exam_Marks'] - class_exam_avg:+.1f} vs Class Avg"
            )

        # Benchmark Bar Chart
        st.markdown("##### 📊 Individual vs Class Benchmark")
        fig_bench, ax_bench = plt.subplots(figsize=(7, 3.2))
        apply_dark_plot_style(fig_bench, ax_bench)

        categories_bench = ["Attendance (%)", "Study Hours (x10)", "Assignments (/100)", "Exam Marks (/100)"]
        student_values = [
            submission["Attendance_Percentage"],
            submission["Study_Hours_Per_Day"] * 10,
            submission["Assignment_Marks"],
            submission["Exam_Marks"]
        ]
        class_values = [
            class_att_avg,
            class_study_avg * 10,
            class_assign_avg,
            class_exam_avg
        ]

        ind = np.arange(len(categories_bench))
        width = 0.35
        ax_bench.bar(ind - width/2, student_values, width, label=f"{submission['Name']}", color="#3B82F6")
        ax_bench.bar(ind + width/2, class_values, width, label="Class Average", color="#64748B")
        ax_bench.set_xticks(ind)
        ax_bench.set_xticklabels(categories_bench, fontsize=9)
        ax_bench.set_ylabel("Score / Normalized Scale", fontsize=9)
        ax_bench.legend(fontsize=8, facecolor="#0F172A", edgecolor="#334155")
        ax_bench.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
        plt.tight_layout()
        st.pyplot(fig_bench)
        plt.close()

        # Actionable Recommendations
        st.markdown("##### 💡 Tailored Academic Guidance & Roadmaps")
        recs = generate_recommendations(
            attendance=submission["Attendance_Percentage"],
            study_hours=submission["Study_Hours_Per_Day"],
            assignment_marks=submission["Assignment_Marks"],
            exam_marks=submission["Exam_Marks"],
            prediction=submission["Performance_Category"]
        )
        for r in recs:
            st.markdown(f"• **{r['area']}:** {r['message']}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Download Report as Text
        report_txt = f"""====================================================
STUDENT PERFORMANCE EVALUATION REPORT
====================================================
Student ID / Roll: {submission['Student_ID']}
Student Name:      {submission['Name']}
Branch / Course:   {current_user['course_branch']}
Evaluation Tier:   {submission['Performance_Category']}
Status:            {submission['Submission_Status']}
Faculty Remarks:   {submission.get('Teacher_Remarks', 'Approved')}
----------------------------------------------------
Attendance:        {submission['Attendance_Percentage']:.1f}% (Class Avg: {class_att_avg:.1f}%)
Study Hours/Day:   {submission['Study_Hours_Per_Day']:.1f} hrs (Class Avg: {class_study_avg:.1f} hrs)
Assignment Marks:  {submission['Assignment_Marks']:.1f}/100 (Class Avg: {class_assign_avg:.1f}/100)
Exam Marks:        {submission['Exam_Marks']:.1f}/100 (Class Avg: {class_exam_avg:.1f}/100)
----------------------------------------------------
ACTIONABLE RECOMMENDATIONS:
"""
        for r in recs:
            report_txt += f"- [{r['area']}] {r['message']}\n"
        report_txt += "====================================================\n"

        st.download_button(
            label="📄 Download My Official Report (.txt)",
            data=report_txt,
            file_name=f"Report_{submission['Student_ID']}.txt",
            mime="text/plain"
        )

# =========================================================
# 👨‍🏫 TEACHER PAGES
# =========================================================

# ---------------------------------------------------------
# TEACHER PAGE 1: Dashboard & Analytics
# ---------------------------------------------------------
elif page == "📊 Dashboard & Analytics":
    st.markdown('<div class="main-title">🎓 Student Performance Report System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A data-driven academic monitoring dashboard for educators and administrators.</div>', unsafe_allow_html=True)

    df_records = load_student_records()

    # Pending Submissions Notification Banner
    pending_records = df_records[df_records["Submission_Status"] == "Pending Review"]
    num_pending = len(pending_records)
    if num_pending > 0:
        st.markdown(f"""
        <div style="background-color: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FBBF24; font-weight: 600; font-size: 1rem;">
                🔔 You have <strong>{num_pending} pending student submission(s)</strong> awaiting review & approval!
            </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👉 Go to Review Submissions", key="btn_quick_review"):
            st.session_state["current_page"] = "📬 Review Submissions"
            st.rerun()

    # Focus charts on approved records for consistency
    approved_df = df_records[df_records["Submission_Status"] == "Approved"]
    if approved_df.empty:
        approved_df = df_records

    if df_records.empty:
        st.warning("No student records found. Add student details using the 'Add Student & Predict' page.")
    else:
        total_students = len(approved_df)
        avg_exam = approved_df["Exam_Marks"].mean()
        avg_attendance = approved_df["Attendance_Percentage"].mean()
        good_pct = (approved_df["Performance_Category"] == "Good").mean() * 100

        # KPI Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_students}</div>
                <div class="metric-label">Approved Students</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_exam:.1f}</div>
                <div class="metric-label">Avg Exam Score (/100)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_attendance:.1f}%</div>
                <div class="metric-label">Avg Attendance</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{good_pct:.1f}%</div>
                <div class="metric-label">High Performers (Good)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Analytics Section
        st.subheader("📈 Class Performance Visualizations")
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: Performance Tier Distribution
        with chart_col1:
            cat_counts = approved_df["Performance_Category"].value_counts()
            category_order = ["Good", "Average", "Needs Improvement"]
            counts = [cat_counts.get(cat, 0) for cat in category_order]
            colors = ["#10B981", "#F59E0B", "#EF4444"]

            fig, ax = plt.subplots(figsize=(6, 4))
            apply_dark_plot_style(fig, ax)
            bars = ax.bar(category_order, counts, color=colors, edgecolor="#475569", linewidth=1.0, width=0.55)
            ax.set_title("Distribution of Performance Categories", fontsize=11, fontweight="bold", pad=12)
            ax.set_ylabel("Number of Students", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.3, color="#475569")

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center", va="bottom", fontweight="bold", color="#F8FAFC")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Chart 2: Attendance vs Exam Marks Scatter Plot
        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            apply_dark_plot_style(fig2, ax2)
            color_map = {"Good": "#10B981", "Average": "#F59E0B", "Needs Improvement": "#EF4444"}

            for cat, grp in approved_df.groupby("Performance_Category"):
                ax2.scatter(
                    grp["Attendance_Percentage"],
                    grp["Exam_Marks"],
                    label=cat,
                    color=color_map.get(cat, "#3B82F6"),
                    alpha=0.85,
                    edgecolors="#1E293B",
                    s=50
                )

            ax2.axvline(75, color="#F87171", linestyle=":", linewidth=1.5, label="75% Attendance Norm")
            ax2.axhline(50, color="#94A3B8", linestyle="--", linewidth=1.2, label="50 Mark Passline")

            ax2.set_title("Attendance % vs Exam Marks", fontsize=11, fontweight="bold", pad=12)
            ax2.set_xlabel("Attendance Percentage (%)", fontsize=10)
            ax2.set_ylabel("Exam Marks (/100)", fontsize=10)
            ax2.set_xlim(40, 102)
            ax2.set_ylim(15, 105)
            ax2.grid(True, linestyle="--", alpha=0.25, color="#475569")
            ax2.legend(loc="lower right", fontsize=8, facecolor="#0F172A", edgecolor="#334155")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # Row 2 Charts
        chart_col3, chart_col4 = st.columns(2)

        # Chart 3: Average Marks Comparison by Category
        with chart_col3:
            grouped_means = approved_df.groupby("Performance_Category")[["Assignment_Marks", "Exam_Marks"]].mean()
            grouped_means = grouped_means.reindex(["Good", "Average", "Needs Improvement"]).dropna()

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            apply_dark_plot_style(fig3, ax3)
            x_indices = np.arange(len(grouped_means))
            width = 0.35

            rects1 = ax3.bar(x_indices - width/2, grouped_means["Assignment_Marks"], width, label="Assignment Marks", color="#818CF8")
            rects2 = ax3.bar(x_indices + width/2, grouped_means["Exam_Marks"], width, label="Exam Marks", color="#38BDF8")

            ax3.set_title("Average Coursework Scores by Category", fontsize=11, fontweight="bold", pad=12)
            ax3.set_xticks(x_indices)
            ax3.set_xticklabels(grouped_means.index)
            ax3.set_ylabel("Average Score (/100)", fontsize=10)
            ax3.set_ylim(0, 100)
            ax3.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
            ax3.legend(fontsize=9, facecolor="#0F172A", edgecolor="#334155")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # Chart 4: Daily Study Hours Distribution
        with chart_col4:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            apply_dark_plot_style(fig4, ax4)
            ax4.hist(approved_df["Study_Hours_Per_Day"], bins=10, color="#A78BFA", edgecolor="#1E293B", alpha=0.85)
            ax4.set_title("Distribution of Daily Study Hours", fontsize=11, fontweight="bold", pad=12)
            ax4.set_xlabel("Hours per Day", fontsize=10)
            ax4.set_ylabel("Number of Students", fontsize=10)
            ax4.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        # Recent Records Preview
        st.subheader("📋 Recent Student Records")
        st.dataframe(df_records.tail(10), use_container_width=True)

# ---------------------------------------------------------
# TEACHER PAGE 2: Review Submissions (NEW INTEGRATION)
# ---------------------------------------------------------
elif page == "📬 Review Submissions":
    st.markdown('<div class="main-title">📬 Student Submissions & Faculty Approvals</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Verify student-submitted metrics, provide feedback remarks, and execute Machine Learning classification.</div>', unsafe_allow_html=True)

    df_records = load_student_records()
    pending_df = df_records[df_records["Submission_Status"] == "Pending Review"]
    approved_df = df_records[df_records["Submission_Status"] == "Approved"]
    rejected_df = df_records[df_records["Submission_Status"] == "Rejected"]

    rev_tab1, rev_tab2, rev_tab3 = st.tabs([
        f"⏳ Pending Review ({len(pending_df)})",
        f"✅ Approved Submissions ({len(approved_df)})",
        f"❌ Rejected Submissions ({len(rejected_df)})"
    ])

    # SUB-TAB 1: PENDING SUBMISSIONS
    with rev_tab1:
        if pending_df.empty:
            st.success("🎉 **All caught up!** There are currently no pending student submissions awaiting review.")
        else:
            st.info(f"Showing **{len(pending_df)}** submission(s) requiring faculty action.")

            for p_idx, p_row in pending_df.iterrows():
                stu_id = str(p_row["Student_ID"])
                stu_name = str(p_row["Name"])
                stu_email = str(p_row.get("Student_Email", ""))
                stu_course = str(p_row.get("Course_Branch", "B.Tech"))
                sub_date = str(p_row.get("Submission_Date", "Recently"))

                att = float(p_row["Attendance_Percentage"])
                study = float(p_row["Study_Hours_Per_Day"])
                assign = float(p_row["Assignment_Marks"])
                exam = float(p_row["Exam_Marks"])

                # Decision Tree Inference Preview
                model_pred = "Average"
                if model_bundle:
                    feat_names = model_bundle["feature_names"]
                    in_df = pd.DataFrame([{
                        "Attendance_Percentage": att,
                        "Study_Hours_Per_Day": study,
                        "Assignment_Marks": assign,
                        "Exam_Marks": exam
                    }])[feat_names]
                    model_pred = model_bundle["model"].predict(in_df)[0]

                with st.expander(f"📌 {stu_name} ({stu_id}) — Submitted on {sub_date}", expanded=True):
                    sc1, sc2 = st.columns([1.5, 1])
                    with sc1:
                        st.markdown(f"**Student Name:** {stu_name} &nbsp;|&nbsp; **Roll Number:** `{stu_id}`")
                        st.markdown(f"**Branch/Course:** {stu_course} &nbsp;|&nbsp; **Email:** {stu_email}")

                        # Metrics grid
                        pm1, pm2, pm3, pm4 = st.columns(4)
                        pm1.metric("Attendance", f"{att:.1f}%")
                        pm2.metric("Study Hours", f"{study:.1f} hrs")
                        pm3.metric("Assignment", f"{assign:.1f}/100")
                        pm4.metric("Exam Marks", f"{exam:.1f}/100")

                    with sc2:
                        st.markdown("**AI Decision Tree Preview:**")
                        st.markdown(get_badge_html(model_pred), unsafe_allow_html=True)

                    st.markdown("---")
                    # Review Form
                    r_remarks = st.text_input(
                        "Faculty Review Feedback / Remarks:",
                        value="Approved. Consistent performance observed.",
                        key=f"remarks_{stu_id}"
                    )

                    act_c1, act_c2, _ = st.columns([1, 1, 2])
                    with act_c1:
                        if st.button("✅ Approve Submission", key=f"btn_app_{stu_id}", type="primary", use_container_width=True):
                            ok, msg = teacher_review_submission(
                                student_id=stu_id,
                                action="Approved",
                                remarks=r_remarks,
                                predicted_category=model_pred
                            )
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with act_c2:
                        if st.button("❌ Reject Submission", key=f"btn_rej_{stu_id}", use_container_width=True):
                            ok, msg = teacher_review_submission(
                                student_id=stu_id,
                                action="Rejected",
                                remarks=r_remarks if r_remarks else "Submitted metrics inconsistent with records. Please revise."
                            )
                            if ok:
                                st.warning(msg)
                                st.rerun()
                            else:
                                st.error(msg)

    # SUB-TAB 2: APPROVED SUBMISSIONS
    with rev_tab2:
        st.subheader("Approved Student Records")
        st.dataframe(
            approved_df[["Student_ID", "Name", "Attendance_Percentage", "Study_Hours_Per_Day", "Assignment_Marks", "Exam_Marks", "Performance_Category", "Teacher_Remarks", "Submission_Date"]],
            use_container_width=True
        )

    # SUB-TAB 3: REJECTED SUBMISSIONS
    with rev_tab3:
        st.subheader("Rejected Submissions Awaiting Student Revision")
        if rejected_df.empty:
            st.info("No rejected records.")
        else:
            st.dataframe(
                rejected_df[["Student_ID", "Name", "Attendance_Percentage", "Study_Hours_Per_Day", "Assignment_Marks", "Exam_Marks", "Teacher_Remarks", "Submission_Date"]],
                use_container_width=True
            )

# ---------------------------------------------------------
# TEACHER PAGE 3: Add Student & Predict Performance
# ---------------------------------------------------------
elif page == "➕ Add Student & Predict":
    st.markdown('<div class="main-title">➕ Add Student Details & Predict Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter academic parameters to evaluate student standing using the Decision Tree model.</div>', unsafe_allow_html=True)

    df_records = load_student_records()
    existing_ids = df_records["Student_ID"].tolist() if not df_records.empty else []

    with st.form("student_input_form", clear_on_submit=False):
        st.markdown("##### 📝 Student Information")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            student_id = st.text_input("Student ID (e.g., STU1050)", placeholder="STU1050").strip().upper()
        with f_col2:
            student_name = st.text_input("Student Full Name", placeholder="e.g., Rahul Sharma").strip()

        st.markdown("##### 📊 Academic Performance Indicators")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            attendance = st.number_input(
                "Attendance Percentage (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=0.5,
                help="Classroom attendance percentage from 0% to 100%"
            )
            study_hours = st.number_input(
                "Daily Study Hours (hrs/day)",
                min_value=0.0,
                max_value=16.0,
                value=3.0,
                step=0.25,
                help="Average self-study hours outside regular college lecture hours"
            )

        with m_col2:
            assignment_marks = st.number_input(
                "Assignment Marks (/100)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
                help="Internal continuous evaluation / assignment marks out of 100"
            )
            exam_marks = st.number_input(
                "Exam Marks (/100)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
                help="End semester or mid-term written exam marks out of 100"
            )

        save_to_csv = st.checkbox("Save this record to database (CSV file)", value=True)
        submitted = st.form_submit_button("🔮 Predict Performance & Generate Report", use_container_width=True)

    if submitted:
        # Validate inputs
        is_valid, validation_errors = validate_student_inputs(
            student_id=student_id,
            name=student_name,
            attendance=attendance,
            study_hours=study_hours,
            assignment_marks=assignment_marks,
            exam_marks=exam_marks,
            existing_ids=existing_ids if save_to_csv else None
        )

        if not is_valid:
            for err in validation_errors:
                st.error(f"❌ {err}")
        elif model_bundle is None:
            st.error("Model is not loaded. Please train the model first.")
        else:
            model = model_bundle["model"]
            feature_names = model_bundle["feature_names"]

            # Format input row for model inference
            input_df = pd.DataFrame([{
                "Attendance_Percentage": attendance,
                "Study_Hours_Per_Day": study_hours,
                "Assignment_Marks": assignment_marks,
                "Exam_Marks": exam_marks
            }])[feature_names]

            # Predict
            pred_category = model.predict(input_df)[0]
            pred_probs = model.predict_proba(input_df)[0]
            classes = model.classes_

            # Save if requested
            if save_to_csv:
                new_record = {
                    "Student_ID": student_id,
                    "Name": student_name,
                    "Attendance_Percentage": attendance,
                    "Study_Hours_Per_Day": study_hours,
                    "Assignment_Marks": assignment_marks,
                    "Exam_Marks": exam_marks,
                    "Performance_Category": pred_category,
                    "Submission_Status": "Approved",
                    "Teacher_Remarks": "Added directly by Faculty",
                    "Submission_Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                }
                success, msg = save_student_record(new_record)
                if success:
                    st.success(f"✅ Record saved for {student_name} ({student_id})!")
                else:
                    st.warning(f"Note: {msg}")

            # Display Prediction Card
            st.markdown("---")
            st.subheader("🎯 Prediction Result & Performance Assessment")

            res_col1, res_col2 = st.columns([1.2, 1.8])

            with res_col1:
                st.markdown(f"**Student ID:** `{student_id}`")
                st.markdown(f"**Name:** **{student_name}**")
                st.markdown(f"**Predicted Category:** {get_badge_html(pred_category)}", unsafe_allow_html=True)

                st.markdown("<br>**Class Probabilities:**", unsafe_allow_html=True)
                for cls_name, prob in zip(classes, pred_probs):
                    st.write(f"• **{cls_name}:** {prob * 100:.1f}%")
                    st.progress(float(prob))

            with res_col2:
                st.markdown("##### 💡 Actionable Recommendations")
                recommendations = generate_recommendations(
                    attendance=attendance,
                    study_hours=study_hours,
                    assignment_marks=assignment_marks,
                    exam_marks=exam_marks,
                    prediction=pred_category
                )

                for rec in recommendations:
                    if rec["type"] == "warning":
                        st.warning(f"**{rec['area']}:** {rec['message']}")
                    elif rec["type"] == "info":
                        st.info(f"**{rec['area']}:** {rec['message']}")
                    elif rec["type"] == "action":
                        st.success(f"**{rec['area']}:** {rec['message']}")
                    else:
                        st.caption(f"**{rec['area']}:** {rec['message']}")

# ---------------------------------------------------------
# TEACHER PAGE 4: Student Records & Reports
# ---------------------------------------------------------
elif page == "📋 Student Records & Reports":
    st.markdown('<div class="main-title">📋 Student Records & Individual Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">View all student profiles, filter records, download CSV data, and generate printable scorecards.</div>', unsafe_allow_html=True)

    df_records = load_student_records()

    if df_records.empty:
        st.warning("No records currently available. Add students in the 'Add Student & Predict' section.")
    else:
        tab1, tab2 = st.tabs(["📑 All Student Records", "🪪 Individual Student Scorecard"])

        with tab1:
            st.subheader("Search & Filter Records")
            f1, f2, f3 = st.columns([2, 1, 1])
            with f1:
                search_query = st.text_input("🔍 Search by Student Name or ID:", placeholder="Enter name or ID...").strip()
            with f2:
                selected_cat = st.selectbox(
                    "Filter by Category:",
                    ["All Categories"] + sorted(list(df_records["Performance_Category"].dropna().unique()))
                )
            with f3:
                selected_stat = st.selectbox(
                    "Filter by Status:",
                    ["All Statuses"] + sorted(list(df_records["Submission_Status"].dropna().unique()))
                )

            filtered_df = df_records.copy()
            if search_query:
                filtered_df = filtered_df[
                    filtered_df["Name"].str.contains(search_query, case=False, na=False) |
                    filtered_df["Student_ID"].str.contains(search_query, case=False, na=False)
                ]
            if selected_cat != "All Categories":
                filtered_df = filtered_df[filtered_df["Performance_Category"] == selected_cat]
            if selected_stat != "All Statuses":
                filtered_df = filtered_df[filtered_df["Submission_Status"] == selected_stat]

            st.write(f"Showing **{len(filtered_df)}** of **{len(df_records)}** total records.")
            st.dataframe(filtered_df, use_container_width=True)

            # CSV Download Button
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Records as CSV",
                data=csv_buffer.getvalue(),
                file_name="student_performance_records.csv",
                mime="text/csv",
                use_container_width=False
            )

        with tab2:
            st.subheader("Generate Student Performance Scorecard")
            student_options = [
                f"{row['Student_ID']} - {row['Name']}"
                for _, row in df_records.iterrows()
            ]

            selected_option = st.selectbox("Select Student Profile:", student_options)

            if selected_option:
                sel_id = selected_option.split(" - ")[0].strip()
                student_data = df_records[df_records["Student_ID"] == sel_id].iloc[0]

                st.markdown("""<div class="report-box">""", unsafe_allow_html=True)
                h_col1, h_col2 = st.columns([2, 1])
                with h_col1:
                    st.markdown(f"### 🎓 Student Performance Report: {student_data['Name']}")
                    st.markdown(f"**Student ID:** `{student_data['Student_ID']}` &nbsp;|&nbsp; **Status:** {get_status_badge_html(student_data['Submission_Status'])}", unsafe_allow_html=True)
                    if student_data.get("Teacher_Remarks"):
                        st.markdown(f"**Faculty Remarks:** *{student_data['Teacher_Remarks']}*")
                with h_col2:
                    st.markdown(f"**Overall Classification:**")
                    st.markdown(get_badge_html(student_data['Performance_Category']), unsafe_allow_html=True)

                st.markdown("---")

                # Metrics Comparison with Class Average
                m1, m2, m3, m4 = st.columns(4)
                class_att_avg = df_records["Attendance_Percentage"].mean()
                class_study_avg = df_records["Study_Hours_Per_Day"].mean()
                class_assign_avg = df_records["Assignment_Marks"].mean()
                class_exam_avg = df_records["Exam_Marks"].mean()

                with m1:
                    st.metric(
                        "Attendance",
                        f"{student_data['Attendance_Percentage']:.1f}%",
                        delta=f"{student_data['Attendance_Percentage'] - class_att_avg:+.1f}% vs Class Avg"
                    )
                with m2:
                    st.metric(
                        "Study Hours",
                        f"{student_data['Study_Hours_Per_Day']:.1f} hrs",
                        delta=f"{student_data['Study_Hours_Per_Day'] - class_study_avg:+.1f} hrs vs Class Avg"
                    )
                with m3:
                    st.metric(
                        "Assignment Marks",
                        f"{student_data['Assignment_Marks']:.1f}/100",
                        delta=f"{student_data['Assignment_Marks'] - class_assign_avg:+.1f} vs Class Avg"
                    )
                with m4:
                    st.metric(
                        "Exam Marks",
                        f"{student_data['Exam_Marks']:.1f}/100",
                        delta=f"{student_data['Exam_Marks'] - class_exam_avg:+.1f} vs Class Avg"
                    )

                # Benchmark Bar Chart
                st.markdown("##### 📊 Individual vs Class Benchmark")
                fig_bench, ax_bench = plt.subplots(figsize=(7, 3.2))
                apply_dark_plot_style(fig_bench, ax_bench)

                categories_bench = ["Attendance (%)", "Study Hours (x10)", "Assignments (/100)", "Exam Marks (/100)"]
                student_values = [
                    student_data["Attendance_Percentage"],
                    student_data["Study_Hours_Per_Day"] * 10,
                    student_data["Assignment_Marks"],
                    student_data["Exam_Marks"]
                ]
                class_values = [
                    class_att_avg,
                    class_study_avg * 10,
                    class_assign_avg,
                    class_exam_avg
                ]

                ind = np.arange(len(categories_bench))
                width = 0.35
                ax_bench.bar(ind - width/2, student_values, width, label=f"{student_data['Name']}", color="#3B82F6")
                ax_bench.bar(ind + width/2, class_values, width, label="Class Average", color="#64748B")
                ax_bench.set_xticks(ind)
                ax_bench.set_xticklabels(categories_bench, fontsize=9)
                ax_bench.set_ylabel("Score / Normalized Scale", fontsize=9)
                ax_bench.legend(fontsize=8, facecolor="#0F172A", edgecolor="#334155")
                ax_bench.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
                plt.tight_layout()
                st.pyplot(fig_bench)
                plt.close()

                # Actionable Roadmap
                st.markdown("##### 💡 Tailored Academic Guidance")
                recs = generate_recommendations(
                    attendance=student_data["Attendance_Percentage"],
                    study_hours=student_data["Study_Hours_Per_Day"],
                    assignment_marks=student_data["Assignment_Marks"],
                    exam_marks=student_data["Exam_Marks"],
                    prediction=student_data["Performance_Category"]
                )
                for r in recs:
                    st.markdown(f"• **{r['area']}:** {r['message']}")

                st.markdown("</div>", unsafe_allow_html=True)

                # Download individual report text
                report_txt = f"""====================================================
STUDENT PERFORMANCE EVALUATION REPORT
====================================================
Student ID:     {student_data['Student_ID']}
Student Name:   {student_data['Name']}
Classification: {student_data['Performance_Category']}
Status:         {student_data['Submission_Status']}
Remarks:        {student_data.get('Teacher_Remarks', '')}
----------------------------------------------------
Attendance:        {student_data['Attendance_Percentage']:.1f}% (Class Avg: {class_att_avg:.1f}%)
Study Hours/Day:   {student_data['Study_Hours_Per_Day']:.1f} hrs (Class Avg: {class_study_avg:.1f} hrs)
Assignment Marks:  {student_data['Assignment_Marks']:.1f}/100 (Class Avg: {class_assign_avg:.1f}/100)
Exam Marks:        {student_data['Exam_Marks']:.1f}/100 (Class Avg: {class_exam_avg:.1f}/100)
----------------------------------------------------
ACTIONABLE RECOMMENDATIONS:
"""
                for r in recs:
                    report_txt += f"- [{r['area']}] {r['message']}\n"
                report_txt += "====================================================\n"

                st.download_button(
                    label="📄 Download Individual Report (.txt)",
                    data=report_txt,
                    file_name=f"Report_{student_data['Student_ID']}.txt",
                    mime="text/plain"
                )

# ---------------------------------------------------------
# TEACHER PAGE 5: Model Evaluation & Insights
# ---------------------------------------------------------
elif page == "🧠 Model Evaluation & Insights":
    st.markdown('<div class="main-title">🧠 Machine Learning Model Evaluation & Limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed inspection of training methodology, performance metrics, and ethical boundaries.</div>', unsafe_allow_html=True)

    if metrics_info is None:
        st.warning("Model metrics not found. Please train the model using `model/train_model.py`.")
    else:
        # Metrics Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Model Algorithm", "Decision Tree")
        with col_m2:
            st.metric("Test Set Accuracy", f"{metrics_info['test_accuracy']}%")
        with col_m3:
            st.metric("Training Samples", f"{metrics_info['num_train_samples']}")
        with col_m4:
            st.metric("Testing Samples", f"{metrics_info['num_test_samples']}")

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)

        # Feature Importance Plot
        with col_g1:
            st.subheader("📊 Feature Importances")
            st.caption("Shows the relative mathematical contribution of each feature in the Decision Tree splits.")
            feat_imp = metrics_info.get("feature_importances", {})
            if feat_imp:
                sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
                labels = [f[0].replace("_", " ") for f in sorted_feats]
                vals = [f[1] for f in sorted_feats]

                fig_imp, ax_imp = plt.subplots(figsize=(6, 3.8))
                apply_dark_plot_style(fig_imp, ax_imp)
                ax_imp.barh(labels[::-1], vals[::-1], color="#38BDF8", edgecolor="#1E293B")
                ax_imp.set_xlabel("Gini Importance Score", fontsize=10)
                ax_imp.grid(axis="x", linestyle="--", alpha=0.25, color="#475569")
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close()

        # Confusion Matrix
        with col_g2:
            st.subheader("🎯 Confusion Matrix (Test Data)")
            st.caption("Rows represent Ground Truth categories; Columns represent Model Predictions.")
            cm = np.array(metrics_info.get("confusion_matrix", []))
            class_names = metrics_info.get("class_names", [])

            if len(cm) > 0:
                fig_cm, ax_cm = plt.subplots(figsize=(5.5, 3.8))
                apply_dark_plot_style(fig_cm, ax_cm)
                cax = ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax_cm.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='large', weight='bold', color="#FFFFFF")

                cbar = fig_cm.colorbar(cax)
                cbar.ax.yaxis.set_tick_params(color='#CBD5E1')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#CBD5E1')

                ax_cm.set_xticks(range(len(class_names)))
                ax_cm.set_yticks(range(len(class_names)))
                ax_cm.set_xticklabels(class_names, rotation=25, fontsize=8)
                ax_cm.set_yticklabels(class_names, fontsize=8)
                ax_cm.set_xlabel("Predicted Label", fontsize=10)
                ax_cm.set_ylabel("True Label", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

        # Classification Report Table
        st.subheader("📑 Precision, Recall & F1-Score Breakdown")
        rep_dict = metrics_info.get("classification_report", {})
        if rep_dict:
            rep_rows = []
            for k, v in rep_dict.items():
                if isinstance(v, dict):
                    rep_rows.append({
                        "Class / Metric": k,
                        "Precision": round(v.get("precision", 0), 3),
                        "Recall": round(v.get("recall", 0), 3),
                        "F1-Score": round(v.get("f1-score", 0), 3),
                        "Support": v.get("support", 0)
                    })
            df_rep = pd.DataFrame(rep_rows)
            st.dataframe(df_rep, use_container_width=True)

        st.markdown("---")

        # Academic Limitations & Ethical Disclaimer
        st.subheader("⚠️ Critical Model Limitations & Data Leakage Prevention")
        st.markdown("""
1. **No Data Leakage Guarantee:**
   - The dataset is partitioned into an **80% Training Set** and a strictly isolated **20% Testing Set** using `sklearn.model_selection.train_test_split`.
   - The Decision Tree classifier is trained **exclusively** on the training set. Evaluation metrics (accuracy, confusion matrix, precision/recall) are computed on the unseen test set.

2. **Synthetic Dataset Caveat:**
   - The model is trained on a synthetic dataset formulated to simulate realistic academic distributions.
   - While it captures realistic correlations (e.g., attendance and daily study hours influencing exam outcomes), real-world educational data exhibits greater unpredictability, institutional differences, and socio-economic variance.

3. **Algorithm Boundary Characteristics:**
   - Decision Trees create axis-parallel decision boundaries. While highly transparent and explainable, they can exhibit sharp step-changes rather than smooth probability curves.
   - We applied regularization hyperparameters (`max_depth=4`, `min_samples_split=10`, `min_samples_leaf=5`) to prevent overfitting and avoid memorization of noise.

4. **External Latent Factors (Human & Mental Wellbeing):**
   - Academic success is multifaceted and cannot be fully captured by numerical metrics alone.
   - Factors such as student physical health, stress levels, quality of sleep, family circumstances, and learning disabilities are not represented in this quantitative feature set.
   - **Recommendation:** ML predictions in education should serve solely as an early-alert supportive diagnostic for educators, never as a punitive or definitive judgment.
""")
