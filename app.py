"""
app.py
------
Streamlit Web Application for the Student Performance Report System.

Features:
1. Authentication & Role-Based Access Control:
   - Separate Teacher & Student roles.
   - Secure PBKDF2 password hashing & verification.
   - Student self-registration with unique roll number and email validation.
   - Teacher registration requiring existing faculty approval.
   - Pre-seeded default faculty login.
2. Soft Pastel Theme & Modern Design System:
   - Palette: #F7F5FF (Background), #FFFFFF (Cards), #E9E2FF (Lavender),
     #E2F0FF (Light Blue), #DDF7EA (Mint), #FFE5EF (Soft Pink),
     #25233A (Text), #77758A (Secondary Text), #E9E6F2 (Borders), #7663C7 (Primary).
   - Consistent typography hierarchy (Inter font with clean sans-serif fallbacks).
   - Fixed Top Header with contextual role badges and secure Logout.
   - Responsive Slide-out Left Navigation Drawer with overlay.
3. Teacher Interface:
   - 📊 Dashboard & Analytics: Class KPI cards, 4 pastel analytics charts, pending submission alert.
   - 📬 Review Submissions: Inspect student metrics, ML Decision Tree preview, approve/reject with remarks.
   - ➕ Add Student & Predict: Teacher manual entry, instant ML inference, recommendations, CSV save.
   - 📋 Student Records & Reports: Real-time search, category & status filters, CSV export, scorecards.
   - 🧠 Model Evaluation & Insights: Accuracy, confusion matrix, Gini importances, ethics & boundaries.
4. Student Interface:
   - 🏠 Student Dashboard: Personalized greeting, status badge, 4 academic KPI cards, benchmark chart,
     academic completion ring, top recommendations, persistent private learning goals.
   - 👤 My Profile: View read-only identity and update permitted details (synced across databases).
   - 📝 Submit Academic Details: Attendance, study hours, coursework, exam marks, revision resubmission.
   - 📑 My Performance Report: Verified scorecard, class comparison metrics, bar chart, text export.
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from datetime import datetime

# Resolve project directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "student_performance_system"
DATA_DIR = PROJECT_DIR / "data" if (PROJECT_DIR / "data").exists() else BASE_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model" if (PROJECT_DIR / "model").exists() else BASE_DIR / "model"

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
    update_teacher_profile,
    validate_student_registration,
    validate_account_registration,
    register_account,
    get_pending_teacher_accounts,
    set_teacher_approval,
    validate_academic_inputs,
    submit_student_academic_details,
    get_student_submission,
    teacher_review_submission,
    get_student_goals,
    add_student_goal,
    update_student_goal_status
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Report System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Soft Pastel Design System & Component Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Design System Tokens */
    :root {
        --bg-main: #F7F5FF;
        --card-bg: #FFFFFF;
        --lavender-accent: #E9E2FF;
        --blue-accent: #E2F0FF;
        --mint-accent: #DDF7EA;
        --pink-accent: #FFE5EF;
        --text-main: #25233A;
        --text-secondary: #77758A;
        --border-color: #E9E6F2;
        --primary-btn: #7663C7;
        --primary-hover: #6350B5;
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Hide Streamlit Native Chrome Elements */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Global Body and Typography */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    .main, [data-testid="stMainBlockContainer"] {
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
        font-family: var(--font-family) !important;
    }

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 10% 0%, rgba(233, 226, 255, 0.4), transparent 30%),
                    radial-gradient(circle at 90% 10%, rgba(226, 240, 255, 0.45), transparent 30%),
                    #F7F5FF !important;
    }

    .block-container {
        max-width: 1360px !important;
        padding-top: 80px !important;
        padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-bottom: 3.5rem !important;
    }

    /* Heading Hierarchy */
    h1, .page-title {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: var(--text-main) !important;
        letter-spacing: -0.02em !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.35rem !important;
        line-height: 1.25 !important;
    }

    .page-subtitle {
        font-size: 14.5px !important;
        color: var(--text-secondary) !important;
        margin-bottom: 1.5rem !important;
        line-height: 1.5 !important;
    }

    h2, .section-heading {
        font-size: 22px !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
        letter-spacing: -0.01em !important;
        margin-top: 1.4rem !important;
        margin-bottom: 0.75rem !important;
    }

    h3, .card-title {
        font-size: 16.5px !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
        margin-bottom: 0.4rem !important;
    }

    p, span, label, li {
        font-size: 14.5px;
        color: var(--text-main);
        line-height: 1.55;
    }

    small, [data-testid="stCaptionContainer"], .caption-text {
        font-size: 12.5px !important;
        color: var(--text-secondary) !important;
    }

    /* Fixed Top Navbar */
    div[class*="st-key-top_navbar_bar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100vw !important;
        height: 62px !important;
        background-color: #FFFFFF !important;
        border-bottom: 1px solid var(--border-color) !important;
        z-index: 9999 !important;
        padding: 8px 24px !important;
        box-shadow: 0 2px 12px rgba(118, 99, 199, 0.05) !important;
        display: flex !important;
        align-items: center !important;
    }

    .navbar-title {
        font-size: 21px !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .navbar-status-badge {
        text-align: right;
        font-size: 13.5px !important;
        color: var(--text-secondary) !important;
    }
    .navbar-status-badge strong {
        color: var(--text-main) !important;
        font-weight: 600;
    }

    .role-badge-teacher {
        background-color: var(--lavender-accent);
        color: #5A45A6;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #D6CAFA;
        font-size: 12px;
        letter-spacing: 0.02em;
    }

    .role-badge-student {
        background-color: var(--mint-accent);
        color: #1E6B4A;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #BCEBD8;
        font-size: 12px;
        letter-spacing: 0.02em;
    }

    /* Hamburger Button */
    .st-key-btn_hamburger button {
        background-color: #FAF9FF !important;
        border: 1px solid var(--border-color) !important;
        color: var(--primary-btn) !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
        padding: 6px 11px !important;
        border-radius: 9px !important;
        cursor: pointer !important;
        transition: all 0.18s ease !important;
    }
    .st-key-btn_hamburger button:hover {
        background-color: var(--lavender-accent) !important;
        border-color: #D6CAFA !important;
        color: var(--primary-btn) !important;
        box-shadow: 0 2px 8px rgba(118, 99, 199, 0.12) !important;
    }

    /* Drawer Overlay */
    .drawer-overlay {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background-color: rgba(37, 35, 58, 0.28) !important;
        backdrop-filter: blur(2px) !important;
        -webkit-backdrop-filter: blur(2px) !important;
        z-index: 10000 !important;
        display: block !important;
    }

    /* Drawer Panel (Left Slide-out) */
    div[class*="st-key-drawer_panel"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 320px !important;
        max-width: 85vw !important;
        height: 100vh !important;
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--border-color) !important;
        z-index: 10001 !important;
        padding: 22px 18px !important;
        box-shadow: 8px 0 26px rgba(118, 99, 199, 0.08) !important;
        overflow-y: auto !important;
        animation: drawerSlide 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    @keyframes drawerSlide {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }

    .drawer-header h3 {
        margin: 0 !important;
        color: var(--text-main) !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    .drawer-divider {
        height: 1px;
        background: var(--border-color);
        margin: 12px 0 14px 0;
    }
    .st-key-btn_close_drawer button {
        background-color: transparent !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-secondary) !important;
        font-size: 1rem !important;
        padding: 4px 9px !important;
        border-radius: 7px !important;
        transition: all 0.15s ease !important;
    }
    .st-key-btn_close_drawer button:hover {
        background-color: var(--pink-accent) !important;
        border-color: #F7C2D4 !important;
        color: #9E254C !important;
    }

    /* Drawer Nav Buttons */
    div[class*="st-key-drawer_panel"] button[kind="secondary"] {
        background-color: #FAF9FF !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-main) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 9px 13px !important;
        border-radius: 10px !important;
        margin-bottom: 7px !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[class*="st-key-drawer_panel"] button[kind="secondary"]:hover {
        background-color: var(--lavender-accent) !important;
        border-color: #D6CAFA !important;
        color: #5A45A6 !important;
        transform: translateX(3px) !important;
    }

    div[class*="st-key-drawer_panel"] button[kind="primary"] {
        background-color: var(--primary-btn) !important;
        border: 1px solid var(--primary-btn) !important;
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 9px 13px !important;
        border-radius: 10px !important;
        margin-bottom: 7px !important;
        box-shadow: 0 3px 10px rgba(118, 99, 199, 0.22) !important;
    }

    /* Card Panels & Containers */
    .dashboard-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(118, 99, 199, 0.04);
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 4px 16px rgba(118, 99, 199, 0.04);
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card.accent-lavender { border-top: 4px solid #B8A0F5; }
    .kpi-card.accent-blue { border-top: 4px solid #82B8EF; }
    .kpi-card.accent-mint { border-top: 4px solid #66C79E; }
    .kpi-card.accent-pink { border-top: 4px solid #F08CAE; }

    .kpi-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 700;
        color: var(--text-main);
        line-height: 1.15;
    }
    .kpi-subtext {
        font-size: 12.5px;
        color: var(--text-secondary);
        margin-top: 6px;
    }

    /* Badges */
    .badge-good, .status-badge-approved {
        display: inline-block;
        background-color: var(--mint-accent);
        color: #1E6B4A;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13.5px;
        border: 1px solid #B8ECCF;
    }
    .badge-average, .status-badge-pending {
        display: inline-block;
        background-color: #FFF3D6;
        color: #8A5A0A;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13.5px;
        border: 1px solid #F2D88C;
    }
    .badge-needs-improvement, .status-badge-rejected {
        display: inline-block;
        background-color: var(--pink-accent);
        color: #9E254C;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13.5px;
        border: 1px solid #F7C2D4;
    }

    /* Welcome Banners */
    .student-welcome-banner {
        background: linear-gradient(135deg, #F4F0FF 0%, #EAF4FF 100%) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 18px !important;
        padding: 22px 26px !important;
        margin: 6px 0 20px 0 !important;
        box-shadow: 0 4px 18px rgba(118, 99, 199, 0.05) !important;
    }
    .student-welcome-banner h2 {
        color: var(--text-main) !important;
        margin: 0 0 4px 0 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    .student-welcome-banner p {
        color: var(--text-secondary) !important;
        margin: 0 !important;
        font-size: 14px !important;
    }

    /* Auth Card & Login Layout */
    .block-container:has(div[class*="st-key-auth_card"]) {
        padding-top: clamp(14px, 2.5vh, 28px) !important;
        padding-bottom: clamp(16px, 3vh, 32px) !important;
        max-width: 520px !important;
    }

    div[class*="st-key-auth_card"] {
        width: min(100%, 480px) !important;
        max-width: 480px !important;
        margin: 0 auto 16px auto !important;
        padding: 24px 26px !important;
        background: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(118, 99, 199, 0.06) !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stForm"] {
        padding: 0 !important;
        border: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    .auth-hero {
        text-align: center;
        max-width: 480px;
        margin: 0 auto 14px auto;
        padding: 0 16px;
    }
    .auth-brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: #FFFFFF;
        font-size: 24px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 14px rgba(118, 99, 199, 0.07);
        margin-bottom: 8px;
    }
    .auth-hero h1, .auth-app-title {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
        letter-spacing: -0.02em !important;
        line-height: 1.25 !important;
        margin: 0 0 5px 0 !important;
    }
    .auth-hero p, .auth-app-subtitle {
        font-size: 14.5px !important;
        color: var(--text-secondary) !important;
        margin: 0 !important;
        line-height: 1.45 !important;
    }

    .auth-card-title {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
        letter-spacing: -0.02em !important;
        margin: 0 0 5px 0 !important;
        line-height: 1.25 !important;
    }
    .auth-card-subtitle {
        font-size: 14.5px !important;
        color: var(--text-secondary) !important;
        margin: 0 0 16px 0 !important;
        line-height: 1.45 !important;
    }

    /* Auth Inputs & Focus State */
    div[class*="st-key-auth_card"] [data-testid="stTextInput"] {
        margin-bottom: 12px !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stTextInput"] div[data-baseweb="base-input"] {
        background: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        min-height: 42px !important;
        height: 42px !important;
        box-sizing: border-box !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {
        border-color: var(--primary-btn) !important;
        box-shadow: 0 0 0 2px rgba(118, 99, 199, 0.22) !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stTextInput"] input {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        height: 100% !important;
        font-size: 14px !important;
        color: var(--text-main) !important;
        padding: 0 12px !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stTextInput"] input:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stTextInput"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--text-secondary) !important;
        padding: 0 10px !important;
        cursor: pointer !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stTextInput"] button:hover {
        color: var(--primary-btn) !important;
        background: transparent !important;
        border: none !important;
    }

    /* Auth Primary Sign In Button */
    div[class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] > button,
    div[class*="st-key-auth_card"] button[kind="primary"] {
        background-color: #7663C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #7663C7 !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        height: 44px !important;
        cursor: pointer !important;
        box-shadow: 0 3px 10px rgba(118, 99, 199, 0.25) !important;
        transition: all 0.18s ease-in-out !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] > button p,
    div[class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] > button span,
    div[class*="st-key-auth_card"] button[kind="primary"] p,
    div[class*="st-key-auth_card"] button[kind="primary"] span {
        color: #FFFFFF !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] > button:hover,
    div[class*="st-key-auth_card"] button[kind="primary"]:hover {
        background-color: #6350B5 !important;
        border-color: #6350B5 !important;
        color: #FFFFFF !important;
        box-shadow: 0 5px 16px rgba(118, 99, 199, 0.35) !important;
        transform: translateY(-1px) !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] > button:focus,
    div[class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] > button:focus-visible,
    div[class*="st-key-auth_card"] button[kind="primary"]:focus,
    div[class*="st-key-auth_card"] button[kind="primary"]:focus-visible {
        outline: 2px solid #B8A0F5 !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 3px rgba(118, 99, 199, 0.25) !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] > button:active,
    div[class*="st-key-auth_card"] button[kind="primary"]:active {
        background-color: #5543A3 !important;
        transform: translateY(0) !important;
        box-shadow: 0 2px 6px rgba(118, 99, 199, 0.2) !important;
    }

    /* Auth Secondary Buttons (Create Account / Back to Sign In) */
    .st-key-open_registration > button,
    .st-key-back_to_login > button,
    div[class*="st-key-auth_card"] .stButton > button:not([kind="primary"]) {
        background-color: #FAF9FF !important;
        color: #5A45A6 !important;
        border: 1px solid #D6CAFA !important;
        border-radius: 10px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        height: 40px !important;
        box-shadow: none !important;
        transition: all 0.18s ease-in-out !important;
    }
    .st-key-open_registration > button p,
    .st-key-open_registration > button span,
    .st-key-back_to_login > button p,
    .st-key-back_to_login > button span,
    div[class*="st-key-auth_card"] .stButton > button:not([kind="primary"]) p,
    div[class*="st-key-auth_card"] .stButton > button:not([kind="primary"]) span {
        color: #5A45A6 !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    .st-key-open_registration > button:hover,
    .st-key-back_to_login > button:hover,
    div[class*="st-key-auth_card"] .stButton > button:not([kind="primary"]):hover {
        background-color: var(--lavender-accent) !important;
        border-color: #B8A0F5 !important;
        color: #49328B !important;
        box-shadow: 0 2px 8px rgba(118, 99, 199, 0.1) !important;
        transform: translateY(-1px) !important;
    }
    .st-key-open_registration > button:focus,
    .st-key-back_to_login > button:focus {
        outline: 2px solid #B8A0F5 !important;
        outline-offset: 2px !important;
    }

    /* Auth Radio & Expander inside card */
    div[class*="st-key-auth_card"] [data-testid="stRadio"] {
        margin-bottom: 12px !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 16px !important;
        padding: 2px 0 !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stRadio"] label {
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: var(--text-main) !important;
        cursor: pointer !important;
    }
    div[class*="st-key-auth_card"] [data-testid="stExpander"] {
        background: #FAF9FF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        margin-top: 14px !important;
        margin-bottom: 12px !important;
    }
    div[class*="st-key-auth_card"] hr {
        margin: 16px 0 !important;
        border-color: var(--border-color) !important;
        opacity: 0.7 !important;
    }

    /* Remarks and Callout Box */
    .remarks-box {
        background-color: #FAF9FF;
        border-left: 4px solid var(--primary-btn);
        padding: 14px 18px;
        border-radius: 8px;
        margin: 12px 0;
        color: var(--text-main);
        font-size: 14px;
        border-top: 1px solid var(--border-color);
        border-right: 1px solid var(--border-color);
        border-bottom: 1px solid var(--border-color);
    }

    /* Buttons */
    .stButton > button, [data-testid="stFormSubmitButton"] > button, .stDownloadButton > button {
        background-color: #FAF9FF !important;
        color: #5A45A6 !important;
        border: 1px solid #D6CAFA !important;
        border-radius: 11px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover, .stDownloadButton > button:hover {
        background-color: var(--lavender-accent) !important;
        border-color: #B8A0F5 !important;
        color: #49328B !important;
        box-shadow: 0 3px 12px rgba(118, 99, 199, 0.12) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background-color: var(--primary-btn) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--primary-btn) !important;
        font-weight: 600 !important;
        box-shadow: 0 3px 12px rgba(118, 99, 199, 0.22) !important;
    }
    .stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
        background-color: var(--primary-hover) !important;
        border-color: var(--primary-hover) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(118, 99, 199, 0.3) !important;
    }

    /* Streamlit Forms & Input Controls */
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border-color) !important;
        border-radius: 16px !important;
    }
    [data-testid="stForm"] {
        background: #FFFFFF !important;
        padding: 22px !important;
        box-shadow: 0 4px 18px rgba(118, 99, 199, 0.04) !important;
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stPasswordInput"] input, [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stMultiSelect"] div[data-baseweb="select"] {
        background: #FFFFFF !important;
        color: var(--text-main) !important;
        border-color: var(--border-color) !important;
        border-radius: 10px !important;
        font-size: 14px !important;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus,
    [data-testid="stPasswordInput"] input:focus {
        border-color: var(--primary-btn) !important;
        box-shadow: 0 0 0 1px var(--primary-btn) !important;
    }
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: var(--text-main) !important;
    }

    /* Tabs, Metrics, DataTables */
    [data-testid="stTabs"] button {
        color: var(--text-secondary) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--primary-btn) !important;
        border-bottom-color: var(--primary-btn) !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        padding: 14px 16px !important;
        box-shadow: 0 3px 12px rgba(118, 99, 199, 0.04) !important;
    }
    [data-testid="stMetricLabel"] { font-size: 13px !important; color: var(--text-secondary) !important; }
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; color: var(--text-main) !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid var(--border-color) !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(118, 99, 199, 0.03) !important;
    }
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        font-size: 14px !important;
    }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #B8A0F5, #82B8EF) !important;
        border-radius: 999px !important;
    }
    [data-testid="stProgressBar"] > div {
        background: #EFEAF8 !important;
        border-radius: 999px !important;
    }

    /* Learning Goal Card */
    div[class*="st-key-goal_card_"] {
        background: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        margin: 8px 0 !important;
        box-shadow: 0 2px 10px rgba(118, 99, 199, 0.04) !important;
    }

    /* Circular Progress Display */
    .progress-ring-container {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 14px 18px;
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        margin: 12px 0 18px 0;
        box-shadow: 0 3px 14px rgba(118, 99, 199, 0.04);
    }
    .progress-ring-circle {
        --progress: 0;
        width: 104px;
        height: 104px;
        flex: 0 0 104px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: conic-gradient(var(--primary-btn) calc(var(--progress) * 1%), #EFEBF8 0);
        position: relative;
    }
    .progress-ring-circle::before {
        content: "";
        position: absolute;
        inset: 9px;
        border-radius: 50%;
        background: #FFFFFF;
        border: 1px solid #F3EFF9;
    }
    .progress-ring-circle span {
        position: relative;
        z-index: 1;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--primary-btn);
    }
    .progress-ring-text strong {
        display: block;
        color: var(--text-main);
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .progress-ring-text span {
        color: var(--text-secondary);
        font-size: 13.5px;
    }

    @media (max-width: 768px) {
        .block-container { padding-top: 74px !important; padding-left: 1rem !important; padding-right: 1rem !important; }
        div[class*="st-key-top_navbar_bar"] { padding: 6px 12px !important; }
        .navbar-title { font-size: 17px !important; }
        .navbar-status-badge { display: none !important; }
        div[class*="st-key-auth_card"] { padding: 20px 16px !important; margin: 12px auto !important; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Path Resolvers & Cache Handlers
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = str(MODEL_DIR / "student_model.pkl")
METRICS_PATH = str(MODEL_DIR / "model_metrics.json")
TRAIN_DATA_PATH = str(DATA_DIR / "student_data.csv")

@st.cache_resource
def load_trained_model():
    """Loads the serialized Decision Tree model bundle."""
    if not os.path.exists(MODEL_PATH):
        try:
            import runpy
            trainer_path = MODEL_DIR / "train_model.py"
            if not trainer_path.exists():
                st.error(f"Model file not found: {MODEL_PATH}. Trainer not found: {trainer_path}")
                return None
            trainer_namespace = runpy.run_path(str(trainer_path))
            trainer_namespace["train_and_evaluate"]()
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
    """Returns styled HTML badge for performance tier matching the soft pastel palette."""
    if category == "Good":
        return '<span class="badge-good">🌟 Good</span>'
    elif category == "Average":
        return '<span class="badge-average">⚡ Average</span>'
    elif category == "Needs Improvement":
        return '<span class="badge-needs-improvement">⚠️ Needs Improvement</span>'
    else:
        return f'<span class="badge-average">{category}</span>'

def get_status_badge_html(status: str) -> str:
    """Returns styled HTML badge for submission review status."""
    if status == "Approved":
        return '<span class="status-badge-approved">✅ Approved</span>'
    elif status == "Rejected":
        return '<span class="status-badge-rejected">❌ Needs Revision</span>'
    elif status == "Pending Review":
        return '<span class="status-badge-pending">⏳ Pending Review</span>'
    else:
        return '<span class="status-badge-pending">ℹ️ Not Submitted</span>'

def render_kpi_card(label: str, value: str, subtext: str = "", accent: str = "lavender") -> str:
    """Renders a soft pastel KPI summary card with consistent typography and borders."""
    return f"""
    <div class="kpi-card accent-{accent}">
        <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        {f'<div class="kpi-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """

def apply_pastel_plot_style(fig, ax):
    """Applies consistent soft pastel styling to Matplotlib figures."""
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.tick_params(colors='#77758A', labelsize=9)
    ax.xaxis.label.set_color('#25233A')
    ax.yaxis.label.set_color('#25233A')
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.title.set_color('#25233A')
    ax.title.set_fontsize(11)
    ax.title.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_color('#E9E6F2')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#F0EDF7')

# Initialize databases & model
initialize_records()
initialize_users()
model_bundle = load_trained_model()
metrics_info = load_metrics()

# ---------------------------------------------------------
# Session State & Navigation Configuration
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

def get_teacher_nav(user=None):
    base_nav = [
        "📊 Dashboard & Analytics",
        "👤 My Profile",
        "📬 Review Submissions",
        "➕ Add Student & Predict",
        "📋 Student Records & Reports",
        "🧠 Model Evaluation & Insights"
    ]
    if user and user.get("is_approver"):
        base_nav.append("👥 Account Approvals")
    return base_nav

TEACHER_NAV = get_teacher_nav()

STUDENT_NAV = [
    "🏠 Student Dashboard",
    "👤 My Profile",
    "📝 Submit Academic Details",
    "📑 My Performance Report"
]


# ---------------------------------------------------------
# Screen 1: Authentication & Account Creation
# ---------------------------------------------------------
if not st.session_state["authenticated"]:
    st.markdown("""<style>
.block-container {
    padding-top: clamp(14px, 2.5vh, 28px) !important;
    padding-bottom: clamp(16px, 3vh, 32px) !important;
    max-width: 520px !important;
}
</style>
<div class="auth-hero">
    <div class="auth-brand-mark">🎓</div>
    <div class="auth-app-title">Student Performance Report System</div>
    <div class="auth-app-subtitle">A data-driven academic monitoring and diagnostic platform.</div>
</div>""", unsafe_allow_html=True)

    if "show_registration" not in st.session_state:
        st.session_state["show_registration"] = False
    if "registration_notice" not in st.session_state:
        st.session_state["registration_notice"] = ""

    if st.session_state["registration_notice"]:
        st.success(st.session_state["registration_notice"])
        st.session_state["registration_notice"] = ""

    with st.container(key="auth_card"):
        if not st.session_state["show_registration"]:
            st.markdown('<div class="auth-card-title">Welcome Back</div>', unsafe_allow_html=True)
            st.markdown('<p class="auth-card-subtitle">Sign in with your registered account credentials to access your dashboard.</p>', unsafe_allow_html=True)

            login_role = st.radio(
                "Select Role:",
                ["👨‍🏫 Teacher / Faculty", "👨‍🎓 Student"],
                horizontal=True,
                key="login_role_choice"
            )
            selected_role = "teacher" if "Teacher" in login_role else "student"

            with st.form("login_form", clear_on_submit=False):
                if selected_role == "teacher":
                    ident_label = "Faculty Email Address or Teacher ID"
                    ident_placeholder = "e.g., teacher@school.edu"
                else:
                    ident_label = "Roll Number or Registered Student Email"
                    ident_placeholder = "e.g., 22CS105 or student@college.edu"

                identifier = st.text_input(ident_label, placeholder=ident_placeholder).strip()
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In", use_container_width=True, type="primary")

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
                        st.session_state["current_page"] = get_teacher_nav(user_data)[0] if user_data["role"] == "teacher" else STUDENT_NAV[0]
                        st.session_state["drawer_open"] = False
                        st.success(f"Welcome back, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            st.markdown("---")
            st.markdown(
                '<div style="text-align: center; margin-bottom: 8px; font-size: 13.5px; color: var(--text-secondary);">'
                'New to the system? <strong style="color: var(--text-main);">Create your account below</strong>'
                '</div>',
                unsafe_allow_html=True
            )
            if st.button("📝 Create New Account", use_container_width=True, key="open_registration"):
                st.session_state["show_registration"] = True
                st.rerun()

        else:
            st.markdown('<div class="auth-card-title">Create New Account</div>', unsafe_allow_html=True)
            st.markdown('<p class="auth-card-subtitle">Register a new student account or submit a faculty access request.</p>', unsafe_allow_html=True)

            account_role_label = st.selectbox("Account Role", ["Student", "Teacher"], key="registration_role")
            account_role = account_role_label.lower()

            with st.form("create_account_form", clear_on_submit=False):
                reg_name = st.text_input("Full Name *", placeholder="e.g., Aryan Sharma")
                reg_email = st.text_input("College Email Address *", placeholder="e.g., aryan@college.edu")

                if account_role == "student":
                    reg_roll = st.text_input("Roll Number / Student ID *", placeholder="e.g., 22CS105")
                    reg_course = st.text_input("Course & Branch *", placeholder="e.g., B.Tech CSE")
                    reg_teacher_id = ""
                    reg_dept = ""
                    reg_desig = ""
                else:
                    reg_roll = ""
                    reg_course = ""
                    reg_teacher_id = st.text_input("Teacher ID / Employee ID *", placeholder="e.g., TCH205")
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        reg_dept = st.text_input("Department *", placeholder="e.g., Computer Science & IT")
                    with t_col2:
                        reg_desig = st.text_input("Designation *", placeholder="e.g., Assistant Professor")

                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    reg_pass = st.text_input("Password (min 6 characters) *", type="password")
                with p_col2:
                    reg_pass_confirm = st.text_input("Confirm Password *", type="password")

                if account_role == "teacher":
                    st.info("ℹ️ Teacher accounts are submitted for approval and will become active once verified and approved by an authorized senior faculty approver.")

                submit_reg = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if submit_reg:
                is_valid, reg_errors = validate_account_registration(
                    name=reg_name,
                    email=reg_email,
                    course_branch=reg_course if account_role == "student" else reg_dept,
                    password=reg_pass,
                    confirm_password=reg_pass_confirm,
                    role=account_role,
                    roll_number=reg_roll if account_role == "student" else reg_teacher_id,
                    department=reg_dept if account_role == "teacher" else "",
                    designation=reg_desig if account_role == "teacher" else "",
                    teacher_id=reg_teacher_id if account_role == "teacher" else ""
                )
                if not is_valid:
                    for err in reg_errors:
                        st.error(f"❌ {err}")
                else:
                    ok, msg = register_account(
                        name=reg_name,
                        email=reg_email,
                        course_branch=reg_course if account_role == "student" else reg_dept,
                        password=reg_pass,
                        role=account_role,
                        roll_number=reg_roll if account_role == "student" else reg_teacher_id,
                        department=reg_dept if account_role == "teacher" else "",
                        designation=reg_desig if account_role == "teacher" else "",
                        teacher_id=reg_teacher_id if account_role == "teacher" else ""
                    )
                    if ok:
                        st.session_state["registration_notice"] = msg
                        st.session_state["show_registration"] = False
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            if st.button("⬅️ Back to Sign In", use_container_width=True, key="back_to_login"):
                st.session_state["show_registration"] = False
                st.rerun()

    st.stop()

# ---------------------------------------------------------
# Authenticated Context & Route Guard
# ---------------------------------------------------------
current_user = st.session_state["user"]
user_role = st.session_state["role"]

valid_pages = get_teacher_nav(current_user) if user_role == "teacher" else STUDENT_NAV
if st.session_state.get("current_page") not in valid_pages:
    st.session_state["current_page"] = valid_pages[0]

current_page = st.session_state["current_page"]

# Handle query parameter drawer close
if "close_drawer" in st.query_params:
    st.session_state["drawer_open"] = False
    try:
        del st.query_params["close_drawer"]
    except Exception:
        pass
    st.rerun()

# ---------------------------------------------------------
# Top Header Navigation Bar
# ---------------------------------------------------------
with st.container(key="top_navbar_bar"):
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([0.05, 0.48, 0.35, 0.12], vertical_alignment="center")
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
            f'{role_label} <strong>{current_user["name"]}</strong> · Active: <strong>{clean_name}</strong>'
            f'</div>',
            unsafe_allow_html=True
        )
    with nav_col4:
        if st.button("Logout", key="btn_logout", help="Sign out of account"):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state["role"] = None
            st.session_state["drawer_open"] = False
            st.rerun()

# ---------------------------------------------------------
# Slide-Out Navigation Drawer
# ---------------------------------------------------------
if st.session_state["drawer_open"]:
    st.markdown('<a href="?close_drawer=1" class="drawer-overlay" target="_self" title="Click anywhere to close"></a>', unsafe_allow_html=True)

    with st.container(key="drawer_panel"):
        d_col_title, d_col_close = st.columns([0.8, 0.2], vertical_alignment="center")
        with d_col_title:
            st.markdown('<div class="drawer-header"><h3>🎓 Navigation</h3></div>', unsafe_allow_html=True)
        with d_col_close:
            if st.button("✕", key="btn_close_drawer", help="Close Menu"):
                st.session_state["drawer_open"] = False
                st.rerun()

        st.markdown('<div class="drawer-divider"></div>', unsafe_allow_html=True)
        st.caption(f"ACCOUNT: {user_role.upper()} · {current_user['name']}")

        for idx, mod in enumerate(valid_pages):
            is_active = (current_page == mod)
            btn_key = f"drawer_btn_{idx}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(mod, key=btn_key, type=btn_type, use_container_width=True):
                st.session_state["current_page"] = mod
                st.session_state["drawer_open"] = False
                st.rerun()

# Route Authorization Guard
page = st.session_state["current_page"]
if user_role == "student" and page not in STUDENT_NAV:
    st.session_state["current_page"] = STUDENT_NAV[0]
    st.warning("⚠️ Access restricted. You do not have permission to view faculty-only pages.")
    st.rerun()

if user_role == "teacher":
    if current_user.get("approval_status", "").lower() != "approved":
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.session_state["role"] = None
        st.error("⚠️ Your teacher account is awaiting approval or is not authorized.")
        st.rerun()
    if page not in valid_pages:
        st.session_state["current_page"] = valid_pages[0]
        st.warning("⚠️ Access restricted. You do not have permission to view this page.")
        st.rerun()

# =========================================================
# 👨‍🎓 STUDENT PAGES
# =========================================================

# ---------------------------------------------------------
# STUDENT PAGE 1: Student Dashboard
# ---------------------------------------------------------
if page == "🏠 Student Dashboard":
    st.markdown(f"""
    <div class="student-welcome-banner">
        <h2>✨ Welcome back, {current_user["name"]}!</h2>
        <p>Enrolled in <strong>{current_user.get('course_branch', 'B.Tech')}</strong> · Roll Number: <strong>{current_user.get('roll_number', '')}</strong> · Email: {current_user.get('email', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    submission = get_student_submission(current_user["roll_number"])
    status = submission.get("Submission_Status", "Not Submitted") if submission else "Not Submitted"

    st.markdown('<div class="section-heading">Academic Submission Status & Evaluation</div>', unsafe_allow_html=True)

    st_col1, st_col2, st_col3 = st.columns([1.5, 1.5, 1])
    with st_col1:
        st.markdown(f"**Verification Status:** {get_status_badge_html(status)}", unsafe_allow_html=True)
    with st_col2:
        sub_date = submission.get("Submission_Date") if submission else None
        st.markdown(f"**Submission Date:** `{sub_date if sub_date else 'Not submitted yet'}`")
    with st_col3:
        if status == "Approved":
            cat = submission.get("Performance_Category", "Average")
            st.markdown(f"**Performance Tier:** {get_badge_html(cat)}", unsafe_allow_html=True)

    # 1. State: Not Submitted
    if status == "Not Submitted":
        st.info("ℹ️ You have not submitted your academic indicators yet. Please submit your classroom attendance, study hours, coursework marks, and exam score for faculty verification.")
        if st.button("📝 Submit Academic Details Now", type="primary"):
            st.session_state["current_page"] = "📝 Submit Academic Details"
            st.rerun()

    # 2. State: Pending Review
    elif status == "Pending Review":
        st.warning("⏳ **Your semester parameters have been submitted and are in the Faculty Review Queue.** Your performance classification and tailored report card will unlock upon faculty approval.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(render_kpi_card("Attendance", f"{submission['Attendance_Percentage']:.1f}%", "Submitted for review", "blue"), unsafe_allow_html=True)
        with c2:
            st.markdown(render_kpi_card("Daily Study", f"{submission['Study_Hours_Per_Day']:.1f} hrs", "Self-study per day", "lavender"), unsafe_allow_html=True)
        with c3:
            st.markdown(render_kpi_card("Assignments", f"{submission['Assignment_Marks']:.1f}/100", "Coursework aggregate", "mint"), unsafe_allow_html=True)
        with c4:
            st.markdown(render_kpi_card("Exam Score", f"{submission['Exam_Marks']:.1f}/100", "Written exam marks", "pink"), unsafe_allow_html=True)

    # 3. State: Rejected / Needs Revision
    elif status == "Rejected":
        st.error("❌ **Your academic submission was reviewed by course faculty and requires revision.**")
        teacher_remarks = submission.get("Teacher_Remarks", "Please review and revise your submitted parameters.")
        st.markdown(f"""
        <div class="remarks-box" style="border-left-color: #9E254C;">
            <strong>👨‍🏫 Faculty Feedback Remarks:</strong><br>
            <span style="color: #9E254C;">{teacher_remarks}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("✏️ Correct & Resubmit Academic Details", type="primary"):
            st.session_state["current_page"] = "📝 Submit Academic Details"
            st.rerun()

    # 4. State: Approved
    elif status == "Approved":
        teacher_remarks = submission.get("Teacher_Remarks", "Approved by Course Faculty.")
        st.markdown(f"""
        <div class="remarks-box">
            <strong>👨‍🏫 Verified by Faculty:</strong> {teacher_remarks}
        </div>
        """, unsafe_allow_html=True)

        df_records = load_student_records()
        approved_records = df_records[df_records["Submission_Status"] == "Approved"]
        class_att_avg = approved_records["Attendance_Percentage"].mean() if not approved_records.empty else 75.0
        class_study_avg = approved_records["Study_Hours_Per_Day"].mean() if not approved_records.empty else 4.0
        class_assign_avg = approved_records["Assignment_Marks"].mean() if not approved_records.empty else 70.0
        class_exam_avg = approved_records["Exam_Marks"].mean() if not approved_records.empty else 65.0

        academic_avg = (float(submission["Assignment_Marks"]) + float(submission["Exam_Marks"])) / 2.0
        category = submission.get("Performance_Category", "Average")

        # 4 Academic KPI Cards
        st.markdown('<div class="section-heading">Verified Academic KPIs</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            att_diff = submission['Attendance_Percentage'] - class_att_avg
            att_diff_txt = f"{att_diff:+.1f}% vs class avg ({class_att_avg:.1f}%)"
            st.markdown(render_kpi_card("Verified Attendance", f"{submission['Attendance_Percentage']:.1f}%", att_diff_txt, "blue"), unsafe_allow_html=True)
        with k2:
            std_diff = submission['Study_Hours_Per_Day'] - class_study_avg
            std_diff_txt = f"{std_diff:+.1f} hrs vs class avg ({class_study_avg:.1f} hrs)"
            st.markdown(render_kpi_card("Daily Study Hours", f"{submission['Study_Hours_Per_Day']:.1f} hrs", std_diff_txt, "lavender"), unsafe_allow_html=True)
        with k3:
            st.markdown(render_kpi_card("Academic Average", f"{academic_avg:.1f}/100", "Coursework & exam combined", "mint"), unsafe_allow_html=True)
        with k4:
            st.markdown(render_kpi_card("Performance Tier", f"{category}", "Decision Tree classification", "pink"), unsafe_allow_html=True)

        # Progress Ring & Interactive Visualizations
        st.markdown('<div class="section-heading">Academic Completion & Class Benchmarks</div>', unsafe_allow_html=True)
        c_vis1, c_vis2 = st.columns([1, 1.6])

        with c_vis1:
            ring_progress = min(100.0, max(0.0, academic_avg))
            st.markdown(
                f"""
                <div class="progress-ring-container">
                    <div class="progress-ring-circle" style="--progress: {ring_progress:.1f};" role="img" aria-label="Academic average {academic_avg:.1f} out of 100">
                        <span>{academic_avg:.0f}%</span>
                    </div>
                    <div class="progress-ring-text">
                        <strong>Verified Academic Score</strong>
                        <span>Combined aggregate from your verified coursework and exam results.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Recommendations teaser
            recs = generate_recommendations(
                attendance=submission["Attendance_Percentage"],
                study_hours=submission["Study_Hours_Per_Day"],
                assignment_marks=submission["Assignment_Marks"],
                exam_marks=submission["Exam_Marks"],
                prediction=category
            )
            st.markdown('<div class="card-title" style="margin-top: 14px;">💡 Top Faculty Recommendations</div>', unsafe_allow_html=True)
            for r in recs[:2]:
                st.info(f"• **{r['area']}:** {r['message']}")

        with c_vis2:
            fig_bench, ax_bench = plt.subplots(figsize=(6.5, 3.4), dpi=140)
            apply_pastel_plot_style(fig_bench, ax_bench)

            categories_bench = ["Attendance (%)", "Study (hrs x10)", "Coursework (/100)", "Exam (/100)"]
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
            ax_bench.bar(ind - width/2, student_values, width, label=f"{submission['Name']}", color="#7663C7", edgecolor="#E9E6F2")
            ax_bench.bar(ind + width/2, class_values, width, label="Class Average", color="#A9D5FF", edgecolor="#E9E6F2")
            ax_bench.set_xticks(ind)
            ax_bench.set_xticklabels(categories_bench, fontsize=8.5)
            ax_bench.set_ylabel("Score / Scaled Units", fontsize=9)
            ax_bench.set_title("Performance vs Class Average", fontsize=10.5, fontweight="bold", pad=10)
            ax_bench.legend(fontsize=8, facecolor="#FFFFFF", edgecolor="#E9E6F2")
            plt.tight_layout()
            st.pyplot(fig_bench)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("📑 View Full Performance Scorecard", use_container_width=True, type="primary"):
                st.session_state["current_page"] = "📑 My Performance Report"
                st.rerun()
        with act_col2:
            if st.button("✏️ Submit Updated Academic Details", use_container_width=True):
                st.session_state["current_page"] = "📝 Submit Academic Details"
                st.rerun()

    # 5. Persistent Personal Learning Goals
    st.markdown("---")
    st.markdown('<div class="section-heading">🎯 My Learning Goals</div>', unsafe_allow_html=True)
    st.caption("Track your personal academic targets. Goals are stored persistently in your student profile.")

    with st.expander("➕ Add a New Learning Goal", expanded=False):
        with st.form("add_student_goal_form", clear_on_submit=True):
            goal_title = st.text_input("Goal Title *", placeholder="e.g., Target 85%+ in Data Structures exam")
            goal_target = st.number_input("Target Marks (out of 100) *", min_value=1.0, max_value=100.0, value=80.0, step=1.0)
            goal_date = st.date_input("Target Completion Date (Optional)", value=None, key="student_goal_target_date")
            save_goal = st.form_submit_button("Save Goal", use_container_width=True, type="primary")

        if save_goal:
            target_date_text = goal_date.isoformat() if goal_date else ""
            goal_ok, goal_message = add_student_goal(
                roll_number=current_user["roll_number"],
                title=goal_title,
                target_marks=goal_target,
                target_date=target_date_text
            )
            if goal_ok:
                st.success(goal_message)
                st.rerun()
            else:
                st.error(goal_message)

    student_goals = get_student_goals(current_user["roll_number"])
    if student_goals.empty:
        st.info("You haven't added any learning goals yet. Use the option above to create your first target.")
    else:
        current_average = None
        if submission and status == "Approved":
            current_average = (float(submission.get("Assignment_Marks", 0)) + float(submission.get("Exam_Marks", 0))) / 2.0

        for _, goal in student_goals.reset_index(drop=True).iterrows():
            with st.container(key=f"goal_card_{goal['goal_id']}"):
                gcol1, gcol2, gcol3 = st.columns([2.2, 1, 1.2], vertical_alignment="center")
                with gcol1:
                    st.markdown(f"**{goal['title']}**")
                    due = str(goal.get("target_date", "")).strip()
                    st.caption(f"Target: {float(goal['target_marks']):.0f}/100" + (f" · Target Date: {due}" if due and due.lower() != "nan" else ""))
                    if current_average is not None:
                        progress_pct = min(100.0, max(0.0, current_average / max(float(goal["target_marks"]), 1.0) * 100))
                        st.progress(int(progress_pct), text=f"Progress: {current_average:.1f}/100 vs target ({progress_pct:.0f}%)")
                    else:
                        st.caption("Progress bar will activate once academic submission is verified.")
                with gcol2:
                    st.markdown(
                        '<span class="status-badge-approved">✓ Completed</span>' if str(goal["status"]) == "Completed"
                        else '<span class="status-badge-pending">In Progress</span>',
                        unsafe_allow_html=True
                    )
                with gcol3:
                    new_goal_status = st.selectbox(
                        "Goal status",
                        ["In Progress", "Completed"],
                        index=1 if str(goal["status"]) == "Completed" else 0,
                        key=f"goal_status_{goal['goal_id']}",
                        label_visibility="collapsed"
                    )
                    if new_goal_status != str(goal["status"]):
                        status_ok, status_message = update_student_goal_status(
                            roll_number=current_user["roll_number"],
                            goal_id=str(goal["goal_id"]),
                            status=new_goal_status
                        )
                        if status_ok:
                            st.toast(status_message)
                            st.rerun()
                        else:
                            st.error(status_message)

# ---------------------------------------------------------
# STUDENT PAGE 2: My Profile
# ---------------------------------------------------------
elif page == "👤 My Profile":
    if user_role == "teacher":
        st.markdown('<div class="page-title">👤 Faculty Profile & Credentials</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">View your verified institutional credentials and manage permitted personal contact information.</div>', unsafe_allow_html=True)

        is_approver_user = current_user.get("is_approver", False)
        auth_role_badge = '<span class="role-badge-teacher" style="background:#E3DCFA; color:#402E7A; padding:4px 10px; border-radius:20px; font-weight:600; font-size:12.5px;">⭐ Authorized Faculty Approver</span>' if is_approver_user else '<span class="role-badge-teacher" style="background:#EDE8F5; color:#5D5370; padding:4px 10px; border-radius:20px; font-weight:600; font-size:12.5px;">Faculty Member</span>'

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #F0EEFC 0%, #FFFFFF 100%); border: 1px solid #E9E6F2; border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h3 style="margin: 0; color: #25233A; font-size: 19px; font-weight: 700;">{current_user.get("name", "Faculty Member")}</h3>
                    <p style="margin: 4px 0 0 0; color: #77758A; font-size: 13.5px;">
                        {current_user.get("designation", "Faculty")} · {current_user.get("department", "Department")}
                    </p>
                </div>
                <div>
                    {auth_role_badge}
                    <span style="background:#D8F3E5; color:#1B5E3C; padding:4px 10px; border-radius:20px; font-weight:600; font-size:12.5px; margin-left:6px;">Verified & Active</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Permanent Identity Card (Read-only)
        st.markdown('<div class="section-heading">🪪 Institutional Credentials (Permanent & Read-Only)</div>', unsafe_allow_html=True)
        id_col1, id_col2 = st.columns(2)
        with id_col1:
            tch_id_val = current_user.get("roll_number") or current_user.get("user_id", "")
            st.text_input("Teacher ID / Employee ID (Permanent)", value=tch_id_val, disabled=True)
            st.text_input("College Email (Institutional Identifier)", value=current_user.get("email", ""), disabled=True)
        with id_col2:
            st.text_input("Administrative Authority", value="Authorized Senior Approver (Can approve faculty accounts)" if is_approver_user else "Standard Faculty Access", disabled=True)
            st.text_input("Account Created On", value=current_user.get("created_at", "N/A"), disabled=True)

        # Editable Permitted Details Form
        st.markdown('<div class="section-heading">✏️ Permitted Profile Updates</div>', unsafe_allow_html=True)
        with st.form("teacher_profile_edit_form"):
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                new_t_name = st.text_input("Full Name *", value=current_user.get("name", "")).strip()
                new_t_dept = st.text_input("Department *", value=current_user.get("department", "")).strip()
            with e_col2:
                new_t_desig = st.text_input("Designation *", value=current_user.get("designation", "")).strip()
                new_t_phone = st.text_input("Contact Phone / Office Extension", value=current_user.get("phone", "")).strip()

            save_t_profile = st.form_submit_button("Save Profile Updates", use_container_width=True, type="primary")

        if save_t_profile:
            ok, msg = update_teacher_profile(
                user_id=current_user["user_id"],
                name=new_t_name,
                department=new_t_dept,
                designation=new_t_desig,
                phone=new_t_phone
            )
            if ok:
                current_user["name"] = new_t_name
                current_user["department"] = new_t_dept
                current_user["designation"] = new_t_desig
                current_user["phone"] = new_t_phone
                current_user["course_branch"] = f"{new_t_dept} ({new_t_desig})"
                st.session_state["user"] = current_user
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")
    else:
        st.markdown('<div class="page-title">👤 My Profile & Account Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">View your student credentials and update permitted personal information.</div>', unsafe_allow_html=True)

        with st.form("student_profile_edit_form"):
            st.markdown('<div class="section-heading">🪪 Identity Information (Permanent)</div>', unsafe_allow_html=True)
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.text_input("Roll Number / Student ID (Permanent)", value=current_user["roll_number"], disabled=True)
            with p_col2:
                st.text_input("Account Role", value=current_user["role"].capitalize(), disabled=True)

            st.markdown('<div class="section-heading">✏️ Permitted Information Updates</div>', unsafe_allow_html=True)
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                new_name = st.text_input("Full Name *", value=current_user["name"]).strip()
                new_email = st.text_input("College Email Address *", value=current_user["email"]).strip()
            with f_col2:
                new_course = st.text_input("Course & Branch *", value=current_user["course_branch"]).strip()
                st.text_input("Account Created On", value=current_user.get("created_at", "N/A"), disabled=True)

            save_profile = st.form_submit_button("Save Profile Updates", use_container_width=True, type="primary")

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
    st.markdown('<div class="page-title">📝 Submit Academic Indicators for Faculty Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Submit your term parameters. Values are reviewed by faculty before classification is finalized.</div>', unsafe_allow_html=True)

    submission = get_student_submission(current_user["roll_number"])
    prev_status = submission.get("Submission_Status", "None") if submission else "None"

    if prev_status == "Rejected":
        teacher_remarks = submission.get("Teacher_Remarks", "")
        st.error(f"⚠️ **Your previous submission requires revision.**\n\n**Faculty Reason:** {teacher_remarks}\n\nPlease revise your metrics below and resubmit.")
    elif prev_status == "Pending Review":
        st.warning("ℹ️ **You currently have a pending submission awaiting faculty review.** Updating the form will refresh your pending request.")
    elif prev_status == "Approved":
        st.info("ℹ️ **Your current record is Approved.** If you have updated examination scores or attendance for this semester, submit new values for faculty re-evaluation.")

    def_att = float(submission["Attendance_Percentage"]) if submission else 75.0
    def_study = float(submission["Study_Hours_Per_Day"]) if submission else 3.0
    def_assign = float(submission["Assignment_Marks"]) if submission else 70.0
    def_exam = float(submission["Exam_Marks"]) if submission else 65.0

    with st.form("student_academic_submission_form"):
        st.markdown('<div class="section-heading">Academic Parameters</div>', unsafe_allow_html=True)
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            att_val = st.number_input(
                "Classroom Attendance Percentage (%) *",
                min_value=0.0,
                max_value=100.0,
                value=def_att,
                step=0.5,
                help="University regulations require at least 75% for exam eligibility."
            )
            study_val = st.number_input(
                "Daily Self-Study Hours (hrs/day) *",
                min_value=0.0,
                max_value=16.0,
                value=def_study,
                step=0.25,
                help="Dedicated self-study hours outside regular college lecture hours."
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
                help="Written exam score out of 100."
            )

        submit_details = st.form_submit_button("Submit Details for Faculty Review", use_container_width=True, type="primary")

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
                st.info("Your details have been placed into status: **Pending Review**. You can track approval progress on your Student Dashboard.")
                if st.button("Go to Student Dashboard"):
                    st.session_state["current_page"] = "🏠 Student Dashboard"
                    st.rerun()
            else:
                st.error(f"❌ {msg}")

# ---------------------------------------------------------
# STUDENT PAGE 4: My Performance Report
# ---------------------------------------------------------
elif page == "📑 My Performance Report":
    st.markdown('<div class="page-title">📑 Official Student Performance Scorecard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Detailed academic evaluation, class benchmark comparison, and actionable recommendations.</div>', unsafe_allow_html=True)

    submission = get_student_submission(current_user["roll_number"])
    status = submission.get("Submission_Status", "None") if submission else "None"

    if status != "Approved":
        if status == "Pending Review":
            st.warning("⏳ **Your report is currently awaiting teacher approval.** Once the faculty validates your submission, your scorecard and comparative benchmarks will become accessible.")
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

        class_att_avg = approved_records["Attendance_Percentage"].mean() if not approved_records.empty else 75.0
        class_study_avg = approved_records["Study_Hours_Per_Day"].mean() if not approved_records.empty else 4.0
        class_assign_avg = approved_records["Assignment_Marks"].mean() if not approved_records.empty else 70.0
        class_exam_avg = approved_records["Exam_Marks"].mean() if not approved_records.empty else 65.0

        st.markdown("""<div class="dashboard-card">""", unsafe_allow_html=True)
        h_col1, h_col2 = st.columns([2, 1])
        with h_col1:
            st.markdown(f"### 🎓 Student Performance Report: {submission['Name']}")
            st.markdown(f"**Roll Number:** `{submission['Student_ID']}` &nbsp;|&nbsp; **Branch:** {current_user['course_branch']}")
            st.markdown(f"**Faculty Remarks:** *{submission.get('Teacher_Remarks', 'Approved by Faculty.')}*")
        with h_col2:
            st.markdown("**Overall Classification:**")
            st.markdown(get_badge_html(submission['Performance_Category']), unsafe_allow_html=True)

        st.markdown("---")

        # 4 Comparison KPIs
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Attendance",
                f"{submission['Attendance_Percentage']:.1f}%",
                delta=f"{submission['Attendance_Percentage'] - class_att_avg:+.1f}% vs Class"
            )
        with m2:
            st.metric(
                "Study Hours",
                f"{submission['Study_Hours_Per_Day']:.1f} hrs",
                delta=f"{submission['Study_Hours_Per_Day'] - class_study_avg:+.1f} hrs vs Class"
            )
        with m3:
            st.metric(
                "Assignment Marks",
                f"{submission['Assignment_Marks']:.1f}/100",
                delta=f"{submission['Assignment_Marks'] - class_assign_avg:+.1f} vs Class"
            )
        with m4:
            st.metric(
                "Exam Marks",
                f"{submission['Exam_Marks']:.1f}/100",
                delta=f"{submission['Exam_Marks'] - class_exam_avg:+.1f} vs Class"
            )

        # Benchmark Bar Chart
        st.markdown('<div class="card-title" style="margin-top: 18px;">📊 Individual vs Class Benchmark</div>', unsafe_allow_html=True)
        fig_bench, ax_bench = plt.subplots(figsize=(7, 3.2), dpi=140)
        apply_pastel_plot_style(fig_bench, ax_bench)

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
        ax_bench.bar(ind - width/2, student_values, width, label=f"{submission['Name']}", color="#7663C7", edgecolor="#E9E6F2")
        ax_bench.bar(ind + width/2, class_values, width, label="Class Average", color="#A9D5FF", edgecolor="#E9E6F2")
        ax_bench.set_xticks(ind)
        ax_bench.set_xticklabels(categories_bench, fontsize=8.5)
        ax_bench.set_ylabel("Score / Normalized Scale", fontsize=9)
        ax_bench.legend(fontsize=8, facecolor="#FFFFFF", edgecolor="#E9E6F2")
        plt.tight_layout()
        st.pyplot(fig_bench)
        plt.close()

        # Actionable Recommendations
        st.markdown('<div class="card-title" style="margin-top: 18px;">💡 Tailored Academic Guidance & Roadmaps</div>', unsafe_allow_html=True)
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
            label="📄 Download Official Scorecard (.txt)",
            data=report_txt,
            file_name=f"Report_{submission['Student_ID']}.txt",
            mime="text/plain",
            type="primary"
        )

# =========================================================
# 👨‍🏫 TEACHER PAGES
# =========================================================

# ---------------------------------------------------------
# TEACHER PAGE 1: Dashboard & Analytics
# ---------------------------------------------------------
elif page == "📊 Dashboard & Analytics":
    st.markdown('<div class="page-title">📊 Faculty Dashboard & Academic Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">A comprehensive academic monitoring overview for faculty and course educators.</div>', unsafe_allow_html=True)

    # Personalized Greeting & Quick Profile Entry
    approver_badge_text = " · ⭐ Authorized Approver" if current_user.get("is_approver") else ""
    dept_label = current_user.get("department") or current_user.get("course_branch") or "Faculty of Computer Science"
    desig_label = current_user.get("designation") or "Faculty Member"
    tch_id_display = current_user.get("roll_number") or current_user.get("user_id", "")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #F0EEFC 0%, #FFFFFF 100%); border: 1px solid #E9E6F2; border-radius: 14px; padding: 18px 24px; margin-bottom: 18px;">
        <h2 style="margin: 0; color: #25233A; font-size: 20px; font-weight: 700;">👋 Welcome back, {current_user.get('name', 'Faculty Member')}!</h2>
        <p style="margin: 4px 0 0 0; color: #77758A; font-size: 13.5px;">
            <strong>{desig_label}</strong> — {dept_label}{approver_badge_text} · Employee ID: <code>{tch_id_display}</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_links = st.columns([1, 1, 2])
    with col_links[0]:
        if st.button("👤 View / Edit My Profile", key="btn_dash_my_profile", use_container_width=True):
            st.session_state["current_page"] = "👤 My Profile"
            st.rerun()

    # 1. Approvals alert (only shown to authorized approver)
    if current_user.get("is_approver"):
        with col_links[1]:
            if st.button("👥 Account Approvals", key="btn_dash_approvals_nav", use_container_width=True):
                st.session_state["current_page"] = "👥 Account Approvals"
                st.rerun()

        pending_teachers = get_pending_teacher_accounts()
        if not pending_teachers.empty:
            st.markdown(f"""
            <div style="background-color: #EDE8F5; border: 1px solid #D6CCEC; border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #4A3E70; font-weight: 600; font-size: 14px;">
                    👥 <strong>{len(pending_teachers)} pending faculty registration request(s)</strong> require your administrative review.
                </span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("👉 Review Faculty Account Requests", key="btn_dash_review_faculty_reqs", type="primary"):
                st.session_state["current_page"] = "👥 Account Approvals"
                st.rerun()

    df_records = load_student_records()

    # 2. Pending Submissions Notification Banner
    pending_records = df_records[df_records["Submission_Status"] == "Pending Review"]
    num_pending = len(pending_records)
    if num_pending > 0:
        st.markdown(f"""
        <div style="background-color: #FFF3D6; border: 1px solid #F2D88C; border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #8A5A0A; font-weight: 600; font-size: 14px;">
                🔔 You have <strong>{num_pending} pending student submission(s)</strong> awaiting review & approval!
            </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👉 Review Pending Submissions", key="btn_quick_review", type="primary"):
            st.session_state["current_page"] = "📬 Review Submissions"
            st.rerun()

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

        # 4 Class KPI Cards
        st.markdown('<div class="section-heading">Class Academic Summary</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(render_kpi_card("Approved Students", f"{total_students}", "Enrolled active records", "lavender"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_kpi_card("Avg Exam Score", f"{avg_exam:.1f}", "Overall class marks (/100)", "blue"), unsafe_allow_html=True)
        with col3:
            st.markdown(render_kpi_card("Avg Attendance", f"{avg_attendance:.1f}%", "Overall classroom presence", "mint"), unsafe_allow_html=True)
        with col4:
            st.markdown(render_kpi_card("High Performers", f"{good_pct:.1f}%", "Class in 'Good' standing", "pink"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Analytics Section
        st.markdown('<div class="section-heading">Class Performance Visualizations</div>', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: Performance Tier Distribution
        with chart_col1:
            cat_counts = approved_df["Performance_Category"].value_counts()
            category_order = ["Good", "Average", "Needs Improvement"]
            counts = [cat_counts.get(cat, 0) for cat in category_order]
            colors = ["#DDF7EA", "#FFF3D6", "#FFE5EF"]
            edge_colors = ["#B8ECCF", "#F2D88C", "#F7C2D4"]

            fig, ax = plt.subplots(figsize=(6, 3.8), dpi=140)
            apply_pastel_plot_style(fig, ax)
            bars = ax.bar(category_order, counts, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.52)
            ax.set_title("Distribution of Performance Categories", pad=12)
            ax.set_ylabel("Number of Students")

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center", va="bottom", fontweight="bold", color="#25233A", fontsize=9.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Chart 2: Attendance vs Exam Marks Scatter Plot
        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(6, 3.8), dpi=140)
            apply_pastel_plot_style(fig2, ax2)
            color_map = {"Good": "#1E6B4A", "Average": "#B2780E", "Needs Improvement": "#B83A63"}

            for cat, grp in approved_df.groupby("Performance_Category"):
                ax2.scatter(
                    grp["Attendance_Percentage"],
                    grp["Exam_Marks"],
                    label=cat,
                    color=color_map.get(cat, "#7663C7"),
                    alpha=0.85,
                    edgecolors="#FFFFFF",
                    s=55
                )

            ax2.axvline(75, color="#B83A63", linestyle=":", linewidth=1.5, label="75% Attendance Norm")
            ax2.axhline(50, color="#77758A", linestyle="--", linewidth=1.2, label="50 Mark Passline")

            ax2.set_title("Attendance % vs Exam Marks", pad=12)
            ax2.set_xlabel("Attendance Percentage (%)")
            ax2.set_ylabel("Exam Marks (/100)")
            ax2.set_xlim(40, 102)
            ax2.set_ylim(15, 105)
            ax2.legend(loc="lower right", fontsize=8, facecolor="#FFFFFF", edgecolor="#E9E6F2")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # Row 2 Charts
        chart_col3, chart_col4 = st.columns(2)

        # Chart 3: Average Marks Comparison by Category
        with chart_col3:
            grouped_means = approved_df.groupby("Performance_Category")[["Assignment_Marks", "Exam_Marks"]].mean()
            grouped_means = grouped_means.reindex(["Good", "Average", "Needs Improvement"]).dropna()

            fig3, ax3 = plt.subplots(figsize=(6, 3.8), dpi=140)
            apply_pastel_plot_style(fig3, ax3)
            x_indices = np.arange(len(grouped_means))
            width = 0.35

            ax3.bar(x_indices - width/2, grouped_means["Assignment_Marks"], width, label="Assignment Marks", color="#B8A0F5", edgecolor="#E9E6F2")
            ax3.bar(x_indices + width/2, grouped_means["Exam_Marks"], width, label="Exam Marks", color="#82B8EF", edgecolor="#E9E6F2")

            ax3.set_title("Average Coursework Scores by Category", pad=12)
            ax3.set_xticks(x_indices)
            ax3.set_xticklabels(grouped_means.index)
            ax3.set_ylabel("Average Score (/100)")
            ax3.set_ylim(0, 100)
            ax3.legend(fontsize=8.5, facecolor="#FFFFFF", edgecolor="#E9E6F2")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # Chart 4: Daily Study Hours Distribution
        with chart_col4:
            fig4, ax4 = plt.subplots(figsize=(6, 3.8), dpi=140)
            apply_pastel_plot_style(fig4, ax4)
            ax4.hist(approved_df["Study_Hours_Per_Day"], bins=10, color="#E9E2FF", edgecolor="#B8A0F5", alpha=0.9)
            ax4.set_title("Distribution of Daily Study Hours", pad=12)
            ax4.set_xlabel("Hours per Day")
            ax4.set_ylabel("Number of Students")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        # Recent Records Preview
        st.markdown('<div class="section-heading">Recent Student Records Preview</div>', unsafe_allow_html=True)
        st.dataframe(df_records.tail(8), use_container_width=True)

# ---------------------------------------------------------
# TEACHER PAGE 2: Review Submissions
# ---------------------------------------------------------
elif page == "📬 Review Submissions":
    st.markdown('<div class="page-title">📬 Student Submissions & Faculty Approvals</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Verify student-submitted metrics, provide feedback remarks, and execute Machine Learning classification.</div>', unsafe_allow_html=True)

    df_records = load_student_records()
    pending_df = df_records[df_records["Submission_Status"] == "Pending Review"]
    approved_df = df_records[df_records["Submission_Status"] == "Approved"]
    rejected_df = df_records[df_records["Submission_Status"] == "Rejected"]

    rev_tab1, rev_tab2, rev_tab3 = st.tabs([
        f"⏳ Pending Review ({len(pending_df)})",
        f"✅ Approved Submissions ({len(approved_df)})",
        f"❌ Rejected Submissions ({len(rejected_df)})"
    ])

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

                # Decision Tree Preview
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
                    sc1, sc2 = st.columns([1.6, 1])
                    with sc1:
                        st.markdown(f"**Student Name:** {stu_name} &nbsp;|&nbsp; **Roll Number:** `{stu_id}`")
                        st.markdown(f"**Branch/Course:** {stu_course} &nbsp;|&nbsp; **Email:** {stu_email}")

                        pm1, pm2, pm3, pm4 = st.columns(4)
                        pm1.metric("Attendance", f"{att:.1f}%")
                        pm2.metric("Study Hours", f"{study:.1f} hrs")
                        pm3.metric("Assignment", f"{assign:.1f}/100")
                        pm4.metric("Exam Marks", f"{exam:.1f}/100")

                    with sc2:
                        st.markdown("**AI Decision Tree Preview:**")
                        st.markdown(get_badge_html(model_pred), unsafe_allow_html=True)

                    st.markdown("---")
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

    with rev_tab2:
        st.markdown('<div class="section-heading">Approved Student Records</div>', unsafe_allow_html=True)
        st.dataframe(
            approved_df[["Student_ID", "Name", "Attendance_Percentage", "Study_Hours_Per_Day", "Assignment_Marks", "Exam_Marks", "Performance_Category", "Teacher_Remarks", "Submission_Date"]],
            use_container_width=True
        )

    with rev_tab3:
        st.markdown('<div class="section-heading">Rejected Submissions Awaiting Student Revision</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="page-title">➕ Add Student Details & Predict Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter academic parameters to evaluate student standing using the Decision Tree model.</div>', unsafe_allow_html=True)

    df_records = load_student_records()
    existing_ids = df_records["Student_ID"].tolist() if not df_records.empty else []

    with st.form("student_input_form", clear_on_submit=False):
        st.markdown('<div class="section-heading">📝 Student Information</div>', unsafe_allow_html=True)
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            student_id = st.text_input("Student ID (e.g., STU1050)", placeholder="STU1050").strip().upper()
        with f_col2:
            student_name = st.text_input("Student Full Name", placeholder="e.g., Rahul Sharma").strip()

        st.markdown('<div class="section-heading">📊 Academic Performance Indicators</div>', unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
            study_hours = st.number_input("Daily Study Hours (hrs/day)", min_value=0.0, max_value=16.0, value=3.0, step=0.25)
        with m_col2:
            assignment_marks = st.number_input("Assignment Marks (/100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            exam_marks = st.number_input("Exam Marks (/100)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)

        save_to_csv = st.checkbox("Save this record to database (CSV file)", value=True)
        submitted = st.form_submit_button("Predict Performance & Generate Report", use_container_width=True, type="primary")

    if submitted:
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

            input_df = pd.DataFrame([{
                "Attendance_Percentage": attendance,
                "Study_Hours_Per_Day": study_hours,
                "Assignment_Marks": assignment_marks,
                "Exam_Marks": exam_marks
            }])[feature_names]

            pred_category = model.predict(input_df)[0]
            pred_probs = model.predict_proba(input_df)[0]
            classes = model.classes_

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
                    "Submission_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                success, msg = save_student_record(new_record)
                if success:
                    st.success(f"✅ Record saved for {student_name} ({student_id})!")
                else:
                    st.warning(f"Note: {msg}")

            st.markdown("---")
            st.markdown('<div class="section-heading">🎯 Prediction Result & Performance Assessment</div>', unsafe_allow_html=True)

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
                st.markdown('<div class="card-title">💡 Actionable Recommendations</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="page-title">📋 Student Records & Individual Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">View all student profiles, filter records, download CSV data, and generate printable scorecards.</div>', unsafe_allow_html=True)

    df_records = load_student_records()

    if df_records.empty:
        st.warning("No records currently available. Add students in the 'Add Student & Predict' section.")
    else:
        tab1, tab2 = st.tabs(["📑 All Student Records", "🪪 Individual Student Scorecard"])

        with tab1:
            st.markdown('<div class="section-heading">Search & Filter Records</div>', unsafe_allow_html=True)
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

            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Records as CSV",
                data=csv_buffer.getvalue(),
                file_name="student_performance_records.csv",
                mime="text/csv",
                use_container_width=False,
                type="primary"
            )

        with tab2:
            st.markdown('<div class="section-heading">Generate Student Performance Scorecard</div>', unsafe_allow_html=True)
            student_options = [
                f"{row['Student_ID']} - {row['Name']}"
                for _, row in df_records.iterrows()
            ]

            selected_option = st.selectbox("Select Student Profile:", student_options)

            if selected_option:
                sel_id = selected_option.split(" - ")[0].strip()
                student_data = df_records[df_records["Student_ID"] == sel_id].iloc[0]

                st.markdown("""<div class="dashboard-card">""", unsafe_allow_html=True)
                h_col1, h_col2 = st.columns([2, 1])
                with h_col1:
                    st.markdown(f"### 🎓 Student Performance Report: {student_data['Name']}")
                    st.markdown(f"**Student ID:** `{student_data['Student_ID']}` &nbsp;|&nbsp; **Status:** {get_status_badge_html(student_data['Submission_Status'])}", unsafe_allow_html=True)
                    if student_data.get("Teacher_Remarks"):
                        st.markdown(f"**Faculty Remarks:** *{student_data['Teacher_Remarks']}*")
                with h_col2:
                    st.markdown("**Overall Classification:**")
                    st.markdown(get_badge_html(student_data['Performance_Category']), unsafe_allow_html=True)

                st.markdown("---")

                m1, m2, m3, m4 = st.columns(4)
                class_att_avg = df_records["Attendance_Percentage"].mean()
                class_study_avg = df_records["Study_Hours_Per_Day"].mean()
                class_assign_avg = df_records["Assignment_Marks"].mean()
                class_exam_avg = df_records["Exam_Marks"].mean()

                with m1:
                    st.metric("Attendance", f"{student_data['Attendance_Percentage']:.1f}%", delta=f"{student_data['Attendance_Percentage'] - class_att_avg:+.1f}% vs Class")
                with m2:
                    st.metric("Study Hours", f"{student_data['Study_Hours_Per_Day']:.1f} hrs", delta=f"{student_data['Study_Hours_Per_Day'] - class_study_avg:+.1f} hrs vs Class")
                with m3:
                    st.metric("Assignment Marks", f"{student_data['Assignment_Marks']:.1f}/100", delta=f"{student_data['Assignment_Marks'] - class_assign_avg:+.1f} vs Class")
                with m4:
                    st.metric("Exam Marks", f"{student_data['Exam_Marks']:.1f}/100", delta=f"{student_data['Exam_Marks'] - class_exam_avg:+.1f} vs Class")

                st.markdown('<div class="card-title" style="margin-top: 18px;">📊 Individual vs Class Benchmark</div>', unsafe_allow_html=True)
                fig_bench, ax_bench = plt.subplots(figsize=(7, 3.2), dpi=140)
                apply_pastel_plot_style(fig_bench, ax_bench)

                categories_bench = ["Attendance (%)", "Study Hours (x10)", "Assignments (/100)", "Exam Marks (/100)"]
                student_values = [
                    student_data["Attendance_Percentage"],
                    student_data["Study_Hours_Per_Day"] * 10,
                    student_data["Assignment_Marks"],
                    student_data["Exam_Marks"]
                ]
                class_values = [class_att_avg, class_study_avg * 10, class_assign_avg, class_exam_avg]

                ind = np.arange(len(categories_bench))
                width = 0.35
                ax_bench.bar(ind - width/2, student_values, width, label=f"{student_data['Name']}", color="#7663C7", edgecolor="#E9E6F2")
                ax_bench.bar(ind + width/2, class_values, width, label="Class Average", color="#A9D5FF", edgecolor="#E9E6F2")
                ax_bench.set_xticks(ind)
                ax_bench.set_xticklabels(categories_bench, fontsize=8.5)
                ax_bench.set_ylabel("Score / Normalized Scale", fontsize=9)
                ax_bench.legend(fontsize=8, facecolor="#FFFFFF", edgecolor="#E9E6F2")
                plt.tight_layout()
                st.pyplot(fig_bench)
                plt.close()

                st.markdown('<div class="card-title" style="margin-top: 18px;">💡 Tailored Academic Guidance</div>', unsafe_allow_html=True)
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
                    mime="text/plain",
                    type="primary"
                )

# ---------------------------------------------------------
# TEACHER PAGE 5: Model Evaluation & Insights
# ---------------------------------------------------------
elif page == "🧠 Model Evaluation & Insights":
    st.markdown('<div class="page-title">🧠 Machine Learning Model Evaluation & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Detailed evaluation of training methodology, performance metrics, and ethical boundaries.</div>', unsafe_allow_html=True)

    if metrics_info is None:
        st.warning("Model metrics not found. Please train the model using `model/train_model.py`.")
    else:
        # 4 Model KPI Cards
        st.markdown('<div class="section-heading">Model Architecture & Test Performance</div>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(render_kpi_card("Algorithm", "Decision Tree", "Cart with Gini index", "lavender"), unsafe_allow_html=True)
        with col_m2:
            st.markdown(render_kpi_card("Test Set Accuracy", f"{metrics_info['test_accuracy']}%", "Stratified test sample", "mint"), unsafe_allow_html=True)
        with col_m3:
            st.markdown(render_kpi_card("Training Samples", f"{metrics_info['num_train_samples']}", "80% partition", "blue"), unsafe_allow_html=True)
        with col_m4:
            st.markdown(render_kpi_card("Testing Samples", f"{metrics_info['num_test_samples']}", "20% partition", "pink"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)

        # Feature Importance Plot
        with col_g1:
            st.markdown('<div class="section-heading">📊 Gini Feature Importances</div>', unsafe_allow_html=True)
            st.caption("Contribution of each academic metric toward tree partition decisions.")
            feat_imp = metrics_info.get("feature_importances", {})
            if feat_imp:
                sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
                labels = [f[0].replace("_", " ") for f in sorted_feats]
                vals = [f[1] for f in sorted_feats]

                fig_imp, ax_imp = plt.subplots(figsize=(6, 3.8), dpi=140)
                apply_pastel_plot_style(fig_imp, ax_imp)
                ax_imp.barh(labels[::-1], vals[::-1], color="#7663C7", edgecolor="#E9E6F2")
                ax_imp.set_xlabel("Gini Importance Score")
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close()

        # Confusion Matrix
        with col_g2:
            st.markdown('<div class="section-heading">🎯 Confusion Matrix (Test Partition)</div>', unsafe_allow_html=True)
            st.caption("Rows represent Ground Truth categories; Columns represent Model Predictions.")
            cm = np.array(metrics_info.get("confusion_matrix", []))
            class_names = metrics_info.get("class_names", [])

            if len(cm) > 0:
                fig_cm, ax_cm = plt.subplots(figsize=(5.5, 3.8), dpi=140)
                apply_pastel_plot_style(fig_cm, ax_cm)
                cax = ax_cm.matshow(cm, cmap=plt.cm.Purples, alpha=0.75)

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax_cm.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='large', weight='bold', color="#25233A")

                cbar = fig_cm.colorbar(cax)
                cbar.ax.yaxis.set_tick_params(color='#77758A')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#77758A')

                ax_cm.set_xticks(range(len(class_names)))
                ax_cm.set_yticks(range(len(class_names)))
                ax_cm.set_xticklabels(class_names, rotation=20, fontsize=8)
                ax_cm.set_yticklabels(class_names, fontsize=8)
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("True Label")
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

        # Classification Report Table
        st.markdown('<div class="section-heading">📑 Precision, Recall & F1-Score Breakdown</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-heading">⚠️ Methodological Limitations & Ethical Considerations</div>', unsafe_allow_html=True)
        st.markdown("""
1. **No Data Leakage Guarantee:**
   - The dataset is partitioned into an **80% Training Set** and an isolated **20% Testing Set** using `sklearn.model_selection.train_test_split`.
   - The Decision Tree classifier is trained **exclusively** on the training set. Evaluation metrics (accuracy, confusion matrix, precision/recall) are computed on the unseen test set.

2. **Synthetic Dataset Characteristics:**
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

# ---------------------------------------------------------
# TEACHER PAGE 6: Account Approvals & Faculty Directory
# ---------------------------------------------------------
elif page == "👥 Account Approvals":
    # Server-side guard: Only authorized faculty approvers
    if not (user_role == "teacher" and current_user.get("is_approver")):
        st.error("🚫 Access denied. Only authorized senior faculty approvers have permission to access Account Approvals.")
        st.stop()

    st.markdown('<div class="page-title">👥 Faculty Account Approvals & Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Review incoming faculty access requests, grant institutional permissions, and inspect verified educators.</div>', unsafe_allow_html=True)

    pending_teachers = get_pending_teacher_accounts()
    st.markdown('<div class="section-heading">⏳ Pending Faculty Access Requests</div>', unsafe_allow_html=True)

    if pending_teachers.empty:
        st.success("✅ No pending faculty registration requests at this time. All accounts are up to date.")
    else:
        st.info(f"There are **{len(pending_teachers)}** faculty registration request(s) awaiting your decision.")
        for idx, pending_teacher in pending_teachers.iterrows():
            with st.container(border=True):
                p_email = str(pending_teacher.get("email", ""))
                p_name = str(pending_teacher.get("name", ""))
                p_id = str(pending_teacher.get("roll_number", pending_teacher.get("user_id", "")))
                p_dept = str(pending_teacher.get("department", pending_teacher.get("course_branch", "")))
                p_desig = str(pending_teacher.get("designation", "Faculty Member"))
                p_date = str(pending_teacher.get("created_at", "N/A"))

                req_col1, req_col2 = st.columns([0.65, 0.35])
                with req_col1:
                    st.markdown(f"### {p_name}")
                    st.markdown(f"**Employee ID:** `{p_id}` · **Email:** `{p_email}`")
                    st.markdown(f"**Department:** {p_dept} · **Designation:** {p_desig}")
                    st.caption(f"Submitted on: {p_date}")
                with req_col2:
                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                    btn_app, btn_rej = st.columns(2)
                    with btn_app:
                        if st.button("✅ Approve", key=f"apprv_btn_{idx}_{p_email}", type="primary", use_container_width=True):
                            ok, message = set_teacher_approval(p_email, True, approver_user_id=current_user["user_id"])
                            if ok:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    with btn_rej:
                        if st.button("❌ Reject", key=f"reject_btn_{idx}_{p_email}", use_container_width=True):
                            ok, message = set_teacher_approval(p_email, False, approver_user_id=current_user["user_id"])
                            if ok:
                                st.warning(message)
                                st.rerun()
                            else:
                                st.error(message)

    st.markdown("---")
    st.markdown('<div class="section-heading">📋 Verified Faculty Directory</div>', unsafe_allow_html=True)
    all_users = load_users()
    if not all_users.empty:
        teacher_mask = all_users["role"].astype(str).str.lower() == "teacher"
        faculty_df = all_users.loc[teacher_mask].copy()

        display_cols = ["user_id", "name", "email", "department", "designation", "approval_status", "is_approver", "created_at"]
        for c in display_cols:
            if c not in faculty_df.columns:
                faculty_df[c] = ""

        faculty_table = faculty_df[display_cols].copy()
        faculty_table.rename(columns={
            "user_id": "Employee ID",
            "name": "Full Name",
            "email": "College Email",
            "department": "Department",
            "designation": "Designation",
            "approval_status": "Approval Status",
            "is_approver": "Authorized Approver",
            "created_at": "Registration Date"
        }, inplace=True)
        st.dataframe(faculty_table, use_container_width=True)
