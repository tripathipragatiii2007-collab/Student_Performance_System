"""
app.py
------
Streamlit Web Application for the Student Performance Report System.
Features:
1. Fixed Top Navbar & Slide-out Hamburger Drawer for modern, full-width navigation.
2. Dashboard & Analytics: Summary KPIs, Dark-themed Matplotlib charts.
3. Add Student & Predict: Input form with validation, ML inference (Decision Tree), probability scores, actionable suggestions.
4. Student Records & Reports: View all records, CSV export, individual student scorecard with class-average benchmarking.
5. Model Insights & Limitations: Empirical accuracy, confusion matrix, feature importance, and educational caveats.
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
    generate_recommendations
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
    else:
        return '<span class="badge-needs-improvement">⚠️ Needs Improvement</span>'

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

# Initialize dataset and records
initialize_records()
model_bundle = load_trained_model()
metrics_info = load_metrics()

# ---------------------------------------------------------
# Navigation State Management
# ---------------------------------------------------------
NAV_MODULES = [
    "📊 Dashboard & Analytics",
    "➕ Add Student & Predict",
    "📋 Student Records & Reports",
    "🧠 Model Evaluation & Insights"
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = NAV_MODULES[0]
if "drawer_open" not in st.session_state:
    st.session_state["drawer_open"] = False

# Handle overlay click via query params
if "close_drawer" in st.query_params:
    st.session_state["drawer_open"] = False
    try:
        del st.query_params["close_drawer"]
    except Exception:
        pass
    st.rerun()

current_page = st.session_state["current_page"]

# ---------------------------------------------------------
# Fixed Top Navbar
# ---------------------------------------------------------
with st.container(key="top_navbar_bar"):
    nav_col1, nav_col2, nav_col3 = st.columns([0.06, 0.64, 0.30], vertical_alignment="center")
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
        st.markdown(
            f'<div class="navbar-status-badge">'
            f'<span class="status-dot"></span> Active: <strong>{clean_name}</strong>'
            f'</div>',
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# Left Slide-out Drawer & Overlay
# ---------------------------------------------------------
if st.session_state["drawer_open"]:
    # 1. Overlay (dims the background; clicking anywhere closes the drawer)
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
        st.caption("SELECT MODULE")

        for idx, mod in enumerate(NAV_MODULES):
            is_active = (current_page == mod)
            btn_key = f"drawer_btn_{idx}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(mod, key=btn_key, type=btn_type, use_container_width=True):
                st.session_state["current_page"] = mod
                st.session_state["drawer_open"] = False
                st.rerun()

        st.markdown('<div class="drawer-divider"></div>', unsafe_allow_html=True)
        st.caption("**Project Quick Specs**")
        st.caption("• Algorithm: Decision Tree Classifier")
        st.caption("• Framework: Streamlit & Scikit-learn")
        st.caption("• Storage: CSV File Storage")
        if metrics_info:
            st.caption(f"• Test Accuracy: **{metrics_info.get('test_accuracy', 0)}%**")
        st.caption("---")
        st.caption("B.Tech 2nd Year Mini Project")

# Page Routing
page = st.session_state["current_page"]

# ---------------------------------------------------------
# PAGE 1: Dashboard & Analytics
# ---------------------------------------------------------
if page == "📊 Dashboard & Analytics":
    st.markdown('<div class="main-title">🎓 Student Performance Report System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A data-driven academic monitoring dashboard for educators and students.</div>', unsafe_allow_html=True)

    df_records = load_student_records()

    if df_records.empty:
        st.warning("No student records found. Add student details using the 'Add Student & Predict' page.")
    else:
        total_students = len(df_records)
        avg_exam = df_records["Exam_Marks"].mean()
        avg_attendance = df_records["Attendance_Percentage"].mean()
        good_pct = (df_records["Performance_Category"] == "Good").mean() * 100

        # KPI Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_students}</div>
                <div class="metric-label">Total Students</div>
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
            cat_counts = df_records["Performance_Category"].value_counts()
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

            for cat, grp in df_records.groupby("Performance_Category"):
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
            grouped_means = df_records.groupby("Performance_Category")[["Assignment_Marks", "Exam_Marks"]].mean()
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
            ax4.hist(df_records["Study_Hours_Per_Day"], bins=10, color="#A78BFA", edgecolor="#1E293B", alpha=0.85)
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
# PAGE 2: Add Student & Predict Performance
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
                    "Performance_Category": pred_category
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
# PAGE 3: Student Records & Reports
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
            f1, f2 = st.columns([2, 1])
            with f1:
                search_query = st.text_input("🔍 Search by Student Name or ID:", placeholder="Enter name or ID...").strip()
            with f2:
                selected_cat = st.selectbox(
                    "Filter by Category:",
                    ["All Categories"] + sorted(list(df_records["Performance_Category"].unique()))
                )

            filtered_df = df_records.copy()
            if search_query:
                filtered_df = filtered_df[
                    filtered_df["Name"].str.contains(search_query, case=False, na=False) |
                    filtered_df["Student_ID"].str.contains(search_query, case=False, na=False)
                ]
            if selected_cat != "All Categories":
                filtered_df = filtered_df[filtered_df["Performance_Category"] == selected_cat]

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
                    st.markdown(f"**Student ID:** `{student_data['Student_ID']}`")
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
# PAGE 4: Model Evaluation & Insights
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
