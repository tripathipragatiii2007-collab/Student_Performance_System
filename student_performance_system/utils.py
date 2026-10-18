"""
utils.py
--------
Helper utilities for the Student Performance Report System:
- Input validation with clear error messages
- Automated personalized recommendations
- Data file handling (loading, saving student records)
"""

import os
import re
import pandas as pd
from typing import Tuple, List, Dict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RECORDS_FILE = os.path.join(DATA_DIR, "student_records.csv")
TRAINING_FILE = os.path.join(DATA_DIR, "student_data.csv")

def get_records_path() -> str:
    """Ensure data directory exists and return student_records.csv path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return RECORDS_FILE

def initialize_records():
    """
    Initializes student_records.csv. If records don't exist yet,
    seeds with a subset of the training dataset so the dashboard has rich data.
    """
    records_path = get_records_path()
    if not os.path.exists(records_path):
        if os.path.exists(TRAINING_FILE):
            df_train = pd.read_csv(TRAINING_FILE)
            # Take the first 35 records for demonstration
            df_seed = df_train.head(35).copy()
            df_seed.to_csv(records_path, index=False)
        else:
            # Empty skeleton
            cols = [
                "Student_ID", "Name", "Attendance_Percentage",
                "Study_Hours_Per_Day", "Assignment_Marks",
                "Exam_Marks", "Performance_Category"
            ]
            pd.DataFrame(columns=cols).to_csv(records_path, index=False)

def load_student_records() -> pd.DataFrame:
    """Load current student records from CSV."""
    initialize_records()
    records_path = get_records_path()
    try:
        df = pd.read_csv(records_path)
        return df
    except Exception as e:
        print(f"Error loading records: {e}")
        return pd.DataFrame()

def save_student_record(record: dict) -> Tuple[bool, str]:
    """Append a new student record to student_records.csv."""
    records_path = get_records_path()
    try:
        df = load_student_records()
        new_df = pd.DataFrame([record])
        df_combined = pd.concat([df, new_df], ignore_index=True)
        df_combined.to_csv(records_path, index=False)
        return True, "Student record successfully saved!"
    except Exception as e:
        return False, f"Failed to save record: {str(e)}"

def validate_student_inputs(
    student_id: str,
    name: str,
    attendance: float,
    study_hours: float,
    assignment_marks: float,
    exam_marks: float,
    existing_ids: List[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validates user inputs with beginner-friendly, descriptive error messages.
    """
    errors = []

    # Student ID validation
    cleaned_id = student_id.strip().upper()
    if not cleaned_id:
        errors.append("Student ID cannot be empty.")
    elif not re.match(r"^[A-Z0-9_-]{3,15}$", cleaned_id):
        errors.append("Student ID must be 3-15 alphanumeric characters (e.g., STU101, CS2024_01).")
    elif existing_ids and cleaned_id in [str(i).strip().upper() for i in existing_ids]:
        errors.append(f"Student ID '{cleaned_id}' already exists. Please use a unique ID.")

    # Name validation
    cleaned_name = name.strip()
    if not cleaned_name:
        errors.append("Student Name cannot be empty.")
    elif len(cleaned_name) < 2:
        errors.append("Student Name must be at least 2 characters long.")
    elif not re.match(r"^[a-zA-Z\s\.]+$", cleaned_name):
        errors.append("Student Name must contain only alphabets, spaces, and periods.")

    # Attendance percentage validation
    if attendance is None or attendance < 0.0 or attendance > 100.0:
        errors.append("Attendance must be a valid percentage between 0% and 100%.")

    # Study hours validation
    if study_hours is None or study_hours < 0.0 or study_hours > 24.0:
        errors.append("Daily Study Hours must be between 0.0 and 24.0 hours.")

    # Assignment marks validation
    if assignment_marks is None or assignment_marks < 0.0 or assignment_marks > 100.0:
        errors.append("Assignment Marks must be between 0 and 100.")

    # Exam marks validation
    if exam_marks is None or exam_marks < 0.0 or exam_marks > 100.0:
        errors.append("Exam Marks must be between 0 and 100.")

    return (len(errors) == 0, errors)

def generate_recommendations(
    attendance: float,
    study_hours: float,
    assignment_marks: float,
    exam_marks: float,
    prediction: str
) -> List[Dict[str, str]]:
    """
    Generates intelligent, actionable improvement tips based on individual metric deficits.
    Returns a list of dicts with 'area', 'status', and 'suggestion'.
    """
    recommendations = []

    # Attendance evaluation
    if attendance < 75.0:
        recommendations.append({
            "area": "Classroom Attendance",
            "type": "warning",
            "message": f"Attendance is {attendance:.1f}%, which is below the mandatory 75% university norm. Attend all upcoming lectures and practical labs to avoid debarment."
        })
    elif attendance < 85.0:
        recommendations.append({
            "area": "Classroom Attendance",
            "type": "info",
            "message": f"Attendance is satisfactory ({attendance:.1f}%), but aim for 85%+ to maximize classroom interaction and internal assessments."
        })
    else:
        recommendations.append({
            "area": "Classroom Attendance",
            "type": "success",
            "message": f"Excellent attendance record ({attendance:.1f}%). Consistent presence aids comprehension of core concepts."
        })

    # Study hours evaluation
    if study_hours < 2.0:
        recommendations.append({
            "area": "Daily Self-Study",
            "type": "warning",
            "message": f"Dedicated study time ({study_hours:.1f} hrs/day) is insufficient. Aim for at least 2.5 to 3 hours daily using focused 25-minute Pomodoro sessions."
        })
    elif study_hours < 4.0:
        recommendations.append({
            "area": "Daily Self-Study",
            "type": "info",
            "message": f"Current study routine ({study_hours:.1f} hrs/day) is balanced. Keep a weekend revision slot for difficult subject topics."
        })
    else:
        recommendations.append({
            "area": "Daily Self-Study",
            "type": "success",
            "message": f"High self-study commitment ({study_hours:.1f} hrs/day). Ensure periodic breaks to prevent mental burnout."
        })

    # Assignment marks evaluation
    if assignment_marks < 50.0:
        recommendations.append({
            "area": "Assignments & Coursework",
            "type": "warning",
            "message": f"Assignment score is {assignment_marks:.1f}/100. Review faculty feedback, solve practice problem sets, and seek peer group assistance."
        })
    elif assignment_marks < 75.0:
        recommendations.append({
            "area": "Assignments & Coursework",
            "type": "info",
            "message": f"Assignment score is moderate ({assignment_marks:.1f}/100). Focus on timely submissions and thorough problem explanations."
        })
    else:
        recommendations.append({
            "area": "Assignments & Coursework",
            "type": "success",
            "message": f"Strong coursework score ({assignment_marks:.1f}/100). Demonstrates steady mastery over ongoing syllabus topics."
        })

    # Exam marks evaluation
    if exam_marks < 50.0:
        recommendations.append({
            "area": "Exam Preparation",
            "type": "warning",
            "message": f"Exam score is {exam_marks:.1f}/100. Solve at least 3 previous-year exam papers, consult course teachers during office hours, and re-attempt failed test problems."
        })
    elif exam_marks < 75.0:
        recommendations.append({
            "area": "Exam Preparation",
            "type": "info",
            "message": f"Exam score is {exam_marks:.1f}/100. Practice time-bound mock exams to enhance accuracy and answer presentation."
        })
    else:
        recommendations.append({
            "area": "Exam Preparation",
            "type": "success",
            "message": f"Outstanding exam score ({exam_marks:.1f}/100). Maintain this standard by engaging with higher-order conceptual questions."
        })

    # Overall Category Roadmap
    if prediction == "Needs Improvement":
        recommendations.append({
            "area": "Overall Action Plan",
            "type": "action",
            "message": "High-Priority Plan: Schedule a 1-on-1 academic mentoring session with your course faculty, form a dedicated study circle, and draft a structured 30-day recovery plan."
        })
    elif prediction == "Average":
        recommendations.append({
            "area": "Overall Action Plan",
            "type": "action",
            "message": "Advancement Plan: You have a dependable foundation. Increasing daily revision by 1 hour and targeting 80%+ on upcoming tests will promote you to the 'Good' tier."
        })
    else:
        recommendations.append({
            "area": "Overall Action Plan",
            "type": "action",
            "message": "Enrichment Plan: Keep leading the class! Consider building open-source projects, participating in technical competitions, or mentoring classmates."
        })

    return recommendations
