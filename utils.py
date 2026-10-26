import os
import re
import json
import uuid
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# Base Directories & Storage Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "student_performance_system"
DATA_DIR = PROJECT_DIR / "data" if (PROJECT_DIR / "data").exists() else BASE_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model" if (PROJECT_DIR / "model").exists() else BASE_DIR / "model"

STUDENT_RECORDS_PATH = DATA_DIR / "student_records.csv"
USERS_PATH = DATA_DIR / "users.csv"
STUDENT_GOALS_PATH = DATA_DIR / "student_goals.csv"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Security & Hashing Helpers
# -----------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Student Records Management
# -----------------------------------------------------------------------------
def get_records_path() -> Path:
    """Returns the resolved path to student_records.csv."""
    return STUDENT_RECORDS_PATH


def initialize_records():
    """Initializes student_records.csv with default structure if missing."""
    if not STUDENT_RECORDS_PATH.exists() or os.stat(STUDENT_RECORDS_PATH).st_size == 0:
        columns = [
            "record_id",
            "roll_number",
            "name",
            "course",
            "study_hours",
            "attendance",
            "previous_score",
            "predicted_score",
            "actual_score",
            "performance_category",
            "status",
            "teacher_remarks",
            "reviewed_by",
            "submitted_at",
            "reviewed_at"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(STUDENT_RECORDS_PATH, index=False)


def load_student_records() -> pd.DataFrame:
    """Loads student records into a DataFrame."""
    initialize_records()
    try:
        df = pd.read_csv(STUDENT_RECORDS_PATH)
        return df
    except Exception:
        return pd.DataFrame()


def save_student_record(record_dict: Dict[str, Any]) -> bool:
    """Appends or updates a student academic record."""
    initialize_records()
    try:
        df = pd.read_csv(STUDENT_RECORDS_PATH)
        if "record_id" not in record_dict or not record_dict["record_id"]:
            record_dict["record_id"] = uuid.uuid4().hex
        if "submitted_at" not in record_dict or not record_dict["submitted_at"]:
            record_dict["submitted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rec_df = pd.DataFrame([record_dict])
        df = pd.concat([df, rec_df], ignore_index=True)
        df.to_csv(STUDENT_RECORDS_PATH, index=False)
        return True
    except Exception as e:
        print(f"Error saving student record: {e}")
        return False


def validate_student_inputs(study_hours: float, attendance: float, previous_score: float) -> Tuple[bool, List[str]]:
    """Validates student academic inputs."""
    errors = []
    if study_hours is None or study_hours < 0 or study_hours > 24:
        errors.append("Study hours must be between 0 and 24 hours per day.")
    if attendance is None or attendance < 0 or attendance > 100:
        errors.append("Attendance must be between 0% and 100%.")
    if previous_score is None or previous_score < 0 or previous_score > 100:
        errors.append("Previous score must be between 0 and 100.")
    return len(errors) == 0, errors


# -----------------------------------------------------------------------------
# User Account Management & Authentication
# -----------------------------------------------------------------------------
def initialize_users():
    """Initializes users.csv if missing."""
    if not USERS_PATH.exists() or os.stat(USERS_PATH).st_size == 0:
        columns = [
            "user_id",
            "email",
            "password_hash",
            "role",
            "name",
            "roll_number",
            "course",
            "department",
            "teacher_id",
            "is_approved",
            "created_at"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(USERS_PATH, index=False)


def load_users() -> pd.DataFrame:
    """Loads users.csv into a DataFrame."""
    initialize_users()
    try:
        return pd.read_csv(USERS_PATH, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def register_student(
    name: str,
    email: str,
    password: str,
    roll_number: str,
    course: str
) -> Tuple[bool, str]:
    """Registers a student account."""
    initialize_users()
    df = load_users()
    clean_email = email.strip().lower()
    clean_roll = roll_number.strip().upper()

    if not df.empty:
        if (df["email"].str.lower() == clean_email).any():
            return False, "An account with this email address already exists."
        if (df["roll_number"].str.upper() == clean_roll).any():
            return False, "An account with this Roll Number already exists."

    new_user = {
        "user_id": uuid.uuid4().hex,
        "email": clean_email,
        "password_hash": hash_password(password),
        "role": "student",
        "name": name.strip(),
        "roll_number": clean_roll,
        "course": course.strip(),
        "department": "",
        "teacher_id": "",
        "is_approved": "True",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
    df.to_csv(USERS_PATH, index=False)
    return True, "Account registered successfully! You can now log in."


def authenticate_user(identifier: str, password: str, role: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticates a user by email or roll_number (for students) / employee_id (for teachers).
    """
    initialize_users()
    df = load_users()
    if df.empty:
        return False, None, "No accounts found in the system. Please register first."

    clean_id = identifier.strip()
    pw_hash = hash_password(password)

    matched = pd.DataFrame()
    if role == "student":
        matched = df[
            (df["role"] == "student") &
            ((df["email"].str.lower() == clean_id.lower()) | (df["roll_number"].str.upper() == clean_id.upper()))
        ]
    elif role == "teacher":
        matched = df[
            (df["role"] == "teacher") &
            ((df["email"].str.lower() == clean_id.lower()) | (df["teacher_id"].str.upper() == clean_id.upper()))
        ]

    if matched.empty:
        return False, None, f"No {role} account found matching '{clean_id}'."

    user_row = matched.iloc[0]
    if user_row["password_hash"] != pw_hash:
        return False, None, "Incorrect password. Please try again."

    if role == "teacher" and str(user_row.get("is_approved", "")).lower() not in ["true", "1"]:
        return False, None, "Your teacher account is pending administrative approval."

    user_dict = user_row.to_dict()
    return True, user_dict, "Login successful!"


def update_student_profile(user_id: str, name: str, course: str) -> Tuple[bool, str]:
    """Updates student profile details."""
    initialize_users()
    df = load_users()
    if df.empty or (df["user_id"] != user_id).all():
        return False, "User not found."

    df.loc[df["user_id"] == user_id, "name"] = name.strip()
    df.loc[df["user_id"] == user_id, "course"] = course.strip()
    df.to_csv(USERS_PATH, index=False)
    return True, "Profile updated successfully!"


def update_teacher_profile(user_id: str, name: str, department: str) -> Tuple[bool, str]:
    """Updates teacher profile details."""
    initialize_users()
    df = load_users()
    if df.empty or (df["user_id"] != user_id).all():
        return False, "User not found."

    df.loc[df["user_id"] == user_id, "name"] = name.strip()
    df.loc[df["user_id"] == user_id, "department"] = department.strip()
    df.to_csv(USERS_PATH, index=False)
    return True, "Profile updated successfully!"


def validate_student_registration(name: str, email: str, roll_number: str, course: str, password: str, confirm_password: str) -> Tuple[bool, List[str]]:
    """Validates registration fields for students."""
    return validate_account_registration("student", name, email, password, confirm_password, roll_number=roll_number, course=course)


def validate_account_registration(
    role: str,
    name: str,
    email: str,
    password: str,
    confirm_password: str,
    roll_number: str = "",
    course: str = "",
    teacher_id: str = "",
    department: str = ""
) -> Tuple[bool, List[str]]:
    """Validates account registration inputs."""
    errors = []
    cleaned_name = name.strip()
    cleaned_email = email.strip().lower()
    cleaned_roll = roll_number.strip().upper()
    cleaned_course = course.strip()
    cleaned_teacher_id = teacher_id.strip().upper()
    cleaned_dept = department.strip()

    if not cleaned_name or len(cleaned_name) < 2:
        errors.append("Full Name must be at least 2 characters long.")

    email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not cleaned_email or not re.match(email_pattern, cleaned_email):
        errors.append("Please provide a valid email address.")

    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters long.")

    if password != confirm_password:
        errors.append("Passwords do not match.")

    if role == "student":
        if not cleaned_roll:
            errors.append("Roll Number is required for student accounts.")
        elif not re.match(r"^[A-Z0-9_-]{3,20}$", cleaned_roll):
            errors.append("Roll Number must be 3–20 letters, numbers, underscores, or hyphens.")
        if not cleaned_course:
            errors.append("Course / Branch is required.")

    elif role == "teacher":
        if not cleaned_teacher_id:
            errors.append("Teacher ID / Employee ID is required.")
        elif not re.match(r"^[A-Z0-9_-]{3,20}$", cleaned_teacher_id):
            errors.append("Teacher ID must be 3–20 letters, numbers, underscores, or hyphens.")

        if not cleaned_dept:
            errors.append("Department is required.")

    return len(errors) == 0, errors


def register_account(
    role: str,
    name: str,
    email: str,
    password: str,
    roll_number: str = "",
    course: str = "",
    teacher_id: str = "",
    department: str = ""
) -> Tuple[bool, str]:
    """Registers a new student or teacher account."""
    initialize_users()
    df = load_users()
    clean_email = email.strip().lower()

    if not df.empty and (df["email"].str.lower() == clean_email).any():
        return False, "An account with this email address already exists."

    if role == "student":
        clean_roll = roll_number.strip().upper()
        if not df.empty and (df["roll_number"].str.upper() == clean_roll).any():
            return False, "An account with this Roll Number already exists."
        is_approved = "True"
    else:
        clean_tid = teacher_id.strip().upper()
        if not df.empty and (df["teacher_id"].str.upper() == clean_tid).any():
            return False, "An account with this Teacher ID already exists."
        is_approved = "False"

    new_user = {
        "user_id": uuid.uuid4().hex,
        "email": clean_email,
        "password_hash": hash_password(password),
        "role": role,
        "name": name.strip(),
        "roll_number": roll_number.strip().upper() if role == "student" else "",
        "course": course.strip() if role == "student" else "",
        "department": department.strip() if role == "teacher" else "",
        "teacher_id": teacher_id.strip().upper() if role == "teacher" else "",
        "is_approved": is_approved,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
    df.to_csv(USERS_PATH, index=False)

    if role == "teacher":
        return True, "Teacher registration submitted! Your account is pending administrator approval."
    return True, "Registration successful! You may now sign in."


def get_pending_teacher_accounts() -> pd.DataFrame:
    """Returns all teacher accounts awaiting administrator approval."""
    df = load_users()
    if df.empty:
        return pd.DataFrame()
    return df[(df["role"] == "teacher") & (df["is_approved"].str.lower() != "true")]


def set_teacher_approval(user_id: str, approve: bool) -> Tuple[bool, str]:
    """Approves or rejects a teacher account."""
    initialize_users()
    df = load_users()
    if df.empty or (df["user_id"] != user_id).all():
        return False, "User not found."

    if approve:
        df.loc[df["user_id"] == user_id, "is_approved"] = "True"
        msg = "Teacher account approved."
    else:
        df = df[df["user_id"] != user_id]
        msg = "Teacher account rejected and removed."

    df.to_csv(USERS_PATH, index=False)
    return True, msg


# -----------------------------------------------------------------------------
# Academic Details & Submission Workflow
# -----------------------------------------------------------------------------
def validate_academic_inputs(study_hours: float, attendance: float, previous_score: float) -> Tuple[bool, List[str]]:
    """Validates inputs for student academic submissions."""
    return validate_student_inputs(study_hours, attendance, previous_score)


def submit_student_academic_details(
    roll_number: str,
    name: str,
    course: str,
    study_hours: float,
    attendance: float,
    previous_score: float,
    predicted_score: float,
    performance_category: str
) -> Tuple[bool, str]:
    """Saves a student academic submission awaiting review."""
    initialize_records()
    df = load_student_records()

    record = {
        "record_id": uuid.uuid4().hex,
        "roll_number": roll_number.strip().upper(),
        "name": name.strip(),
        "course": course.strip(),
        "study_hours": float(study_hours),
        "attendance": float(attendance),
        "previous_score": float(previous_score),
        "predicted_score": round(float(predicted_score), 2),
        "actual_score": "",
        "performance_category": performance_category,
        "status": "Submitted",
        "teacher_remarks": "",
        "reviewed_by": "",
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reviewed_at": ""
    }

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(STUDENT_RECORDS_PATH, index=False)
    return True, "Academic details submitted successfully!"


def get_student_submission(roll_number: str) -> Optional[Dict[str, Any]]:
    """Fetches the latest academic submission for a given roll number."""
    df = load_student_records()
    if df.empty:
        return None
    matched = df[df["roll_number"].astype(str).str.upper() == roll_number.strip().upper()]
    if matched.empty:
        return None
    return matched.iloc[-1].to_dict()


def teacher_review_submission(
    record_id: str,
    teacher_name: str,
    status: str,
    remarks: str,
    actual_score: Optional[float] = None
) -> Tuple[bool, str]:
    """Records teacher feedback and reviews on a student submission."""
    initialize_records()
    df = load_student_records()
    if df.empty or (df["record_id"] != record_id).all():
        return False, "Submission record not found."

    df.loc[df["record_id"] == record_id, "status"] = status
    df.loc[df["record_id"] == record_id, "teacher_remarks"] = remarks.strip()
    df.loc[df["record_id"] == record_id, "reviewed_by"] = teacher_name.strip()
    df.loc[df["record_id"] == record_id, "reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if actual_score is not None:
        df.loc[df["record_id"] == record_id, "actual_score"] = float(actual_score)

    df.to_csv(STUDENT_RECORDS_PATH, index=False)
    return True, "Submission reviewed successfully!"


# -----------------------------------------------------------------------------
# Learning Goals Management
# -----------------------------------------------------------------------------
def initialize_goals():
    """Initializes student_goals.csv if missing."""
    if not STUDENT_GOALS_PATH.exists() or os.stat(STUDENT_GOALS_PATH).st_size == 0:
        columns = [
            "goal_id",
            "roll_number",
            "title",
            "category",
            "target_date",
            "status",
            "created_at"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(STUDENT_GOALS_PATH, index=False)


def get_student_goals(roll_number: str) -> pd.DataFrame:
    """Returns learning goals for a given student."""
    initialize_goals()
    try:
        df = pd.read_csv(STUDENT_GOALS_PATH)
        if df.empty:
            return df
        return df[df["roll_number"].astype(str).str.upper() == roll_number.strip().upper()]
    except Exception:
        return pd.DataFrame()


def add_student_goal(
    roll_number: str,
    title: str,
    category: str,
    target_date: str
) -> Tuple[bool, str]:
    """Creates a new learning goal for a student."""
    initialize_goals()
    if not title.strip():
        return False, "Goal title cannot be empty."

    df = pd.read_csv(STUDENT_GOALS_PATH)
    new_goal = {
        "goal_id": uuid.uuid4().hex,
        "roll_number": roll_number.strip().upper(),
        "title": title.strip(),
        "category": category.strip(),
        "target_date": str(target_date),
        "status": "In Progress",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df = pd.concat([df, pd.DataFrame([new_goal])], ignore_index=True)
    df.to_csv(STUDENT_GOALS_PATH, index=False)
    return True, "Learning goal added successfully!"


def update_student_goal_status(goal_id: str, new_status: str) -> Tuple[bool, str]:
    """Updates the status of a student goal."""
    initialize_goals()
    df = pd.read_csv(STUDENT_GOALS_PATH)
    if df.empty or (df["goal_id"] != goal_id).all():
        return False, "Goal not found."

    df.loc[df["goal_id"] == goal_id, "status"] = new_status
    df.to_csv(STUDENT_GOALS_PATH, index=False)
    return True, f"Goal marked as {new_status}!"


# -----------------------------------------------------------------------------
# Actionable Recommendations Engine
# -----------------------------------------------------------------------------
def generate_recommendations(study_hours: float, attendance: float, previous_score: float, predicted_score: float) -> List[Dict[str, str]]:
    """Generates personalized academic recommendations."""
    recommendations = []

    if attendance < 75.0:
        recommendations.append({
            "title": "Improve Class Attendance",
            "category": "Attendance",
            "priority": "High",
            "advice": f"Your current attendance is {attendance:.1f}%. Aim for at least 80% to avoid missing crucial lectures."
        })
    elif attendance >= 90.0:
        recommendations.append({
            "title": "Maintain Exemplary Attendance",
            "category": "Attendance",
            "priority": "Low",
            "advice": "Excellent attendance! Staying consistent ensures you don't miss exam tips and discussions."
        })

    if study_hours < 3.0:
        recommendations.append({
            "title": "Increase Self-Study Hours",
            "category": "Study Habits",
            "priority": "High",
            "advice": "Daily self-study under 3 hours can impact exam readiness. Aim for 3.5 to 4.5 hours daily."
        })
    elif study_hours > 7.0:
        recommendations.append({
            "title": "Balance Study and Rest",
            "category": "Wellness",
            "priority": "Medium",
            "advice": "High study hours are great, but ensure you take adequate breaks to prevent burnout."
        })

    if previous_score < 60.0:
        recommendations.append({
            "title": "Strengthen Core Fundamentals",
            "category": "Academics",
            "priority": "High",
            "advice": "Review past exam papers and consult teachers during office hours to solidify foundational concepts."
        })

    if predicted_score >= 80.0:
        recommendations.append({
            "title": "Aim for Academic Honors",
            "category": "Excellence",
            "priority": "Low",
            "advice": "Your predicted trajectory is strong! Practice advanced problems and assist peers to reinforce mastery."
        })
    elif predicted_score < 50.0:
        recommendations.append({
            "title": "Seek Immediate Academic Support",
            "category": "Remedial",
            "priority": "High",
            "advice": "Schedule a 1-on-1 session with your course instructor to formulate an urgent recovery study plan."
        })

    return recommendations


# -----------------------------------------------------------------------------
# Comparison & Metric Delta Calculations
# -----------------------------------------------------------------------------
def format_diff_magnitude(magnitude: float) -> str:
    """
    Rounds magnitude to 1 decimal place for readability.
    If a non-zero difference rounds to 0.0, displays enough decimal places
    to show that the difference is non-zero rather than incorrectly showing zero.
    """
    formatted = f"{magnitude:.1f}"
    if float(formatted) == 0.0 and magnitude > 0:
        for d in range(2, 11):
            candidate = f"{magnitude:.{d}f}"
            if float(candidate) > 0:
                trimmed = candidate.rstrip('0').rstrip('.')
                return trimmed if float(trimmed) > 0 else candidate
        return f"{magnitude:.6f}".rstrip('0').rstrip('.')
    return formatted


def get_comparison_delta(student_val: float, class_avg: float, unit: str) -> Tuple[str, str]:
    """
    Computes directional comparison against class average following strict rules:
    - Below average (student_val < class_avg): ↓ X units below class average (red)
    - Above average (student_val > class_avg): ↑ X units above class average (green)
    - Exactly equal (student_val == class_avg): At class average (off)
    - Strict zero-tolerance threshold: no epsilon/tolerance range.
    - Non-negative magnitude alongside appropriate directional indicator (no redundant negative signs).
    - If non-zero difference rounds to 0.0, displays enough decimal places to show it is non-zero.
    Returns:
        (delta_text, delta_color)
    """
    student_val_f = float(student_val)
    class_avg_f = float(class_avg)
    diff = student_val_f - class_avg_f

    if student_val_f < class_avg_f:
        mag = abs(diff)
        mag_str = format_diff_magnitude(mag)
        return f"↓ {mag_str} {unit} below class average", "red"
    elif student_val_f > class_avg_f:
        mag = abs(diff)
        mag_str = format_diff_magnitude(mag)
        return f"↑ {mag_str} {unit} above class average", "green"
    else:
        return "At class average", "off"
