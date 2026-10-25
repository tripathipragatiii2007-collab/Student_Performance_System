"""
utils.py
--------
Helper utilities for the Student Performance Report System:
- User authentication & role management (Teacher / Student)
- Secure PBKDF2 password hashing & validation
- Input validation with clear error messages
- Automated personalized recommendations
- Data file handling (loading, saving, updating student records)
- Academic details submission & teacher approval workflow
"""

import os
import re
import hashlib
import secrets
import uuid
from datetime import datetime
import pandas as pd
from typing import Tuple, List, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Prefer the bundled project's data directory when it exists; otherwise use
# a conventional root-level data directory. This keeps records and accounts
# beside the dataset already included in the ZIP/deployment.
BUNDLED_PROJECT_DIR = os.path.join(PROJECT_ROOT, "student_performance_system")
BUNDLED_DATA_DIR = os.path.join(BUNDLED_PROJECT_DIR, "data")
DATA_DIR = BUNDLED_DATA_DIR if os.path.isdir(BUNDLED_DATA_DIR) else os.path.join(PROJECT_ROOT, "data")
RECORDS_FILE = os.path.join(DATA_DIR, "student_records.csv")
TRAINING_FILE = os.path.join(DATA_DIR, "student_data.csv")
USERS_FILE = os.path.join(DATA_DIR, "users.csv")
GOALS_FILE = os.path.join(DATA_DIR, "student_goals.csv")

# Standard schema for student records
RECORD_COLUMNS = [
    "Student_ID", "Name", "Attendance_Percentage",
    "Study_Hours_Per_Day", "Assignment_Marks",
    "Exam_Marks", "Performance_Category",
    "Submission_Status", "Teacher_Remarks",
    "Submission_Date", "Student_Email", "Course_Branch"
]

def get_records_path() -> str:
    """Ensure data directory exists and return student_records.csv path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return RECORDS_FILE

def get_users_path() -> str:
    """Ensure data directory exists and return users.csv path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return USERS_FILE


def get_student_goals(roll_number: str) -> pd.DataFrame:
    """Return goals belonging only to the requested student roll number."""
    os.makedirs(DATA_DIR, exist_ok=True)
    columns = ["goal_id", "roll_number", "title", "target_marks", "target_date", "status", "created_at"]
    if not os.path.exists(GOALS_FILE):
        pd.DataFrame(columns=columns).to_csv(GOALS_FILE, index=False)
        return pd.DataFrame(columns=columns)
    try:
        goals = pd.read_csv(GOALS_FILE).fillna("")
        for column in columns:
            if column not in goals.columns:
                goals[column] = ""
        return goals[goals["roll_number"].astype(str).str.casefold() == str(roll_number).casefold()][columns].copy()
    except (pd.errors.ParserError, OSError, ValueError):
        return pd.DataFrame(columns=columns)


def add_student_goal(roll_number: str, title: str, target_marks: float, target_date: str = "") -> Tuple[bool, str]:
    """Persist a new academic goal for one student."""
    clean_title = str(title).strip()
    if len(clean_title) < 3:
        return False, "Please enter a goal title with at least 3 characters."
    try:
        target = float(target_marks)
    except (TypeError, ValueError):
        return False, "Target marks must be a number."
    if not 1 <= target <= 100:
        return False, "Target marks must be between 1 and 100."
    os.makedirs(DATA_DIR, exist_ok=True)
    columns = ["goal_id", "roll_number", "title", "target_marks", "target_date", "status", "created_at"]
    if os.path.exists(GOALS_FILE):
        try:
            goals = pd.read_csv(GOALS_FILE).fillna("")
        except (pd.errors.ParserError, OSError, ValueError):
            goals = pd.DataFrame(columns=columns)
    else:
        goals = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in goals.columns:
            goals[column] = ""
    new_goal = {
        "goal_id": uuid.uuid4().hex,
        "roll_number": str(roll_number),
        "title": clean_title,
        "target_marks": target,
        "target_date": str(target_date or ""),
        "status": "In Progress",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    goals = pd.concat([goals[columns], pd.DataFrame([new_goal])], ignore_index=True)
    goals.to_csv(GOALS_FILE, index=False)
    return True, "Your learning goal has been saved."


def update_student_goal_status(roll_number: str, goal_id: str, status: str) -> Tuple[bool, str]:
    """Update a goal only when it belongs to the authenticated student."""
    if status not in {"In Progress", "Completed"}:
        return False, "Invalid goal status."
    if not os.path.exists(GOALS_FILE):
        return False, "No saved goals were found."
    try:
        goals = pd.read_csv(GOALS_FILE).fillna("")
        required = {"goal_id", "roll_number", "status"}
        if not required.issubset(goals.columns):
            return False, "The goals file has an unexpected format."
        owned = (
            goals["goal_id"].astype(str).eq(str(goal_id))
            & goals["roll_number"].astype(str).str.casefold().eq(str(roll_number).casefold())
        )
        if not owned.any():
            return False, "That goal could not be found in your account."
        goals.loc[owned, "status"] = status
        goals.to_csv(GOALS_FILE, index=False)
        return True, "Goal status updated."
    except (pd.errors.ParserError, OSError, ValueError):
        return False, "Unable to update the goal right now."

# ---------------------------------------------------------
# Password Hashing & Security
# ---------------------------------------------------------
def hash_password(password: str, salt: str = None) -> str:
    """
    Secure password hashing using standard library PBKDF2 with SHA-256 and 100,000 iterations.
    Never stores plain text passwords.
    """
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return f"{salt}${key.hex()}"

def verify_password(stored_hash: str, password: str) -> bool:
    """Verifies a plain password against the stored salt$hash string using constant-time comparison."""
    try:
        if not stored_hash or "$" not in stored_hash:
            return False
        salt, hex_key = stored_hash.split('$', 1)
        new_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return secrets.compare_digest(new_key.hex(), hex_key)
    except Exception:
        return False

# ---------------------------------------------------------
# User Authentication Storage & Initialization
# ---------------------------------------------------------
def initialize_users():
    """
    Initializes users.csv. If users file doesn't exist, seeds it with
    an authorized senior faculty account.
    Guarantees backward-compatible schema with department, designation, phone, and is_approver.
    """
    users_path = get_users_path()
    expected = [
        "user_id", "name", "email", "roll_number", "course_branch",
        "password_hash", "role", "created_at", "approval_status",
        "department", "designation", "phone", "is_approver"
    ]
    if not os.path.exists(users_path):
        df_users = pd.DataFrame(columns=expected)
        default_pwd = hash_password("teacher123")
        df_users.loc[0] = [
            "TCH101", "Prof. Sharma", "teacher@school.edu", "TEACHER01",
            "Faculty of Computer Science & IT", default_pwd, "teacher",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "approved",
            "Computer Science & IT", "Senior Professor & Department Chair", "", "True"
        ]
        df_users.to_csv(users_path, index=False)
    else:
        # Guarantee all expected columns exist
        try:
            df = pd.read_csv(users_path, dtype=str).fillna("")
            missing = [c for c in expected if c not in df.columns]
            if missing:
                for c in missing:
                    df[c] = ""
            for c in expected:
                if c in df.columns:
                    df[c] = df[c].astype(str)

            # Ensure approval_status is populated
            blank_status = df["approval_status"].astype(str).str.strip().eq("") | df["approval_status"].astype(str).str.lower().eq("nan")
            df.loc[blank_status, "approval_status"] = "approved"

            # Ensure is_approver column is string representation
            admin_mask = (df["email"].astype(str).str.strip().str.lower() == "teacher@school.edu") | (df["user_id"].astype(str).str.strip() == "TCH101")
            df.loc[admin_mask, "is_approver"] = "True"
            non_admin_mask = ~admin_mask & (df["is_approver"].str.strip().isin(["", "nan", "None"]))
            df.loc[non_admin_mask, "is_approver"] = "False"

            # Populate default department & designation for Prof. Sharma if empty
            if admin_mask.any():
                dept_val = str(df.loc[admin_mask, "department"].values[0]).strip()
                if not dept_val or dept_val.lower() == "nan":
                    df.loc[admin_mask, "department"] = "Computer Science & IT"
                desig_val = str(df.loc[admin_mask, "designation"].values[0]).strip()
                if not desig_val or desig_val.lower() == "nan":
                    df.loc[admin_mask, "designation"] = "Senior Professor & Department Chair"

            df.to_csv(users_path, index=False)
        except Exception as e:
            print(f"Error checking users file: {e}")

def load_users() -> pd.DataFrame:
    """Load users database."""
    initialize_users()
    users_path = get_users_path()
    try:
        return pd.read_csv(users_path, dtype=str).fillna("")
    except Exception as e:
        print(f"Error loading users: {e}")
        return pd.DataFrame()

def validate_student_registration(
    name: str,
    roll_number: str,
    email: str,
    course_branch: str,
    password: str,
    confirm_password: str
) -> Tuple[bool, List[str]]:
    """Validates registration fields with clear, beginner-friendly error messages."""
    errors = []

    # Name validation
    cleaned_name = name.strip()
    if not cleaned_name:
        errors.append("Student Name cannot be empty.")
    elif len(cleaned_name) < 2:
        errors.append("Student Name must be at least 2 characters long.")
    elif not re.match(r"^[a-zA-Z\s\.]+$", cleaned_name):
        errors.append("Student Name must contain only alphabets, spaces, and periods.")

    # Roll Number validation
    cleaned_roll = roll_number.strip().upper()
    if not cleaned_roll:
        errors.append("Roll Number cannot be empty.")
    elif not re.match(r"^[A-Z0-9_-]{3,20}$", cleaned_roll):
        errors.append("Roll Number must be 3-20 alphanumeric characters (e.g., STU2024_01, 22CS101).")

    # Email validation
    cleaned_email = email.strip().lower()
    if not cleaned_email:
        errors.append("Email address cannot be empty.")
    elif not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", cleaned_email):
        errors.append("Please provide a valid email address (e.g., student@college.edu).")

    # Course / Branch validation
    cleaned_course = course_branch.strip()
    if not cleaned_course:
        errors.append("Course / Branch cannot be empty.")

    # Password validation
    if not password:
        errors.append("Password cannot be empty.")
    elif len(password) < 6:
        errors.append("Password must be at least 6 characters long.")
    elif password != confirm_password:
        errors.append("Passwords do not match.")

    # Check for uniqueness against existing users
    df_users = load_users()
    if not df_users.empty:
        existing_rolls = df_users["roll_number"].astype(str).str.strip().str.upper().tolist()
        if cleaned_roll in existing_rolls:
            errors.append(f"Roll Number '{cleaned_roll}' is already registered. Please login or use a unique Roll Number.")

        existing_emails = df_users["email"].astype(str).str.strip().str.lower().tolist()
        if cleaned_email in existing_emails:
            errors.append(f"Email '{cleaned_email}' is already in use by an existing account.")

    return (len(errors) == 0, errors)

def register_student(
    name: str,
    roll_number: str,
    email: str,
    course_branch: str,
    password: str
) -> Tuple[bool, str]:
    """Registers a new student user with hashed password and unique roll number & email."""
    users_path = get_users_path()
    try:
        df_users = load_users()
        cleaned_roll = roll_number.strip().upper()
        cleaned_email = email.strip().lower()
        cleaned_name = name.strip()
        cleaned_course = course_branch.strip()

        # Generate unique user_id
        user_id = f"USR_{cleaned_roll}"
        pwd_hash = hash_password(password)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_row = {
            "user_id": user_id,
            "name": cleaned_name,
            "email": cleaned_email,
            "roll_number": cleaned_roll,
            "course_branch": cleaned_course,
            "password_hash": pwd_hash,
            "role": "student",
            "created_at": now_str
        }

        df_updated = pd.concat([df_users, pd.DataFrame([new_row])], ignore_index=True)
        df_updated.to_csv(users_path, index=False)
        return True, "Registration successful! You can now log in."
    except Exception as e:
        return False, f"Registration failed due to a server error: {str(e)}"

def validate_account_registration(
    name: str,
    email: str,
    course_branch: str = "",
    password: str = "",
    confirm_password: str = "",
    role: str = "student",
    roll_number: str = "",
    department: str = "",
    designation: str = "",
    teacher_id: str = ""
) -> Tuple[bool, List[str]]:
    """Validate student or teacher registration without trusting the selected role."""
    errors = []
    cleaned_name = name.strip()
    cleaned_email = email.strip().lower()
    cleaned_role = role.strip().lower()

    if not cleaned_name:
        errors.append("Full Name is required.")
    elif len(cleaned_name) < 2:
        errors.append("Full Name must be at least 2 characters long.")
    elif not re.match(r"^[a-zA-Z\s.]+$", cleaned_name):
        errors.append("Full Name must contain letters, spaces, and periods only.")

    if not cleaned_email:
        errors.append("College Email is required.")
    elif not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+$", cleaned_email):
        errors.append("Please enter a valid email address.")

    if cleaned_role not in {"student", "teacher"}:
        errors.append("Please select Student or Teacher as the account role.")

    if cleaned_role == "student":
        cleaned_roll = roll_number.strip().upper()
        cleaned_course = course_branch.strip()
        if not cleaned_roll:
            errors.append("Roll Number is required for student accounts.")
        elif not re.match(r"^[A-Z0-9_-]{3,20}$", cleaned_roll):
            errors.append("Roll Number must be 3–20 letters, numbers, underscores, or hyphens.")
        if not cleaned_course:
            errors.append("Course / Branch is required.")

    elif cleaned_role == "teacher":
        cleaned_teacher_id = (teacher_id or roll_number or "").strip().upper()
        cleaned_dept = (department or course_branch or "").strip()
        cleaned_desig = designation.strip()

        if not cleaned_teacher_id:
            errors.append("Teacher ID / Employee ID is required.")
        elif not re.match(r"^[A-Z0-9_-]{3,20}$", cleaned_teacher_id):
            errors.append("Teacher ID must be 3–20 letters, numbers, underscores, or hyphens.")

        if not cleaned_dept:
            errors.append("Department is required.")
        elif len(cleaned_dept) < 2:
            errors.append("Department must be at least 2 characters long.")

        if not cleaned_desig:
            errors.append("Designation is required (e.g., Assistant Professor, Professor).")
        elif len(cleaned_desig) < 2:
            errors.append("Designation must be at least 2 characters long.")

    if not password:
        errors.append("Password is required.")
    elif len(password) < 6:
        errors.append("Password must be at least 6 characters long.")
    if password != confirm_password:
        errors.append("Passwords do not match.")

    df_users = load_users()
    if not df_users.empty:
        existing_emails = df_users["email"].astype(str).str.strip().str.lower().tolist()
        if cleaned_email in existing_emails:
            errors.append("This email is already registered. Please sign in or use another email.")
        if cleaned_role == "student":
            cleaned_roll = roll_number.strip().upper()
            existing_rolls = df_users["roll_number"].astype(str).str.strip().str.upper().tolist()
            if cleaned_roll in existing_rolls:
                errors.append(f"Roll Number '{cleaned_roll}' is already registered.")
        elif cleaned_role == "teacher":
            cleaned_teacher_id = (teacher_id or roll_number or "").strip().upper()
            teacher_rolls = df_users.loc[df_users["role"].astype(str).str.lower() == "teacher", "roll_number"].astype(str).str.strip().str.upper().tolist()
            user_ids = df_users["user_id"].astype(str).str.strip().str.upper().tolist()
            if (cleaned_teacher_id in teacher_rolls) or (cleaned_teacher_id in user_ids) or (f"TCH_{cleaned_teacher_id}" in user_ids):
                errors.append(f"Teacher ID / Employee ID '{cleaned_teacher_id}' is already registered. Please use your assigned ID.")

    return len(errors) == 0, errors


def register_account(
    name: str,
    email: str,
    course_branch: str = "",
    password: str = "",
    role: str = "student",
    roll_number: str = "",
    department: str = "",
    designation: str = "",
    teacher_id: str = ""
) -> Tuple[bool, str]:
    """Create a student account (approved by default) or a teacher account (pending authorized approver approval)."""
    users_path = get_users_path()
    try:
        df_users = load_users()
        cleaned_role = role.strip().lower()
        cleaned_email = email.strip().lower()
        cleaned_name = name.strip()

        if cleaned_role not in {"student", "teacher"}:
            return False, "Invalid account role selected."
        if not df_users.empty and (df_users["email"].astype(str).str.strip().str.lower() == cleaned_email).any():
            return False, "This email is already registered. Please sign in or use another email."

        if cleaned_role == "student":
            cleaned_roll = roll_number.strip().upper()
            cleaned_course = course_branch.strip()
            if not cleaned_roll:
                return False, "Roll Number is required for student accounts."
            if not df_users.empty and (df_users["roll_number"].astype(str).str.strip().str.upper() == cleaned_roll).any():
                return False, f"Roll Number '{cleaned_roll}' is already registered."
            user_id = f"USR_{cleaned_roll}"
            approval_status = "approved"
            dept_val = ""
            desig_val = ""
        else:
            cleaned_teacher_id = (teacher_id or roll_number or "").strip().upper()
            cleaned_dept = (department or course_branch or "").strip()
            cleaned_desig = designation.strip()
            if not cleaned_teacher_id:
                return False, "Teacher ID / Employee ID is required."
            teacher_rolls = df_users.loc[df_users["role"].astype(str).str.lower() == "teacher", "roll_number"].astype(str).str.strip().str.upper().tolist() if not df_users.empty else []
            user_ids = df_users["user_id"].astype(str).str.strip().str.upper().tolist() if not df_users.empty else []
            if (cleaned_teacher_id in teacher_rolls) or (cleaned_teacher_id in user_ids) or (f"TCH_{cleaned_teacher_id}" in user_ids):
                return False, f"Teacher ID / Employee ID '{cleaned_teacher_id}' is already registered."

            user_id = cleaned_teacher_id if cleaned_teacher_id.startswith("TCH_") else f"TCH_{cleaned_teacher_id}"
            cleaned_roll = cleaned_teacher_id
            cleaned_course = f"{cleaned_dept} ({cleaned_desig})" if cleaned_desig else cleaned_dept
            dept_val = cleaned_dept
            desig_val = cleaned_desig
            approval_status = "pending"

        new_row = {
            "user_id": user_id,
            "name": cleaned_name,
            "email": cleaned_email,
            "roll_number": cleaned_roll,
            "course_branch": cleaned_course,
            "password_hash": hash_password(password),
            "role": cleaned_role,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "approval_status": approval_status,
            "department": dept_val,
            "designation": desig_val,
            "phone": "",
            "is_approver": "False"
        }
        df_updated = pd.concat([df_users, pd.DataFrame([new_row])], ignore_index=True)
        df_updated.to_csv(users_path, index=False)
        if cleaned_role == "teacher":
            return True, "Your teacher account has been submitted for approval. You can sign in after an authorized approver approves your account."
        return True, "Student account created successfully. You can now sign in."
    except Exception as e:
        return False, f"Registration failed due to a server error: {str(e)}"


def get_pending_teacher_accounts() -> pd.DataFrame:
    """Return teacher accounts awaiting approval."""
    df_users = load_users()
    cols = ["user_id", "name", "email", "roll_number", "department", "designation", "course_branch", "created_at", "approval_status"]
    if df_users.empty or "approval_status" not in df_users.columns:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df_users.columns:
            df_users[c] = ""
    mask = (df_users["role"].astype(str).str.lower() == "teacher") & (df_users["approval_status"].astype(str).str.lower() == "pending")
    return df_users.loc[mask, cols].copy()


def set_teacher_approval(email: str, approved: bool, approver_user_id: Optional[str] = None) -> Tuple[bool, str]:
    """
    Approve or reject a pending teacher account.
    Enforces authorization check: only an authorized approver can approve/reject accounts.
    """
    try:
        users_path = get_users_path()
        df_users = load_users()

        # Authorization Guard
        if approver_user_id is not None:
            approver_mask = (df_users["user_id"].astype(str) == str(approver_user_id)) & (df_users["role"].astype(str).str.lower() == "teacher")
            if not approver_mask.any():
                return False, "Unauthorized: Approver account not found."
            approver_row = df_users.loc[approver_mask].iloc[0]
            is_approver = str(approver_row.get("is_approver", "")).strip().lower() in ["true", "1", "yes"]
            status = str(approver_row.get("approval_status", "")).strip().lower()
            if not is_approver or status != "approved":
                return False, "Unauthorized: You do not have permission to approve or reject accounts."

        mask = (df_users["email"].astype(str).str.lower() == email.strip().lower()) & (df_users["role"].astype(str).str.lower() == "teacher")
        if not mask.any():
            return False, "Teacher account was not found."

        new_status = "approved" if approved else "rejected"
        df_users.loc[mask, "approval_status"] = new_status
        df_users.to_csv(users_path, index=False)
        return True, f"Teacher account {new_status} successfully."
    except Exception as e:
        return False, f"Unable to update teacher approval: {e}"


def authenticate_user(
    identifier: str,
    password: str,
    expected_role: Optional[str] = None
) -> Tuple[bool, Optional[Dict], str]:
    """
    Authenticates a user by email or roll number/teacher ID.
    Returns (True, user_dict, message) on success, or (False, None, error_message).
    """
    df_users = load_users()
    if df_users.empty:
        return False, None, "No user accounts registered in system."

    cleaned_ident = identifier.strip().lower()

    # Search by email or roll_number (case-insensitive)
    matches = df_users[
        (df_users["email"].astype(str).str.strip().str.lower() == cleaned_ident) |
        (df_users["roll_number"].astype(str).str.strip().str.lower() == cleaned_ident)
    ]

    if matches.empty:
        return False, None, "Account not found. Please check your Roll Number or Email."

    user_row = matches.iloc[0]

    # Verify password
    if not verify_password(str(user_row["password_hash"]), password):
        return False, None, "Incorrect password. Please try again."

    user_dict = {
        "user_id": str(user_row.get("user_id", "")),
        "name": str(user_row.get("name", "")),
        "email": str(user_row.get("email", "")),
        "roll_number": str(user_row.get("roll_number", "")),
        "course_branch": str(user_row.get("course_branch", "")),
        "role": str(user_row.get("role", "student")),
        "created_at": str(user_row.get("created_at", "")),
        "approval_status": str(user_row.get("approval_status", "approved") or "approved").lower(),
        "department": str(user_row.get("department", "") if pd.notna(user_row.get("department")) else ""),
        "designation": str(user_row.get("designation", "") if pd.notna(user_row.get("designation")) else ""),
        "phone": str(user_row.get("phone", "") if pd.notna(user_row.get("phone")) else ""),
        "is_approver": bool(str(user_row.get("is_approver", "")).strip().lower() in ["true", "1", "yes"])
    }

    # Strict server-side access guard for teachers
    if user_dict["role"].lower() == "teacher" and user_dict["approval_status"] != "approved":
        if user_dict["approval_status"] == "pending":
            return False, None, "Your teacher account is awaiting approval by an authorized approver. You can sign in once approved."
        return False, None, "This teacher account request was not approved. Please contact the administrator."

    # Role validation if specified
    if expected_role and user_dict["role"].lower() != expected_role.lower():
        return False, None, f"Access denied. Account role is '{user_dict['role']}', not '{expected_role}'."

    return True, user_dict, "Login successful."


def update_teacher_profile(
    user_id: str,
    name: str,
    department: str,
    designation: str,
    phone: str = ""
) -> Tuple[bool, str]:
    """
    Updates permitted personal profile fields for an authenticated teacher.
    Email and Teacher ID remain protected and read-only.
    Role and approval_status cannot be modified.
    """
    users_path = get_users_path()
    try:
        df_users = load_users()
        for col in ["name", "department", "designation", "phone", "course_branch"]:
            if col not in df_users.columns:
                df_users[col] = ""
            df_users[col] = df_users[col].astype(str)

        mask = (df_users["user_id"].astype(str) == str(user_id)) & (df_users["role"].astype(str).str.lower() == "teacher")
        if not mask.any():
            return False, "Teacher account record was not found."

        cleaned_name = name.strip()
        cleaned_dept = department.strip()
        cleaned_desig = designation.strip()
        cleaned_phone = phone.strip()

        if not cleaned_name or len(cleaned_name) < 2:
            return False, "Full Name must be at least 2 characters long."
        if not re.match(r"^[a-zA-Z\s.]+$", cleaned_name):
            return False, "Full Name must contain letters, spaces, and periods only."
        if not cleaned_dept or len(cleaned_dept) < 2:
            return False, "Department is required."
        if not cleaned_desig or len(cleaned_desig) < 2:
            return False, "Designation is required."
        if cleaned_phone and not re.match(r"^[0-9+\-\s()]{7,20}$", cleaned_phone):
            return False, "Please enter a valid phone number (7-20 digits/symbols)."

        df_users.loc[mask, "name"] = cleaned_name
        df_users.loc[mask, "department"] = cleaned_dept
        df_users.loc[mask, "designation"] = cleaned_desig
        df_users.loc[mask, "course_branch"] = f"{cleaned_dept} ({cleaned_desig})"
        if "phone" in df_users.columns:
            df_users.loc[mask, "phone"] = cleaned_phone

        df_users.to_csv(users_path, index=False)
        return True, "Profile updated successfully!"
    except Exception as e:
        return False, f"Failed to update profile: {e}"

def update_student_profile(
    roll_number: str,
    name: str,
    email: str,
    course_branch: str
) -> Tuple[bool, str]:
    """Updates permitted profile information for a student."""
    users_path = get_users_path()
    try:
        df_users = load_users()
        cleaned_roll = roll_number.strip().upper()
        cleaned_email = email.strip().lower()
        cleaned_name = name.strip()
        cleaned_course = course_branch.strip()

        # Check email uniqueness against other users
        other_users = df_users[df_users["roll_number"].astype(str).str.strip().str.upper() != cleaned_roll]
        if not other_users.empty:
            if cleaned_email in other_users["email"].astype(str).str.strip().str.lower().tolist():
                return False, f"Email '{cleaned_email}' is already taken by another account."

        idx_matches = df_users.index[df_users["roll_number"].astype(str).str.strip().str.upper() == cleaned_roll].tolist()
        if not idx_matches:
            return False, f"Student profile for '{cleaned_roll}' not found."

        idx = idx_matches[0]
        df_users.at[idx, "name"] = cleaned_name
        df_users.at[idx, "email"] = cleaned_email
        df_users.at[idx, "course_branch"] = cleaned_course
        df_users.to_csv(users_path, index=False)

        # Also synchronize student name and email in student_records.csv if record exists
        df_records = load_student_records()
        if not df_records.empty:
            r_idx_matches = df_records.index[df_records["Student_ID"].astype(str).str.strip().str.upper() == cleaned_roll].tolist()
            if r_idx_matches:
                r_idx = r_idx_matches[0]
                df_records.at[r_idx, "Name"] = cleaned_name
                df_records.at[r_idx, "Student_Email"] = cleaned_email
                df_records.at[r_idx, "Course_Branch"] = cleaned_course
                df_records.to_csv(get_records_path(), index=False)

        return True, "Profile details successfully updated!"
    except Exception as e:
        return False, f"Failed to update profile: {str(e)}"

# ---------------------------------------------------------
# Student Records CSV Handling (Backward Compatible)
# ---------------------------------------------------------
def initialize_records():
    """
    Initializes student_records.csv. If records don't exist yet,
    seeds with a subset of the training dataset. Ensures all 12 schema columns exist.
    """
    records_path = get_records_path()
    if not os.path.exists(records_path):
        if os.path.exists(TRAINING_FILE):
            df_train = pd.read_csv(TRAINING_FILE)
            df_seed = df_train.head(35).copy()
            # Add missing schema columns with sensible defaults
            df_seed["Submission_Status"] = "Approved"
            df_seed["Teacher_Remarks"] = ""
            df_seed["Submission_Date"] = datetime.now().strftime("%Y-%m-%d")
            df_seed["Student_Email"] = ""
            df_seed["Course_Branch"] = "B.Tech CSE"
            df_seed.to_csv(records_path, index=False)
        else:
            pd.DataFrame(columns=RECORD_COLUMNS).to_csv(records_path, index=False)
    else:
        # Ensure existing file has all columns
        try:
            df = pd.read_csv(records_path)
            modified = False
            defaults = {
                "Submission_Status": "Approved",
                "Teacher_Remarks": "",
                "Submission_Date": datetime.now().strftime("%Y-%m-%d"),
                "Student_Email": "",
                "Course_Branch": "B.Tech CSE"
            }
            for col, def_val in defaults.items():
                if col not in df.columns:
                    df[col] = def_val
                    modified = True
            if modified:
                df.to_csv(records_path, index=False)
        except Exception as e:
            print(f"Error checking records schema: {e}")

def load_student_records() -> pd.DataFrame:
    """
    Load current student records from CSV.
    Guarantees all expected columns exist, filling defaults for legacy records.
    """
    initialize_records()
    records_path = get_records_path()
    try:
        df = pd.read_csv(records_path)
        defaults = {
            "Submission_Status": "Approved",
            "Teacher_Remarks": "",
            "Submission_Date": "",
            "Student_Email": "",
            "Course_Branch": ""
        }
        for col, def_val in defaults.items():
            if col not in df.columns:
                df[col] = def_val
            else:
                df[col] = df[col].fillna(def_val)
        return df
    except Exception as e:
        print(f"Error loading records: {e}")
        return pd.DataFrame(columns=RECORD_COLUMNS)

def save_student_record(record: dict) -> Tuple[bool, str]:
    """Append or save a student record directly from teacher entry."""
    records_path = get_records_path()
    try:
        df = load_student_records()
        # Ensure default metadata fields are set
        if "Submission_Status" not in record:
            record["Submission_Status"] = "Approved"
        if "Teacher_Remarks" not in record:
            record["Teacher_Remarks"] = "Added directly by Teacher"
        if "Submission_Date" not in record:
            record["Submission_Date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        if "Student_Email" not in record:
            record["Student_Email"] = ""
        if "Course_Branch" not in record:
            record["Course_Branch"] = "B.Tech"

        new_df = pd.DataFrame([record])
        df_combined = pd.concat([df, new_df], ignore_index=True)
        df_combined.to_csv(records_path, index=False)
        return True, "Student record successfully saved!"
    except Exception as e:
        return False, f"Failed to save record: {str(e)}"

# ---------------------------------------------------------
# Academic Details Submission & Teacher Review Workflow
# ---------------------------------------------------------
def validate_academic_inputs(
    attendance: float,
    study_hours: float,
    assignment_marks: float,
    exam_marks: float
) -> Tuple[bool, List[str]]:
    """Validates student academic parameters with strict ranges."""
    errors = []
    if attendance is None or attendance < 0.0 or attendance > 100.0:
        errors.append("Attendance must be between 0% and 100%.")
    if study_hours is None or study_hours < 0.0 or study_hours > 16.0:
        errors.append("Daily Study Hours must be between 0.0 and 16.0 hours.")
    if assignment_marks is None or assignment_marks < 0.0 or assignment_marks > 100.0:
        errors.append("Assignment Marks must be between 0 and 100.")
    if exam_marks is None or exam_marks < 0.0 or exam_marks > 100.0:
        errors.append("Examination Marks must be between 0 and 100.")
    return (len(errors) == 0, errors)

def submit_student_academic_details(
    roll_number: str,
    name: str,
    email: str,
    course_branch: str,
    attendance: float,
    study_hours: float,
    assignment_marks: float,
    exam_marks: float
) -> Tuple[bool, str]:
    """
    Saves or updates a student's academic details with status 'Pending Review'.
    If previous submission was rejected or pending, updates the existing row.
    """
    records_path = get_records_path()
    try:
        df = load_student_records()
        cleaned_roll = roll_number.strip().upper()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Check if record already exists for this roll number
        match_idx = df.index[df["Student_ID"].astype(str).str.strip().str.upper() == cleaned_roll].tolist()

        if match_idx:
            idx = match_idx[0]
            df.at[idx, "Name"] = name
            df.at[idx, "Attendance_Percentage"] = float(attendance)
            df.at[idx, "Study_Hours_Per_Day"] = float(study_hours)
            df.at[idx, "Assignment_Marks"] = float(assignment_marks)
            df.at[idx, "Exam_Marks"] = float(exam_marks)
            df.at[idx, "Submission_Status"] = "Pending Review"
            df.at[idx, "Submission_Date"] = now_str
            df.at[idx, "Student_Email"] = email
            df.at[idx, "Course_Branch"] = course_branch
            # Preserve or clear previous remarks on new submission
            df.at[idx, "Teacher_Remarks"] = "Awaiting teacher review."
        else:
            new_record = {
                "Student_ID": cleaned_roll,
                "Name": name,
                "Attendance_Percentage": float(attendance),
                "Study_Hours_Per_Day": float(study_hours),
                "Assignment_Marks": float(assignment_marks),
                "Exam_Marks": float(exam_marks),
                "Performance_Category": "Pending Review",
                "Submission_Status": "Pending Review",
                "Teacher_Remarks": "Awaiting teacher review.",
                "Submission_Date": now_str,
                "Student_Email": email,
                "Course_Branch": course_branch
            }
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

        df.to_csv(records_path, index=False)
        return True, "Academic details successfully submitted for Teacher review!"
    except Exception as e:
        return False, f"Failed to submit details: {str(e)}"

def get_student_submission(roll_number: str) -> Optional[Dict]:
    """Retrieves the record corresponding to a specific student's roll number."""
    df = load_student_records()
    if df.empty:
        return None
    cleaned_roll = roll_number.strip().upper()
    matches = df[df["Student_ID"].astype(str).str.strip().str.upper() == cleaned_roll]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()

def teacher_review_submission(
    student_id: str,
    action: str,  # 'Approved' or 'Rejected'
    remarks: str,
    predicted_category: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Allows a teacher to Approve or Reject a student's submission with remarks.
    If Approved, updates the Performance_Category with the ML predicted tier.
    """
    records_path = get_records_path()
    try:
        df = load_student_records()
        cleaned_id = student_id.strip().upper()
        match_idx = df.index[df["Student_ID"].astype(str).str.strip().str.upper() == cleaned_id].tolist()

        if not match_idx:
            return False, f"Student submission for '{cleaned_id}' not found."

        idx = match_idx[0]
        cleaned_remarks = remarks.strip() if remarks else ""

        if action == "Approved":
            df.at[idx, "Submission_Status"] = "Approved"
            if predicted_category:
                df.at[idx, "Performance_Category"] = predicted_category
            df.at[idx, "Teacher_Remarks"] = cleaned_remarks if cleaned_remarks else "Approved by Course Faculty."
            msg = f"Submission for {cleaned_id} was successfully APPROVED!"
        elif action == "Rejected":
            df.at[idx, "Submission_Status"] = "Rejected"
            df.at[idx, "Teacher_Remarks"] = cleaned_remarks if cleaned_remarks else "Submission rejected. Please review faculty remarks and resubmit."
            msg = f"Submission for {cleaned_id} was REJECTED with feedback."
        else:
            return False, f"Invalid review action: '{action}'."

        df.to_csv(records_path, index=False)
        return True, msg
    except Exception as e:
        return False, f"Failed to update submission: {str(e)}"

# ---------------------------------------------------------
# Input Validation for Teacher Add Student Form
# ---------------------------------------------------------
def validate_student_inputs(
    student_id: str,
    name: str,
    attendance: float,
    study_hours: float,
    assignment_marks: float,
    exam_marks: float,
    existing_ids: List[str] = None
) -> Tuple[bool, List[str]]:
    """Validates user inputs with descriptive error messages."""
    errors = []

    # Student ID validation
    cleaned_id = student_id.strip().upper()
    if not cleaned_id:
        errors.append("Student ID cannot be empty.")
    elif not re.match(r"^[A-Z0-9_-]{3,20}$", cleaned_id):
        errors.append("Student ID must be 3-20 alphanumeric characters (e.g., STU101, CS2024_01).")
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

    # Numeric bounds
    if attendance is None or attendance < 0.0 or attendance > 100.0:
        errors.append("Attendance must be a valid percentage between 0% and 100%.")

    if study_hours is None or study_hours < 0.0 or study_hours > 24.0:
        errors.append("Daily Study Hours must be between 0.0 and 24.0 hours.")

    if assignment_marks is None or assignment_marks < 0.0 or assignment_marks > 100.0:
        errors.append("Assignment Marks must be between 0 and 100.")

    if exam_marks is None or exam_marks < 0.0 or exam_marks > 100.0:
        errors.append("Exam Marks must be between 0 and 100.")

    return (len(errors) == 0, errors)

# ---------------------------------------------------------
# Actionable Recommendations Engine
# ---------------------------------------------------------
def generate_recommendations(
    attendance: float,
    study_hours: float,
    assignment_marks: float,
    exam_marks: float,
    prediction: str
) -> List[Dict[str, str]]:
    """
    Generates intelligent, actionable improvement tips based on individual metric deficits.
    Returns a list of dicts with 'area', 'type', and 'message'.
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
    elif prediction == "Good":
        recommendations.append({
            "area": "Overall Action Plan",
            "type": "action",
            "message": "Enrichment Plan: Keep leading the class! Consider building open-source projects, participating in technical competitions, or mentoring classmates."
        })

    return recommendations
