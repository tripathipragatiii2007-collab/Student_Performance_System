"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset for student academic performance prediction.
Features included:
- Student_ID: Unique identifier (e.g., STU1001)
- Name: Student name
- Attendance_Percentage: Class attendance (40.0% - 100.0%)
- Study_Hours_Per_Day: Daily self-study hours (0.5 - 8.0)
- Assignment_Marks: Average assignment score out of 100 (30 - 100)
- Exam_Marks: Final exam score out of 100 (20 - 100)
- Performance_Category: Target label ('Good', 'Average', 'Needs Improvement')
"""

import os
import numpy as np
import pandas as pd

def generate_student_data(num_records: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student performance data with realistic educational logic.
    """
    np.random.seed(random_state)

    first_names = [
        "Aarav", "Ananya", "Rohan", "Priya", "Rahul", "Neha", "Aditya", "Sneha",
        "Vikram", "Pooja", "Amit", "Divya", "Karan", "Kavya", "Siddharth", "Ishita",
        "Manish", "Rhea", "Arjun", "Tanvi", "Varun", "Meera", "Gaurav", "Simran",
        "Harsh", "Deepika", "Kunal", "Anjali", "Suresh", "Sunita"
    ]
    last_names = [
        "Sharma", "Verma", "Gupta", "Patel", "Singh", "Kumar", "Mishra", "Joshi",
        "Reddy", "Nair", "Das", "Choudhury", "Bose", "Mehta", "Bhat"
    ]

    names = [
        f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        for _ in range(num_records)
    ]
    student_ids = [f"STU{1000 + i + 1}" for i in range(num_records)]

    # Generate correlated academic features
    # Base academic aptitude factor (0 to 1) with normal distribution
    aptitude = np.clip(np.random.normal(loc=0.6, scale=0.2, size=num_records), 0.1, 1.0)

    # Attendance: aptitude influences attendance with natural variance
    attendance = np.clip(
        np.round(aptitude * 50 + np.random.uniform(40, 50, size=num_records), 1),
        45.0, 100.0
    )

    # Daily study hours: aptitude influences study hours
    study_hours = np.clip(
        np.round(aptitude * 5.0 + np.random.uniform(0.5, 3.0, size=num_records), 1),
        0.5, 8.0
    )

    # Assignment marks (0-100)
    assignment_marks = np.clip(
        np.round(aptitude * 60 + np.random.uniform(25, 40, size=num_records), 1),
        30.0, 100.0
    )

    # Exam marks (0-100)
    exam_marks = np.clip(
        np.round(aptitude * 65 + np.random.uniform(20, 35, size=num_records), 1),
        25.0, 100.0
    )

    # Compute composite academic index (out of 100)
    # Weights: Exam (45%), Assignment (25%), Attendance (20%), Study hours normalized (10%)
    study_hours_norm = (study_hours / 8.0) * 100.0
    composite_score = (
        0.45 * exam_marks +
        0.25 * assignment_marks +
        0.20 * attendance +
        0.10 * study_hours_norm +
        np.random.normal(0, 3.0, size=num_records)  # small noise for realistic boundary overlap
    )

    # Determine Performance Categories
    categories = []
    for score in composite_score:
        if score >= 72.0:
            categories.append("Good")
        elif score >= 50.0:
            categories.append("Average")
        else:
            categories.append("Needs Improvement")

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Name": names,
        "Attendance_Percentage": attendance,
        "Study_Hours_Per_Day": study_hours,
        "Assignment_Marks": assignment_marks,
        "Exam_Marks": exam_marks,
        "Performance_Category": categories
    })

    return df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "student_data.csv")

    df = generate_student_data(num_records=500, random_state=42)
    df.to_csv(output_path, index=False)
    print(f"[OK] Successfully generated {len(df)} student records at: {output_path}")
    print("\nClass Distribution:")
    print(df["Performance_Category"].value_counts())
    print("\nSample Data (First 5 rows):")
    print(df.head())

if __name__ == "__main__":
    main()
