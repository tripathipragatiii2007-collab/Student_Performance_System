"""
train_model.py
--------------
Trains and evaluates a Decision Tree Classifier to predict student performance.
Features:
  - Attendance_Percentage
  - Study_Hours_Per_Day
  - Assignment_Marks
  - Exam_Marks
Target:
  - Performance_Category ('Good', 'Average', 'Needs Improvement')

Steps:
1. Load student dataset.
2. Train-test split (80-20 stratified) to avoid data leakage.
3. Train Decision Tree Classifier with regularization (max_depth, min_samples_split).
4. Evaluate model using accuracy, confusion matrix, and classification report.
5. Save model and evaluation metrics to disk.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate():
    # Base paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "student_data.csv")
    model_output_path = os.path.join(script_dir, "student_model.pkl")
    metrics_output_path = os.path.join(script_dir, "model_metrics.json")

    # Ensure dataset exists; if not, create it
    if not os.path.exists(data_path):
        print("[INFO] Dataset not found. Generating dataset first...")
        from data.generate_dataset import generate_student_data
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        df_new = generate_student_data(num_records=500, random_state=42)
        df_new.to_csv(data_path, index=False)

    # 1. Load data
    df = pd.read_csv(data_path)
    feature_cols = [
        "Attendance_Percentage",
        "Study_Hours_Per_Day",
        "Assignment_Marks",
        "Exam_Marks"
    ]
    target_col = "Performance_Category"

    X = df[feature_cols]
    y = df[target_col]

    # 2. Train-Test Split (80% training, 20% testing)
    # Stratified split ensures equal proportion of each class in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 3. Train Decision Tree Classifier
    # max_depth=4 and min_samples_split=10 prevent overfitting and ensure interpretability
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 4. Model Evaluation
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    class_names = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred_test, labels=class_names).tolist()
    report_dict = classification_report(y_test, y_pred_test, target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, y_pred_test, target_names=class_names)

    print("==================================================")
    print("           MODEL TRAINING & EVALUATION            ")
    print("==================================================")
    print(f"Training Set Samples: {len(X_train)}")
    print(f"Testing Set Samples:  {len(X_test)}")
    print(f"Training Accuracy:    {train_acc * 100:.2f}%")
    print(f"Testing Accuracy:     {test_acc * 100:.2f}%")
    print("\nClassification Report (Test Data):")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)
    print("==================================================")

    # 5. Save model bundle
    bundle = {
        "model": clf,
        "feature_names": feature_cols,
        "classes": clf.classes_.tolist()
    }
    joblib.dump(bundle, model_output_path)
    print(f"[OK] Model successfully saved at: {model_output_path}")

    # 6. Save metrics as JSON for the web UI
    metrics_data = {
        "train_accuracy": round(train_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "confusion_matrix": cm,
        "class_names": class_names,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in zip(feature_cols, clf.feature_importances_)
        }
    }
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[OK] Metrics successfully saved at: {metrics_output_path}")

if __name__ == "__main__":
    train_and_evaluate()
