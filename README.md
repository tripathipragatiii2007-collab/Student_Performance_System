# Student Performance Report System using Python and Machine Learning

A complete, beginner-friendly B.Tech 2nd-Year Mini Project that analyzes student academic indicators, predicts performance categories using a Machine Learning Decision Tree Classifier, and generates visual analytics with personalized improvement roadmaps.

---

## 📌 Project Overview
The **Student Performance Report System** is an educational diagnostic and reporting platform built with **Python**, **Scikit-learn**, and **Streamlit**. It tracks vital student academic metrics (attendance percentage, daily study hours, assignment marks, and exam marks) and classifies overall student performance into three actionable tiers:
1. **🌟 Good** – Performing with consistent academic excellence.
2. **⚡ Average** – Meeting baseline standards but has clear room for advancement.
3. **⚠️ Needs Improvement** – In need of immediate faculty intervention and structured study habits.

---

## 🗂️ Project Directory Structure

```text
dazzling-shannon/
├── data/
│   ├── generate_dataset.py      # Generates synthetic labeled dataset with realistic educational logic
│   ├── student_data.csv         # 500-sample dataset used for training the Decision Tree model
│   └── student_records.csv      # CSV database storing current student entries & added records
├── model/
│   ├── train_model.py           # Trains Decision Tree Classifier, evaluates metrics, saves model
│   ├── student_model.pkl        # Serialized trained model & feature artifacts (joblib)
│   └── model_metrics.json       # JSON file containing test accuracy, confusion matrix, precision/recall
├── app.py                       # Main Streamlit web application (UI, charts, forms, reports)
├── utils.py                     # Helper routines for validation, recommendations, and CSV operations
├── requirements.txt             # Required Python libraries and dependencies
└── README.md                    # Project documentation, execution guide, and viva preparation
```

---

## 🚀 Step-by-Step Installation & Execution Guide

### Step 1: Open Terminal / Command Prompt
Navigate to your project root folder:
```powershell
cd c:\Users\tripa\Documents\antigravity\dazzling-shannon
```

### Step 2: Install Required Dependencies
Run pip to install all needed libraries:
```powershell
pip install -r requirements.txt
```

### Step 3: Generate the Dataset
Create the synthetic student dataset:
```powershell
python data/generate_dataset.py
```
*Output: Creates `data/student_data.csv` containing 500 realistic student records.*

### Step 4: Train and Evaluate the Machine Learning Model
Train the Decision Tree model and compute performance metrics:
```powershell
python model/train_model.py
```
*Output: Saves the trained model to `model/student_model.pkl` and evaluation metrics to `model/model_metrics.json`.*

### Step 5: Launch the Streamlit Web Application
Start the interactive dashboard:
```powershell
streamlit run app.py
```
*Your browser will automatically open at `http://localhost:8501`.*

---

## 📁 Beginner-Friendly Explanation of Every File

| File Name | Purpose in Simple Language |
| :--- | :--- |
| **`requirements.txt`** | Lists the third-party Python packages (`streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`) needed to run this project. Running `pip install -r requirements.txt` installs them all at once. |
| **`data/generate_dataset.py`** | A Python script that creates 500 realistic student entries using math and probability (`numpy`). It ensures students with higher study hours and attendance generally score better, while adding realistic noise to mirror real-life classrooms. |
| **`data/student_data.csv`** | A spreadsheet-like comma-separated file storing historical student records (ID, Name, Attendance, Study Hours, Assignment, Exam, and Category). This serves as the textbook from which our ML model learns. |
| **`data/student_records.csv`** | The active database file. Whenever an educator adds a new student via the web interface, the details and predicted category are safely stored here. |
| **`model/train_model.py`** | The machine learning trainer. It splits data into 80% training and 20% testing sets, fits the Decision Tree Classifier, measures accuracy, and exports the model file. |
| **`model/student_model.pkl`** | A binary serialized file created by `joblib`. It contains the saved "brain" of the trained Decision Tree so the web app can make instant predictions without retraining every time. |
| **`model/model_metrics.json`** | Stores the verified test accuracy, confusion matrix, precision, and recall so they can be transparently displayed on the dashboard. |
| **`utils.py`** | The helper toolkit. It checks user inputs for errors (e.g., negative marks or duplicate IDs), loads and appends data to the CSV file, and formulates tailored academic recommendations. |
| **`app.py`** | The front-end user interface built using Streamlit. It renders the sidebar, interactive charts, input forms, student scorecards, and CSV download buttons. |
| **`README.md`** | This comprehensive project guide, complete with code walkthroughs, viva questions, and academic documentation. |

---

## 🧠 Machine Learning Methodology

### 1. Algorithm Selection: Decision Tree Classifier
We chose the **Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`) because:
- **High Interpretability:** Decision Trees split data using logical rules (e.g., `If Exam_Marks <= 52.5 and Attendance <= 73.0 -> 'Needs Improvement'`), mirroring how human educators evaluate students.
- **Scale Invariance:** It does not require feature scaling or normalization.
- **Multi-class Capability:** Seamlessly classifies between three categories: Good, Average, and Needs Improvement.

### 2. Preventing Overfitting & Data Leakage
- **Train-Test Split:** The dataset is split into an isolated **80% Training Set** and **20% Testing Set** using stratified sampling (`stratify=y`). This ensures that the test set remains completely unseen during training.
- **Hyperparameter Regularization:**
  - `max_depth=4`: Restricts tree depth to prevent the model from memorizing noise.
  - `min_samples_split=10`: Requires at least 10 samples before a branch can split.
  - `min_samples_leaf=5`: Requires at least 5 samples at any leaf node.

---

## 🎓 Academic Viva & Exam Preparation Guide

### Project Abstract
> Academic institutions face a growing challenge in proactively identifying students requiring academic intervention before end-of-semester examinations. The **Student Performance Report System** is an intelligent diagnostic decision-support system developed using Python, Scikit-learn, and Streamlit. The system evaluates multidimensional academic indicators—specifically class attendance percentage, daily study hours, internal assignment scores, and written exam marks. Utilizing a supervised Decision Tree Classifier trained with stratified regularization, the system classifies student performance into three distinct tiers: Good, Average, and Needs Improvement. Beyond predictive classification, the system generates visual class analytics, individual comparative scorecards, personalized remediation advice, and CSV data exports.

### Objectives of the Project
1. To develop an accessible, automated web application for monitoring student academic performance.
2. To apply supervised machine learning techniques (Decision Tree) to classify student academic standing.
3. To eliminate data leakage by evaluating the trained model strictly on an isolated test set.
4. To deliver personalized, metric-specific suggestions for academic remediation.
5. To provide educators with intuitive graphical insights regarding attendance, coursework scores, and overall class performance.

### Limitations of the Project
1. **Synthetic Data Foundation:** The model is trained on simulated data reflecting educational heuristics rather than a longitudinal multi-university dataset.
2. **Exclusion of Latent Behavioral Factors:** Key determinants such as student physical health, stress, socio-economic factors, and family support are not captured in numerical features.
3. **Axis-Parallel Decision Boundaries:** Standard Decision Trees make orthogonal splits, which can sometimes produce abrupt threshold changes for borderline student cases.
4. **Diagnostic Support Only:** The system serves as an early-alert tool and must never replace teacher counseling or human empathy.

---

### Top 10 Viva Questions with Concise Answers

#### Q1: What is the main objective of this project?
**Ans:** To build an automated academic decision-support tool that predicts student performance categories (Good, Average, Needs Improvement) using machine learning, visualizes class trends, and provides targeted improvement recommendations.

#### Q2: Why did you choose a Decision Tree Classifier instead of other algorithms like SVM or Neural Networks?
**Ans:** Decision Trees offer superior **interpretability** and **explainability**. In education, it is essential to understand *why* a student received a particular category. Decision trees provide clear if-else rule paths that educators and students can easily comprehend.

#### Q3: How did you prevent Data Leakage during model training?
**Ans:** We split the dataset into an 80% training set and a 20% test set using `train_test_split` with `stratify=y` *before* training. The model only fits on the training set, and accuracy/classification reports are evaluated purely on the unseen test set.

#### Q4: What is the purpose of setting `max_depth` in Decision Trees?
**Ans:** If `max_depth` is not limited, a decision tree will continue splitting until every training sample is uniquely classified, leading to severe **overfitting** (high training accuracy, poor real-world test accuracy). Setting `max_depth=4` acts as regularization, ensuring generalizability.

#### Q5: What is the difference between Precision and Recall in this project?
**Ans:** 
- **Precision:** Of all students predicted as "Needs Improvement", how many actually needed improvement?
- **Recall:** Of all students who genuinely needed improvement, how many did the model correctly identify? High recall ensures vulnerable students are not missed.

#### Q6: Why did you use Stratified Splitting (`stratify=y`)?
**Ans:** Stratification guarantees that the proportion of classes (Good, Average, Needs Improvement) is identical in both the training and testing sets, preventing class imbalance skew during evaluation.

#### Q7: What role does `joblib` play in this project?
**Ans:** `joblib` serializes (saves) the trained scikit-learn model object to a `.pkl` file on disk. In the Streamlit app, we simply load the file into memory to make real-time predictions without needing to retrain the model on every user action.

#### Q8: How does the application validate user input?
**Ans:** In `utils.py`, the `validate_student_inputs()` function checks that Student ID is unique and alphanumeric, student name contains valid characters, and numerical values (attendance, marks, hours) stay strictly within permissible ranges (e.g., 0–100%).

#### Q9: What are the primary features used for prediction?
**Ans:** Four continuous numerical features:
1. `Attendance_Percentage` (0–100%)
2. `Study_Hours_Per_Day` (0–16 hours)
3. `Assignment_Marks` (0–100)
4. `Exam_Marks` (0–100)

#### Q10: How can this system be enhanced in future iterations?
**Ans:** Future extensions include:
- Integrating with an institutional SQL database (PostgreSQL/MySQL) instead of CSV files.
- Adding historical semester-over-semester trend tracking.
- Incorporating natural language feedback from instructors.
- Testing ensemble models (Random Forest, Gradient Boosting) for comparative evaluation.
