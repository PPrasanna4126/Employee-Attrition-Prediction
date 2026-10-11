# Employee Attrition Prediction System

## 📌 Project Overview

The Employee Attrition Prediction System is a machine learning project that predicts whether an employee is likely to leave an organization based on factors such as age, monthly income, years at the company, job satisfaction, work-life balance, performance rating, distance from home, department, and overtime.

The project uses machine learning algorithms to analyze employee data and display predictions through an interactive Streamlit dashboard.

## 🎯 Objectives

- Predict employee attrition using machine learning.
- Analyze employee data to identify patterns associated with attrition.
- Compare different machine learning models.
- Visualize employee statistics through charts.
- Provide an interactive dashboard for exploring predictions.

## ✨ Features

- Employee attrition prediction.
- Attrition probability display.
- Interactive employee input form.
- Employee attrition analytics.
- Department-wise attrition analysis.
- Overtime and job satisfaction analysis.
- Data visualization.
- Model comparison using evaluation metrics.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Model Storage:** Joblib

## 📂 Project Structure

```text
Employee-Attrition-Prediction/
│
├── data/
│   ├── employee_data.csv
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── expanded_employee_data.csv
│
├── models/
│   ├── best_model.pkl
│   ├── model_comparison.csv
│   └── best_model_info.json
│
├── visuals/
│
├── generate_data.py
├── eda.py
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Employee-Attrition-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Generate the employee dataset

```bash
python generate_data.py
```

Run this command only if the required dataset is not already present or you want to regenerate it.

### 5. Perform exploratory data analysis

```bash
python eda.py
```

### 6. Train the machine learning models

```bash
python train_model.py
```

This step generates the trained model and model evaluation files required by the application.

### 7. Launch the Streamlit application

```bash
python -m streamlit run app.py
```

The application should open in your browser.

## 🤖 Machine Learning Models

The project compares the following algorithms:

- Logistic Regression
- Random Forest Classifier
- Extra Trees Classifier

The models are evaluated using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The current training script selects a model based on its F1-score on the test set.

## 📊 Dashboard and Analytics

The Streamlit dashboard allows users to:

- Enter employee details.
- Generate attrition predictions.
- View the predicted attrition probability.
- Explore employee statistics.
- Analyze attrition across departments.
- Examine the relationship between overtime, job satisfaction, and attrition.

## 📁 Dataset Information

The project uses employee data containing demographic, compensation, job satisfaction, and employment-related features.

The expanded dataset includes synthetic employee records generated for educational and demonstration purposes.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only.

Predictions generated using synthetic data may not reflect real-world employee behavior. The model has not been established as a reliable predictor of actual employee attrition.

The system should not be used to make real employment, hiring, promotion, or termination decisions.

## 👨‍💻 Author

Developed as a machine learning project to explore employee attrition analysis and prediction.