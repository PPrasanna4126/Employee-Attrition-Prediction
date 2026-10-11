import json
import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Employee Attrition Prediction", page_icon="👥", layout="wide"
)
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F7FB;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main headings */
h1 {
    color: #17365D;
    font-weight: 800;
}

h2, h3 {
    color: #24476B;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #EAF0F8;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #DCE5F1;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(30, 60, 90, 0.06);
}

/* Buttons */
.stButton > button {
    background-color: #1769AA;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #104E80;
    color: white;
    border: none;
}

/* Charts and tables */
div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
}

/* Horizontal separators */
hr {
    border-color: #DCE5F1;
}

</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/expanded_employee_data.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("Employee dataset not found. Please check the data folder.")


# 2. Main Title and Intro
# 2. Professional Dashboard Header
st.markdown("""
<div style="
    background: linear-gradient(120deg, #17365D, #1769AA);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 20px;
">
    <h1 style="color:white; margin:0;">
        Employee Attrition Analytics
    </h1>
    <p style="color:#E5EFFB; font-size:17px; margin-top:10px;">
        AI-Powered Workforce Insights and Employee Retention Analysis
    </p>
    <p style="color:#D5E5F7; font-size:13px; margin-bottom:0;">
        MACHINE LEARNING | HR ANALYTICS | PREDICTIVE INSIGHTS
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Explore workforce trends, analyze employee satisfaction, "
    "and generate machine-learning-based attrition estimates."
)
st.divider()

# 3. Model Loading & Verification
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found.")
    st.info("Run python train_model.py in your project terminal first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# 4. Sidebar Inputs
st.sidebar.header("Employee Information")
st.sidebar.write("Enter the employee details.")

age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=30)
monthly_income = st.sidebar.number_input(
    "Monthly Income (₹)",
    min_value=0,
    max_value=500000,
    value=30000,
    step=1000,
)
years_at_company = st.sidebar.number_input(
    "Years at Company", min_value=0, max_value=50, value=3
)
job_satisfaction = st.sidebar.selectbox(
    "Job Satisfaction", [1, 2, 3, 4], index=2
)
work_life_balance = st.sidebar.selectbox(
    "Work-Life Balance", [1, 2, 3, 4], index=2
)
performance_rating = st.sidebar.selectbox(
    "Performance Rating", [1, 2, 3, 4], index=2
)
distance_from_home = st.sidebar.number_input(
    "Distance From Home", min_value=1, max_value=50, value=10
)
department = st.sidebar.selectbox(
    "Department",
    [
        "Research & Development",
        "Sales",
        "Human Resources",
        "Engineering",
        "Finance",
        "Marketing",
        "Information Technology",
        "Customer Support",
        "Operations",
        "Administration",
        "Business Development",
        "Legal",
        "Product Management",
        "Quality Assurance",
        "Data Science",
        "Cybersecurity",
        "Design",
        "Logistics",
        "Procurement",
        "Research",
    ],
)
overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])

# 5. Selected Employee Overview Metrics
st.subheader("👤 Employee Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Age", age)
with col2:
    st.metric("Monthly Income", f"₹{monthly_income:,}")
with col3:
    st.metric("Years at Company", years_at_company)

st.divider()

# 6. Attrition Prediction Section
st.subheader("🤖 AI-Powered Attrition Prediction")
st.write(
    "Click the button below to generate a prediction "
    "using the trained machine learning model."
)

if st.button("Predict Attrition", type="primary"):
    employee_data = pd.DataFrame(
        [
            {
                "Age": age,
                "MonthlyIncome": monthly_income,
                "YearsAtCompany": years_at_company,
                "JobSatisfaction": job_satisfaction,
                "WorkLifeBalance": work_life_balance,
                "PerformanceRating": performance_rating,
                "DistanceFromHome": distance_from_home,
                "Department": department,
                "OverTime": overtime,
            }
        ]
    )

    try:
        prediction = int(model.predict(employee_data)[0])

        probabilities = model.predict_proba(employee_data)[0]
        attrition_index = list(model.classes_).index(1)
        attrition_probability = float(probabilities[attrition_index]) * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning("⚠️ Model Prediction: Attrition")
        else:
            st.success("✅ Model Prediction: No Attrition")

        st.metric(
            "Estimated Attrition Probability", f"{attrition_probability:.2f}%"
        )

        st.progress(min(max(attrition_probability / 100, 0.0), 1.0))

        st.caption(
            "This probability is a model estimate, not a certainty "
            "about an employee's future behavior."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

# 7. About Project Section
st.subheader("About This Project")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Technologies Used")
    st.write("- Python")
    st.write("- Pandas")
    st.write("- Scikit-learn")
    st.write("- Streamlit")

with col2:
    st.markdown("#### Project Features")
    st.write("- Employee information input")
    st.write("- Machine learning predictions")
    st.write("- Estimated attrition probability")
    st.write("- Interactive dashboard")

st.info(
    "This project uses a synthetic employee dataset generated "
    "for educational and demonstration purposes. "
    "Predictions are estimates and should not be used to make "
    "actual employment decisions."
)

# 8. Dashboard Analytics Section
st.divider()

st.markdown("## 📊 Workforce Analytics")

st.caption(
    "Explore employee attrition patterns across departments, "
    "overtime, and job satisfaction."
)

if os.path.exists(DATA_PATH):

    df = pd.read_csv(DATA_PATH)
    

    # Employee Statistics
    total_employees = len(df)

    employees_left = (
        df["Attrition"] == "Yes"
    ).sum()

    employees_stayed = (
        df["Attrition"] == "No"
    ).sum()

    attrition_rate = (
        employees_left / total_employees * 100
        if total_employees > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Employees",
        f"{total_employees:,}"
    )

    col2.metric(
        "Employees Retained",
        f"{employees_stayed:,}"
    )

    col3.metric(
        "Employees Left",
        f"{employees_left:,}"
    )

    col4.metric(
        "Attrition Rate",
        f"{attrition_rate:.1f}%"
    )

    st.divider()

    # Attrition by Department
    st.subheader("Attrition by Department")

    department_data = pd.crosstab(
        df["Department"],
        df["Attrition"]
    )

    st.bar_chart(department_data)

    # Attrition by Overtime
    st.subheader("Attrition by Overtime")

    overtime_data = pd.crosstab(
        df["OverTime"],
        df["Attrition"]
    )

    st.bar_chart(overtime_data)

    # Attrition by Job Satisfaction
    st.subheader("Attrition by Job Satisfaction")

    satisfaction_data = pd.crosstab(
        df["JobSatisfaction"],
        df["Attrition"]
    ).reindex([1, 2, 3, 4], fill_value=0)

    st.bar_chart(satisfaction_data)

    st.caption(
        "Job satisfaction levels: "
        "1 = Lowest, 2 = Low, 3 = High, 4 = Highest."
    )

else:

    st.error(
        "Employee dataset not found. "
        "Please run generate_data.py first."
    )

st.divider()

st.markdown("### ⚠️ Disclaimer")

st.caption(
    "This project uses a synthetic employee dataset generated "
    "for educational and demonstration purposes. "
    "Predictions are illustrative and should not be used "
    "for real-world employee decisions."
)