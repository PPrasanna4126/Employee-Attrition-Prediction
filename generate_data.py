import pandas as pd
import numpy as np
import os

np.random.seed(42)

departments = [
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
    "Research"
]

n = 3000

data = pd.DataFrame({
    "Age": np.random.randint(21, 61, n),
    "MonthlyIncome": np.random.randint(15000, 150001, n),
    "YearsAtCompany": np.random.randint(0, 31, n),
    "JobSatisfaction": np.random.randint(1, 5, n),
    "WorkLifeBalance": np.random.randint(1, 5, n),
    "PerformanceRating": np.random.randint(1, 5, n),
    "DistanceFromHome": np.random.randint(1, 51, n),
    "Department": np.random.choice(departments, n),
    "OverTime": np.random.choice(["Yes", "No"], n)
})

# Generate illustrative synthetic attrition labels.
# These rules are for demonstration, not validated HR predictions.
risk_score = (
    (data["OverTime"] == "Yes").astype(int) * 0.20
    + (data["JobSatisfaction"] == 1).astype(int) * 0.20
    + (data["WorkLifeBalance"] == 1).astype(int) * 0.15
    + (data["YearsAtCompany"] < 2).astype(int) * 0.15
    + (data["DistanceFromHome"] > 25).astype(int) * 0.10
    + (data["MonthlyIncome"] < 30000).astype(int) * 0.10
)

probability = np.clip(0.10 + risk_score, 0.05, 0.90)

data["Attrition"] = np.where(
    np.random.random(n) < probability,
    "Yes",
    "No"
)

os.makedirs("data", exist_ok=True)

data.to_csv("data/expanded_employee_data.csv", index=False)

print("Expanded synthetic dataset generated successfully!")
print("Total employees:", len(data))
print("Number of departments:", data["Department"].nunique())
print("\nDepartment counts:")
print(data["Department"].value_counts())
print("\nAttrition distribution:")
print(data["Attrition"].value_counts())