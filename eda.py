import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def perform_eda():
    data_path = "data/expanded_employee_data.csv"
    output_dir = "visuals"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("Dataset not found. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)

    # Remove rows with missing values only from this EDA copy.
    # Model preprocessing will handle missing values separately.
    df = df.dropna(subset=["Age", "MonthlyIncome", "JobSatisfaction"]).copy()

    sns.set_theme(style="whitegrid")

    # 1. Attrition rate by department
    department_rate = (
        df.assign(Left=df["Attrition"].eq("Yes"))
        .groupby("Department")["Left"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))
    department_rate.plot(kind="bar")
    plt.title("Attrition Rate by Department")
    plt.xlabel("Department")
    plt.ylabel("Attrition Rate (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attrition_department.png")
    plt.close()

    # 2. Monthly income versus attrition
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
    plt.title("Monthly Income vs Attrition")
    plt.xlabel("Attrition")
    plt.ylabel("Monthly Income (₹)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/income_boxplot.png")
    plt.close()

    # 3. Correlation heatmap
    numeric_df = df.select_dtypes(include="number").drop(
        columns=["EmployeeID"], errors="ignore"
    )

    numeric_df["AttritionBinary"] = (df["Attrition"] == "Yes").astype(int)

    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # 4. Attrition rate by overtime
    overtime_rate = (
        df.assign(Left=df["Attrition"].eq("Yes"))
        .groupby("OverTime")["Left"]
        .mean()
        .mul(100)
    )

    plt.figure(figsize=(7, 5))
    overtime_rate.plot(kind="bar")
    plt.title("Attrition Rate by Overtime")
    plt.xlabel("OverTime")
    plt.ylabel("Attrition Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attrition_overtime.png")
    plt.close()

    # 5. Attrition rate by job satisfaction
    satisfaction_rate = (
        df.assign(Left=df["Attrition"].eq("Yes"))
        .groupby("JobSatisfaction")["Left"]
        .mean()
        .mul(100)
    )

    plt.figure(figsize=(7, 5))
    satisfaction_rate.plot(kind="bar")
    plt.title("Attrition Rate by Job Satisfaction")
    plt.xlabel("Job Satisfaction (1–4)")
    plt.ylabel("Attrition Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attrition_satisfaction.png")
    plt.close()

    print("EDA completed successfully!")
    print(f"Charts saved in the '{output_dir}' folder.")

    print("\nAttrition rate by department (%):")
    print(department_rate.round(2))

    print("\nAttrition rate by overtime (%):")
    print(overtime_rate.round(2))

    print("\nAttrition rate by job satisfaction (%):")
    print(satisfaction_rate.round(2))


if __name__ == "__main__":
    perform_eda()