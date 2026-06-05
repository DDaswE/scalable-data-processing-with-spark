"""
MIE1628 Assignment 2 - Part A-2
Salary Aggregation with Statistical Analysis
"""

import os
import sys

# Configure Spark Python environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, stddev, expr


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SalaryAnalysis").getOrCreate()

# 2. Load Data
file_path = "salary.txt"
print(f"Loading data from: {file_path}")
print()

# Read as RDD first
salary_rdd = spark.sparkContext.textFile(file_path)


# Split each line and create (department, salary) pairs
parsed_rdd = salary_rdd.map(lambda line: line.split()) \
    .map(lambda parts: (parts[0], int(parts[1])))

# Convert to DataFrame for easier analysis
salary_df = parsed_rdd.toDF(["department", "salary"])

# Display basic info
total_records = salary_df.count()
print(f"Successfully loaded {total_records:,} salary records")
print(f"\nSample data:")
salary_df.show(10, truncate=False)

# 3.  Compute Statistics per Department
print("\nSTATISTICAL ANALYSIS BY DEPARTMENT")


# Calculate comprehensive statistics
stats_df = salary_df.groupBy("department").agg(
    count("salary").alias("employee_count"),
    sum("salary").alias("total_salary"),
    avg("salary").alias("mean_salary"),
    stddev("salary").alias("std_salary"),
    expr("percentile_approx(salary, 0.5)").alias("median_salary"),
    expr("min(salary)").alias("min_salary"),
    expr("max(salary)").alias("max_salary")
).orderBy(col("total_salary").desc())

# Add Coefficient of Variation (CV) column
stats_df = stats_df.withColumn(
    "cv_percent",
    (col("std_salary") / col("mean_salary") * 100)
)

# Reorder columns to show CV after std_salary
stats_df = stats_df.select(
    "department",
    "employee_count",
    "total_salary",
    "mean_salary",
    "median_salary",
    "std_salary",
    "cv_percent",
    "min_salary",
    "max_salary"
)

print("Comprehensive Statistics:")
stats_df.show(truncate=False)

# 4. Detailed Analysis
print("\nDETAILED FINDINGS")


# Convert to pandas for easier manipulation
stats_pd = stats_df.toPandas()
salary_pd = salary_df.toPandas()

# Analysis 1: Department comparison
print("1. Department Comparison:")
print("-" * 50)
for idx, row in stats_pd.iterrows():
    dept = row['department']
    print(f"\n{dept} Department:")
    print(f"Number of employees: {row['employee_count']}")
    print(f"Total salary: ${row['total_salary']:,.0f}")
    print(f"Mean salary: ${row['mean_salary']:,.2f}")
    print(f"Median salary: ${row['median_salary']:,.2f}")
    print(f"Std deviation: ${row['std_salary']:,.2f}")
    print(f"Coefficient of variation: {row['cv_percent']:.2f}%")
    print(f"Salary range: ${row['min_salary']:,.0f} - ${row['max_salary']:,.0f}")

    # Interpret CV
    cv = row['cv_percent']
    if cv > 50:
        print(f"Analysis: HIGH variability in {dept} salaries")
    elif cv > 30:
        print(f"Analysis: MODERATE variability in {dept} salaries")
    else:
        print(f"Analysis: LOW variability in {dept} salaries")

# Analysis 2: Salary distribution characteristics
print("\n\n2. Distribution Characteristics:")
print("-" * 50)
for dept in stats_pd['department']:
    dept_salaries = salary_pd[salary_pd['department'] == dept]['salary']

    # Check for skewness
    skewness = dept_salaries.skew()

    print(f"\n{dept}:")
    print(f"Skewness: {skewness:.3f}", end=" ")
    if abs(skewness) < 0.5:
        print("(approximately symmetric)")
    elif skewness > 0:
        print("(right-skewed: more low salaries)")
    else:
        print("(left-skewed: more high salaries)")

# Visualizations
# 1. Box Plot - Compare salary distributions across departments
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(1, 1, 1)
salary_pd.boxplot(column='salary', by='department', ax=ax1)
plt.title('Salary Distribution by Department\n(Box Plot)', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove the default title
plt.xlabel('Department', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('salary_boxplot.png', dpi=300, bbox_inches='tight')
print("Saved: salary_boxplot.png")
plt.show()

# 2. Pie chart - Total salary distribution
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(1, 1, 1)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
wedges, texts, autotexts = ax2.pie(stats_pd['total_salary'],
                                   labels=stats_pd['department'],
                                   autopct='%1.1f%%',
                                   colors=colors,
                                   startangle=90)
ax2.set_title('Total Salary Distribution\nby Department',
              fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)
plt.tight_layout()
plt.savefig('salary_piechart.png', dpi=300, bbox_inches='tight')
print("Saved: salary_piechart.png")
plt.show()


spark.stop()
print("\n----- Complete -----\n")
