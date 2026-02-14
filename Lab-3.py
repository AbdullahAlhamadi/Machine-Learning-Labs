"""
ARTI308 - Machine Learning
Lab 3: Exploratory Data Analysis (EDA)

Dataset: Student Performance Dataset (Mathematics)
This script demonstrates EDA techniques including:
- Data loading and inspection
- Missing value detection
- Duplicate checking
- Data type conversion
- Descriptive statistics
- Univariate and bivariate analysis
- Correlation analysis
- Performance analysis
"""

# ==================== Import Libraries ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look cleaner
sns.set_theme()

# ==================== Load Dataset ====================
# Load the student performance dataset (semicolon-separated)
df = pd.read_csv("student-mat.csv", sep=';')

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())
print("\n" + "="*70 + "\n")

# ==================== Check Missing Values ====================
print("Checking for missing values:")
print("\nMissing values per column:")
print(df.isna().sum())
print("\n" + "="*70 + "\n")

# ==================== Check Duplicate Rows ====================
print("Checking for duplicate rows:")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
print("\n" + "="*70 + "\n")

# ==================== Number of Rows and Columns ====================
print("Dataset dimensions:")
print(f"Shape (rows, columns): {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\n" + "="*70 + "\n")

# ==================== Data Type of Columns ====================
print("Data types of columns:")
print(df.dtypes)
print("\n" + "="*70 + "\n")

# ==================== Column Information ====================
print("Column names:")
print(df.columns.tolist())
print("\n" + "="*70 + "\n")

# ==================== Descriptive Summary Statistics ====================
print("Statistical summary:")
print(df.describe(include='all'))
print("\n" + "="*70 + "\n")

# ==================== Univariate Analysis ====================
# Distribution of Final Grades (G3)
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("final_grade_distribution.png")
plt.show()

# Distribution of Study Time
plt.figure(figsize=(8, 5))
df['studytime'].value_counts().sort_index().plot(kind='bar', color='lightcoral')
plt.title("Distribution of Study Time")
plt.xlabel("Study Time (1: <2hrs, 2: 2-5hrs, 3: 5-10hrs, 4: >10hrs)")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("studytime_distribution.png")
plt.show()

# ==================== Bivariate Analysis ====================
# Print statistics without plotting some
gender_performance = df.groupby('sex')['G3'].mean().sort_values(ascending=False)
print("Average Final Grade by Gender:")
print(gender_performance)
print("\n" + "="*70 + "\n")

school_performance = df.groupby('school')['G3'].mean().sort_values(ascending=False)
print("Average Final Grade by School:")
print(school_performance)
print("\n" + "="*70 + "\n")

# Average Final Grade by Mother's Education Level
medu_performance = df.groupby('Medu')['G3'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
medu_performance.plot(kind='bar', color='lightgreen')
plt.title("Average Final Grade by Mother's Education Level")
plt.xlabel("Mother's Education (0: none, 1: 4th grade, 2: 9th grade, 3: secondary, 4: higher)")
plt.ylabel("Average Final Grade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("grade_by_mother_education.png")
plt.show()

print("Average Final Grade by Mother's Education:")
print(medu_performance)
print("\n" + "="*70 + "\n")

# Average Final Grade by Study Time
studytime_performance = df.groupby('studytime')['G3'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
studytime_performance.plot(kind='bar', color='plum')
plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time (1: <2hrs, 2: 2-5hrs, 3: 5-10hrs, 4: >10hrs)")
plt.ylabel("Average Final Grade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("grade_by_studytime.png")
plt.show()

print("Average Final Grade by Study Time:")
print(studytime_performance)
print("\n" + "="*70 + "\n")

# Print Internet statistics without plotting
internet_performance = df.groupby('internet')['G3'].mean().sort_values(ascending=False)
print("Average Final Grade by Internet Access:")
print(internet_performance)
print("\n" + "="*70 + "\n")

# ==================== Absences vs Grades Relationship ====================
plt.figure(figsize=(10, 6))
sns.scatterplot(x='absences', y='G3', data=df, alpha=0.6, color='orange')
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("absences_vs_grade.png")
plt.show()

# ==================== Correlation Analysis ====================
# Select numerical columns for correlation
numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                  'failures', 'famrel', 'freetime', 'goout', 'Dalc',
                  'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

correlation = df[numerical_cols].corr()

print("Correlation with Final Grade (G3):")
print(correlation['G3'].sort_values(ascending=False))
print("\n" + "="*70 + "\n")

# ==================== Grade Progression Analysis ====================
# Analyzing grade progression from G1 to G2 to G3
grade_progression = df[['G1', 'G2', 'G3']].mean()

plt.figure(figsize=(8, 5))
grade_progression.plot(kind='line', marker='o', linewidth=2, markersize=10)
plt.title("Average Grade Progression")
plt.xlabel("Period (G1: 1st, G2: 2nd, G3: Final)")
plt.ylabel("Average Grade")
plt.xticks([0, 1, 2], ['G1 (Period 1)', 'G2 (Period 2)', 'G3 (Final)'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("grade_progression.png")
plt.show()

print("Average Grades Across Periods:")
print(grade_progression)
print("\n" + "="*70 + "\n")

# ==================== Additional Analysis ====================
# Pass/Fail Analysis (assuming pass grade >= 10)
df['pass_fail'] = df['G3'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
pass_fail_counts = df['pass_fail'].value_counts()

print("Pass/Fail Distribution:")
print(pass_fail_counts)
print("\n")

# Performance by Family Support
famsup_performance = df.groupby('famsup')['G3'].mean().sort_values(ascending=False)
print("Average Final Grade by Family Support:")
print(famsup_performance)
print("\n")

# Performance by School Support
schoolsup_performance = df.groupby('schoolsup')['G3'].mean().sort_values(ascending=False)
print("Average Final Grade by School Support:")
print(schoolsup_performance)
print("\n")

# Performance by Higher Education Aspiration
higher_performance = df.groupby('higher')['G3'].mean().sort_values(ascending=False)
print("Average Final Grade by Higher Education Aspiration:")
print(higher_performance)
print("\n")

# Count of students by address type
address_counts = df['address'].value_counts()
print("Students by Address Type (U: Urban, R: Rural):")
print(address_counts)
print("\n")

# Performance by address type
address_performance = df.groupby('address')['G3'].mean().sort_values(ascending=False)
print("Average Final Grade by Address Type:")
print(address_performance)
print("\n" + "="*70 + "\n")

# ==================== Summary Statistics ====================
print("SUMMARY STATISTICS:")
print(f"Total Students: {len(df)}")
print(f"Average Final Grade: {df['G3'].mean():.2f}")
print(f"Median Final Grade: {df['G3'].median():.2f}")
print(f"Standard Deviation: {df['G3'].std():.2f}")
print(f"Pass Rate (Grade >= 10): {(df['G3'] >= 10).sum() / len(df) * 100:.1f}%")
print(f"Average Absences: {df['absences'].mean():.2f}")
print(f"Average Study Time: {df['studytime'].mean():.2f}")
print("\n" + "="*70 + "\n")

print("EDA Complete! All visualizations have been saved as PNG files.")
