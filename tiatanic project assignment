
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
titanic_data = pd.read_csv(url)

# Exploratory Data Analysis (EDA)

# Overall survival rate
overall_survival_rate = titanic_data['Survived'].mean()
print(f"Overall Survival Rate: {overall_survival_rate:.2%}")

# Survival rate by gender
survival_by_gender = titanic_data.groupby('Sex')['Survived'].mean()
print("\nSurvival Rate by Gender:")
print(survival_by_gender)

# Distribution of passenger classes
class_distribution = titanic_data['Pclass'].value_counts()
print("\nDistribution of Passenger Classes:")
print(class_distribution)

# Age distribution of passengers
plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['Age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Survival rate by passenger class
survival_by_class = titanic_data.groupby('Pclass')['Survived'].mean()
print("\nSurvival Rate by Passenger Class:")
print(survival_by_class)

# Number of siblings/spouses aboard
sns.countplot(x='SibSp', data=titanic_data)
plt.title('Number of Siblings/Spouses Aboard')
plt.xlabel('Number of Siblings/Spouses')
plt.ylabel('Count')
plt.show()

# Mapping questions to Pandas queries

# Average age of passengers
average_age = titanic_data['Age'].mean()
print(f"\nAverage Age of Passengers: {average_age:.2f} years")

# Families aboard and survival
titanic_data['Family_Size'] = titanic_data['SibSp'] + titanic_data['Parch']
family_survival = titanic_data.groupby('Family_Size')['Survived'].mean()
print("\nSurvival Rate by Family Size:")
print(family_survival)

# Additional Exploratory Data Analysis

# Fare distribution by Passenger Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=titanic_data)
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

# Survival rate by Embarked port
survival_by_embarked = titanic_data.groupby('Embarked')['Survived'].mean()
print("\nSurvival Rate by Embarked Port:")
print(survival_by_embarked)

# Mapping Additional Questions to Pandas Queries

# Average fare by passenger class
average_fare_by_class = titanic_data.groupby('Pclass')['Fare'].mean()
print("\nAverage Fare by Passenger Class:")
print(average_fare_by_class)

# Deriving More Insights

# Analyzing correlations between numerical features
correlation_matrix = titanic_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# More Communicating Insights

"""
The boxplot shows variations in fare across different passenger classes, with higher fares generally associated with higher classes.
Additionally, the correlation matrix reveals interesting relationships between numerical features, such as a negative correlation between Pclass and Fare.
"""

# Feature engineering - creating a new feature 'is_alone'
titanic_data['is_alone'] = (titanic_data['SibSp'] + titanic_data['Parch']) == 0

# More Data Exploration and Visualization

# Countplot for 'is_alone'
plt.figure(figsize=(8, 5))
sns.countplot(x='is_alone', hue='Survived', data=titanic_data)
plt.title('Survival Count based on Traveling Alone')
plt.xlabel('Is Alone')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right')
plt.show()


