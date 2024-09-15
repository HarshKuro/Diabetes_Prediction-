# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Setting a seed for reproducibility
SEED = 42

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Display first few rows of the dataset
print(data.head())

# Descriptive statistics for each feature
print(data.describe())

# Correlation matrix of the features
correlation_matrix = data.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Glucose level distribution (Histogram)
plt.figure(figsize=(10, 6))
data['Glucose'].hist(bins=20, color='skyblue', alpha=0.7)
plt.title('Distribution of Glucose')
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
plt.show()

# Outcome count (Countplot)
plt.figure(figsize=(8, 5))
sns.countplot(x='Outcome', data=data, palette='viridis')
plt.title('Count of Outcomes')
plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
plt.ylabel('Count')
plt.show()

# Proportion of diabetic and non-diabetic patients (Pie Chart)
outcome_counts = data['Outcome'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(outcome_counts, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral'])
plt.title('Proportion of Outcomes')
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(data, hue='Outcome', diag_kind='kde', palette='husl')
plt.show()

# Boxplot to examine distribution and outliers for all features
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, orient="h", palette="Set2")
plt.title("Box Plot of Feature Distributions")
plt.show()

# Function to replace 0 values in certain features with the mean (based on diabetes outcome)
def replace_zero(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    df.loc[(df[field] == 0) & (df[target] == 0), field] = mean_by_target.iloc[0][0]
    df.loc[(df[field] == 0) & (df[target] == 1), field] = mean_by_target.iloc[1][0]

# Replace zero values in the specified columns with mean based on outcome
for col in ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    replace_zero(data, col, 'Outcome')

# Turkey's method for detecting outliers
def TurkeyOutliers(df, nameOfFeature, drop=False):
    valueOfFeature = df[nameOfFeature]
    Q1 = np.percentile(valueOfFeature, 25.)
    Q3 = np.percentile(valueOfFeature, 75.)
    step = (Q3 - Q1) * 1.5
    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    feature_outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values
    if drop:
        df.drop(outliers, inplace=True)
    return outliers

# Outlier detection in the 'Insulin' feature
insulin_outliers = TurkeyOutliers(data, 'Insulin', drop=True)

# Checking the shape of the data after outlier removal
print("Data shape after removing outliers:", data.shape)

# Splitting the dataset into features (X) and the target variable (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=SEED),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
    'AdaBoost': AdaBoostClassifier(random_state=SEED),
    'Naive Bayes': GaussianNB()
}

# Function to train classifier and return model
def train_clf(clf, X_train, y_train):
    return clf.fit(X_train, y_train)

# Function to make predictions and calculate F1-score
def pred_clf(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target, y_pred, pos_label=1)

# Function to calculate accuracy of the classifier
def accu_clf(clf, features, target):
    y_pred = clf.predict(features)
    return accuracy_score(target, y_pred)

# Dictionary to store F1-scores and accuracies of models
clf_scores = {}

# Train and evaluate all classifiers
for clf_name, clf in classifiers.items():
    model = train_clf(clf, X_train, y_train)
    train_f1 = pred_clf(model, X_train, y_train)
    test_f1 = pred_clf(model, X_test, y_test)
    test_acc = accu_clf(model, X_test, y_test)
    
    # Store results
    clf_scores[clf_name] = {
        'Train F1 Score': train_f1,
        'Test F1 Score': test_f1,
        'Test Accuracy': test_acc
    }

# Print evaluation results
for clf_name, scores in clf_scores.items():
    print(f"\nClassifier: {clf_name}")
    print(f"Train F1 Score: {scores['Train F1 Score']:.3f}")
    print(f"Test F1 Score: {scores['Test F1 Score']:.3f}")
    print(f"Test Accuracy: {scores['Test Accuracy']:.3f}")

# Visualizing the performance of classifiers (F1 scores and accuracies)

# Bar plot for F1 scores
clf_names = list(clf_scores.keys())
train_f1_scores = [clf_scores[name]['Train F1 Score'] for name in clf_names]
test_f1_scores = [clf_scores[name]['Test F1 Score'] for name in clf_names]
test_accuracies = [clf_scores[name]['Test Accuracy'] for name in clf_names]

# Plotting F1 scores for training and testing sets
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(clf_names))

plt.bar(index, train_f1_scores, bar_width, label='Train F1 Score', color='skyblue')
plt.bar(index + bar_width, test_f1_scores, bar_width, label='Test F1 Score', color='salmon')
plt.xlabel('Classifiers')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison (Train vs Test)')
plt.xticks(index + bar_width / 2, clf_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting Test Accuracy for each classifier
plt.figure(figsize=(10, 6))
sns.barplot(x=clf_names, y=test_accuracies, palette='coolwarm')
plt.xlabel('Classifiers')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Classifiers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature importance for Random Forest (optional, if Random Forest is used)
rf_model = classifiers['Random Forest'].fit(X_train, y_train)
feature_importances = rf_model.feature_importances_

# Bar plot for feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns, palette='coolwarm')
plt.title('Feature Importances (Random Forest)')
plt.show()
