#### Install all the dependences
  ```python
pip install -r requirement.txt
  ```
### Brief Description of the Code (Diabetes Prediction Model)

This Python code builds a machine learning pipeline to predict diabetes using the Pima Indians Diabetes Database. The code performs data preprocessing, visualization, and modeling using various classification algorithms from scikit-learn. Here's a detailed explanation of the steps and key components:

#### 1. **Importing Libraries**
The code starts by importing essential libraries:
- **numpy** and **pandas** are used for numerical computations and data manipulation, respectively.
- **matplotlib.pyplot** and **seaborn** are used for data visualization.
- **warnings** are suppressed to make the output cleaner.
- Various machine learning models are imported from **scikit-learn**, including logistic regression, decision trees, random forests, and others. 

The `SEED` variable is set for reproducibility purposes, ensuring that random operations yield the same result each time the code is run.

#### 2. **Loading the Dataset**
The dataset is loaded using `pd.read_csv()`. The dataset, `diabetes.csv`, contains information on various medical predictor variables such as glucose levels, blood pressure, and body mass index (BMI), with an `Outcome` column indicating whether a patient has diabetes (1) or not (0).

#### 3. **Descriptive Statistics and Correlation**
The first part of the code focuses on exploring the dataset:
- The descriptive statistics are printed using the `.describe()` method, which provides information like the mean, standard deviation, and min/max values of each column.
- A correlation matrix is computed using `.corr()`. This matrix shows the pairwise correlation between each feature and is visualized using a heatmap.

#### 4. **Data Visualization**
The code visualizes key aspects of the data through various plots:
- **Glucose Level Distribution**: A histogram is plotted to observe how glucose values are distributed in the dataset.
  
  ```python
  plt.figure(figsize=(10, 6))
  data['Glucose'].hist(bins=20, color='skyblue', alpha=0.7)
  plt.title('Distribution of Glucose')
  plt.xlabel('Glucose Level')
  plt.ylabel('Frequency')
  plt.show()
  ```

- **Outcome Count**: A countplot is created to show how many people in the dataset have diabetes (`Outcome = 1`) and how many don’t (`Outcome = 0`).
  
  ```python
  plt.figure(figsize=(8, 5))
  sns.countplot(x='Outcome', data=data, palette='viridis')
  plt.title('Count of Outcomes')
  plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
  plt.ylabel('Count')
  plt.show()
  ```

- **Correlation Heatmap**: A heatmap of the correlation matrix is plotted to better understand the relationship between features.
  
  ```python
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
  plt.title('Correlation Heatmap')
  plt.show()
  ```

- **Pie Chart for Outcome Proportion**: A pie chart visualizes the proportion of patients with and without diabetes, making it easier to comprehend class imbalance in the dataset.
  
  ```python
  outcome_counts = data['Outcome'].value_counts()
  plt.figure(figsize=(6, 6))
  plt.pie(outcome_counts, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral'])
  plt.title('Proportion of Outcomes')
  plt.show()
  ```

#### 5. **Handling Missing Data**
The dataset contains zero values in some features where zero is not a valid value (like blood pressure or BMI). These are treated as missing data and are replaced with the mean values for each class (diabetic or non-diabetic). This is done by grouping the dataset based on `Outcome` and computing the mean for each feature.

```python
def replace_zero(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    df.loc[(df[field] == 0) & (df[target] == 0), field] = mean_by_target.iloc[0][0]
    df.loc[(df[field] == 0) & (df[target] == 1), field] = mean_by_target.iloc[1][0]

for col in ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    replace_zero(data, col, 'Outcome')
```

This ensures that the zero entries are replaced with more plausible values, making the dataset cleaner for model training.

#### 6. **Outlier Removal**
The `TurkeyOutliers()` function is used to remove outliers from the dataset. The code uses Tukey's method, which identifies outliers based on interquartile ranges. Outliers are removed from the dataset to prevent them from skewing the model's results.

```python
def TurkeyOutliers(df, nameOfFeature, drop=False):
    valueOfFeature = df[nameOfFeature]
    Q1 = np.percentile(valueOfFeature, 25.)
    Q3 = np.percentile(valueOfFeature, 75.)
    step = (Q3 - Q1) * 1.5
    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    feature_outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values
    # Rest of the function continues...
```

#### 7. **Feature Selection and Splitting Data**
After preprocessing, the features (`X`) and target variable (`y`) are separated. The data is then split into training and testing sets using `train_test_split()`. This ensures that the model is evaluated on unseen data to prevent overfitting.

#### 8. **Training and Evaluating Models**
The code evaluates multiple machine learning algorithms including:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **AdaBoost**
- **Naive Bayes**

For each classifier, the code:
- Trains the model on the training set.
- Evaluates the model using both training and testing data.
- Computes the F1-score, which balances precision and recall, and accuracy.

```python
def train_clf(clf, X_train, y_train):
    return clf.fit(X_train, y_train)

def pred_clf(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target, y_pred, pos_label=1)

def accu_clf(clf, features, target):
    y_pred = clf.predict(features)
    return accuracy_score(target, y_pred)
```

For example, the `RandomForestClassifier()` might produce a higher F1 score on both training and testing sets, indicating that it performs well in predicting diabetes.

#### 9. **Cross-Validation**
The code uses K-fold cross-validation to assess the stability of the model's performance across different subsets of data. The `KFold()` function splits the dataset into 5 parts and evaluates the Random Forest model on different training/testing combinations.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
cv_f1_scores = []

for train_idx, test_idx in kf.split(X_clean):
    X_train_cv, X_test_cv = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
    y_train_cv, y_test_cv = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
    
    rf_clf.fit(X_train_cv, y_train_cv)
    y_pred_cv = rf_clf.predict(X_test_cv)
    cv_f1_scores.append(f1_score(y_test_cv, y_pred_cv))

print(f"Cross-validation F1 scores: {cv_f1_scores}")
print(f"Average F1 score: {np.mean(cv_f1_scores):.3f}")
```

This gives a better understanding of how the model will generalize to unseen data by averaging performance across multiple data splits.

---

### Additional Graphs

We can add more visualizations to improve our understanding of the dataset:

1. **Pair Plot**: Visualize relationships between all features.
   ```python
   sns.pairplot(data, hue='Outcome')
   plt.show()
   ```

2. **Box Plot for Feature Distributions**:
   ```python
   plt.figure(figsize=(12, 6))
   sns.boxplot(data=data, orient="h", palette="Set2")
   plt.title("Box Plot of Feature Distributions")
   plt.show()
   ```

3. **Bar Plot for Feature Importance** (After model training, especially RandomForest):
   ```python
   model = RandomForestClassifier(random_state=0)
   model.fit(X_train, y_train)
   feature_importances = model.feature_importances_

   plt.figure(figsize=(10, 6))
   sns.barplot(x=feature_importances, y=X_train.columns, palette='coolwarm')
   plt.title('Feature Importances')
   plt.show()
   ```

   **How it works:**
Data Preparation:

The dataset is loaded, and initial analysis is done with descriptive statistics, histograms, and correlation heatmaps.
Missing or erroneous values (like zeros in non-meaningful places) are handled.
Outliers in certain features (like insulin) are detected and removed.
Feature and Target Separation:

The feature matrix (X) includes columns such as glucose, BMI, etc.
The target variable (y) is the Outcome column, indicating whether a patient has diabetes (1) or not (0).
Model Training:

The dataset is split into training (80%) and testing (20%) sets.
Various classifiers are trained on the training data. These include:
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
AdaBoost
Naive Bayes
Prediction:

Each model makes predictions on both the training and testing datasets.
The F1 score and accuracy are calculated to evaluate how well the models perform in predicting whether a patient has diabetes.
Model Comparison:

The performance of each model is visualized using bar plots comparing F1 scores and accuracy.
What it predicts:
The model takes patient data (such as glucose, BMI, insulin levels, etc.) and predicts whether they are likely to have diabetes. This prediction can be used in medical diagnostics or for preventative health care.
