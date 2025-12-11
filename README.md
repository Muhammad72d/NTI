# 📊 Dataset

File: Healthcare-Diabetes.csv

Description: A healthcare dataset containing patient diagnostic features used to predict whether an individual is diabetic.

Features include (example fields):

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Target Variable:

Outcome — 1 for diabetic, 0 for non-diabetic.

(If your dataset includes different column names, I can adjust the README accordingly.)

# 🎯 Project Goals

Conduct preprocessing and exploratory data analysis (EDA).

Train multiple machine-learning models:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Support Vector Machine (SVM)

Gradient Boosting / XGBoost (optional)

Evaluate each model using:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Visualize results (confusion matrices, ROC curves, performance comparison).

Summarize insights to determine the most suitable model for this dataset.

# 🧪 Workflow Overview
1. Data Preprocessing

Handling missing values

Scaling/normalizing features

Splitting into training and testing sets

2. Model Training

Building and fitting each algorithm

Hyperparameter tuning (GridSearchCV or manual tuning if used)

3. Model Evaluation

Compute classification metrics

Plot confusion matrices

Plot ROC curves

Aggregate results into a comparison table

# 📈 Summary of Findings

The notebook provides:

Performance comparison of classical ML algorithms

Insights into how preprocessing affects model quality

Identification of the most reliable model for predicting diabetes from the Healthcare-Diabetes.csv dataset

(Your exact results may vary depending on tuning and random state.)

# 📊 Model Performance Results

The following table summarises the performance of all evaluated models on the diabetes classification task using Healthcare-Diabetes.csv:

Classification Report Summary
Model	Accuracy	Precision (Class 1)	Recall (Class 1)	F1 (Class 1)
Naive Bayes	0.74	0.65	0.53	0.58
Logistic Regression	0.78	0.77	0.51	0.61
SVC	0.78	0.75	0.51	0.61
Decision Tree	0.94	0.90	0.92	0.91
KNN	0.95	0.98	0.87	0.92
Random Forest	0.94	0.99	0.83	0.90
Key Observations

Best Overall Performance:

KNN achieved the highest accuracy (0.95) and the best F1-score for the positive class (0.92).

This indicates excellent balance between precision and recall.

Strong Tree-Based Models:

Decision Tree and Random Forest both exceeded 0.94 accuracy, with strong recall and precision values.

Random Forest shows very high precision for diabetic cases (0.99), but slightly lower recall.

Linear Models (Logistic Regression, SVC):

Achieved moderate accuracy (0.78).

Both struggled with recall for the diabetic class (0.51), meaning they missed many true positive cases.

Naive Bayes:

Lowest performance overall (accuracy 0.74).

Significant difficulty identifying diabetic cases (recall 0.53).

Conclusion

KNN is the best-performing algorithm for this dataset, followed by Decision Tree and Random Forest.

Simpler linear classifiers (Logistic Regression, SVC) underperformed due to the non-linear nature of the dataset.

Naive Bayes performed the weakest, likely due to violated feature-independence assumptions.

# 🛠️ Requirements

Install required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost   # optional

# ▶️ How to Run
* Clone the repository*
git clone https://github.com/Muhammad72d/NTI/blob/main/Team%204%20-Final%20project.ipynb
* Navigate to folder*
cd diabetes-classification-comparison

* Launch notebook *
jupyter notebook diabetes_classification_comparison.ipynb

📚 References

Healthcare-Diabetes.csv (dataset source)

Scikit-learn documentation

Classic diabetes prediction datasets used in literaturement with other classifiers or feature engineering techniques.
