# Diabetes Prediction System

A machine learning-based application to predict the likelihood of diabetes using health-related data. It uses the Pima Indians Diabetes Dataset and includes data cleaning, feature scaling, and classification modeling.

#Features

📊 Handles missing values by replacing zeros with NaN in specific medical columns.

⚖️ Standardizes features using StandardScaler.

🤖 Trains a machine learning model (e.g., Logistic Regression, Random Forest, etc.)

🧪 Evaluates model performance using metrics like accuracy, precision, recall, and confusion matrix.

🧮 Predicts whether a person is diabetic based on input features.

# Dataset

The dataset used is the Pima Indians Diabetes Dataset, which contains the following features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Outcome (0 = Non-diabetic, 1 = Diabetic)

# Data Preprocessing

Load Dataset
url = "/content/drive/MyDrive/diabetes.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=0, names=columns)

Replaces invalid zero values in clinical features with NaN.

Handles missing values using imputation (mean/median strategy).

Applies feature scaling using StandardScaler.

# Model Training
Example with Logistic Regression:
 
 from sklearn.linear_model import LogisticRegression
 
 from sklearn.model_selection import train_test_split

 X = data.drop('Outcome', axis=1)

y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# Prediction

Make predictions for new users:

import pandas as pd
sample = pd.DataFrame([[3, 120, 70, 25, 100, 32.0, 0.45, 30]], columns=X.columns)
sample_scaled = scaler.transform(sample)
print(model.predict(sample_scaled))  # Output: [0] or [1]

# Technologies Used
Python 3.11+

NumPy

Pandas

scikit-learn

Matplotlib / Seaborn (optional for visualization)

# Project Structure
├── diabetes_prediction.ipynb
├── diabetes.csv
├── README.md
├── requirements.txt
└── models/
    └── trained_model.pkl

# How to Run

Clone the repo:

git clone https://github.com/yourusername/diabetes-prediction.git

cd diabetes-prediction

Install dependencies:

pip install -r requirements.txt

Run the notebook or script:

jupyter notebook diabetes_prediction.ipynb
