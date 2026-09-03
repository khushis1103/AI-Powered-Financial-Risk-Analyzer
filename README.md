# AI-Powered Financial Risk Analyzer

An explainable machine learning application that predicts whether a loan applicant represents a **Good Risk** or **Bad Risk** based on demographic and financial information.

The project covers the full workflow — data preprocessing, model training, evaluation, model interpretation with SHAP, and deployment through an interactive Streamlit application.

---

## Project Overview

Credit risk assessment involves determining whether a loan applicant is likely to represent a lower or higher repayment risk.

This project uses the **German Credit Dataset** to build a binary classification system that predicts:

- **Good Risk**
- **Bad Risk**

Two classification approaches were evaluated:

- Logistic Regression
- Random Forest

Random Forest was selected for the deployed application and interpreted using **SHAP (SHapley Additive exPlanations)**.

---

## Dataset

**Dataset:** German Credit Dataset

- 1,000 customer records
- 9 input features
- Binary target variable: `Risk`

### Input Features

- Age
- Sex
- Job
- Housing
- Saving accounts
- Checking account
- Credit amount
- Duration
- Purpose

The target variable contains two classes: `good` and `bad`. For modeling, these were encoded as `good → 1`, `bad → 0`.

---

## Machine Learning Pipeline

```
German Credit Dataset
        ↓
Data Cleaning
        ↓
Missing Value Handling
        ↓
Categorical Encoding
        ↓
Train-Test Split
        ↓
Logistic Regression  →  Random Forest
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Save Trained Model
        ↓
Streamlit Application
```

---

## Preprocessing

The dataset was prepared by:

- Removing the unnecessary index column
- Handling missing categorical values using an `unknown` category rather than dropping or imputing them, since missingness itself carries information in banking data
- Encoding categorical variables using one-hot encoding
- Splitting the data into training and test sets
- Scaling numerical features for Logistic Regression

The final encoded feature set contains 21 features.

---

## Models

### Logistic Regression

Used as a baseline classification model.

**Test accuracy:** 75%

### Random Forest

A Random Forest classifier with 200 trees was trained for comparison and used as the deployed model.

**Test accuracy:** 74%

Although Logistic Regression achieved slightly higher accuracy on this particular test split, Random Forest provided better recall for the Bad Risk class and enabled tree-based SHAP explanations — which is why it was the model selected for deployment despite the marginally lower accuracy.

### Model Comparison

| Model | Accuracy | Bad Risk Recall |
|---|---|---|
| Logistic Regression | 75% | 36% |
| Random Forest | 74% | 41% |

### Random Forest Classification Report

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Bad Risk | 0.59 | 0.41 | 0.48 |
| Good Risk | 0.78 | 0.88 | 0.83 |

Accuracy alone is not sufficient for evaluating a credit-risk model. In this application, recall for the Bad Risk class is particularly important, since missing a potentially risky applicant carries a higher business cost than incorrectly flagging a lower-risk one.

---

## Model Explainability

The project uses SHAP to understand the factors contributing to individual Random Forest predictions.

### Global Feature Importance

The most important features in the Random Forest model included:

- Credit Amount
- Age
- Duration
- Checking Account status
- Job Category

### Local Explainability

For an individual applicant, SHAP values are used to identify the features that contribute most strongly toward or away from the model's predicted class — providing a more interpretable view than a bare Good/Bad output.

---

## Streamlit Application

The trained Random Forest model and the exact feature columns used during training are saved using Joblib.

The Streamlit application allows a user to:

- Enter applicant information
- Generate a Good Risk / Bad Risk prediction
- View the predicted class probabilities
- View the top contributing factors from SHAP

The application re-aligns input features with the feature columns used during model training to ensure consistent model input.

---

## Project Structure

```
AI-Powered-Financial-Risk-Analyzer/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── german_credit_data.csv
│
├── models/
│   ├── random_forest.pkl
│   └── feature_columns.pkl
│
└── notebooks/
    └── credit_risk_analysis.ipynb
```

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Models | Logistic Regression, Random Forest |
| Explainability | SHAP |
| Visualization | Matplotlib |
| Model Persistence | Joblib |
| Deployment / UI | Streamlit |

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/khushis1103/AI-Powered-Financial-Risk-Analyzer.git
cd AI-Powered-Financial-Risk-Analyzer
```

**2. Create a virtual environment**
```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Run the Streamlit application**
```bash
streamlit run app.py
```

The application will open in the browser at `http://localhost:8501`.

---

## Limitations

- The dataset contains only 1,000 observations
- Model evaluation is based on a single train-test split
- Extensive hyperparameter tuning and cross-validation were not performed
- The dataset is a standard educational credit-risk dataset and should not be treated as a production lending system

---

## Future Improvements

- Stratified cross-validation for more robust model evaluation
- Hyperparameter tuning for Random Forest
- ROC-AUC and Precision-Recall AUC analysis
- Cost-sensitive evaluation of credit-risk errors
- Comparison with boosting-based models such as XGBoost
- Improved preprocessing using a scikit-learn Pipeline
- Monitoring model performance on new data

---

## Author

**Khushi Sahu**
M.Sc. Computer Science
University of Delhi
