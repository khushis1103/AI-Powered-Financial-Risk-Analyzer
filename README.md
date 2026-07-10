# 💳 AI-Powered Financial Risk Analyzer

An end-to-end Machine Learning application that predicts whether a loan applicant is a **Good Risk** or **Bad Risk** customer based on their financial and demographic information.

The project covers the full ML lifecycle — data preprocessing, feature engineering, model training, explainable AI (SHAP), model serialization, and an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Table of Contents
- [Features](#-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Performance](#-model-performance)
- [Feature Importance](#-feature-importance)
- [Project Structure](#-project-structure)
- [Installation & Setup](#️-installation--setup)
- [Run Locally](#️-run-locally)
- [Application Screenshots](#-application-screenshots)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## Features

- Credit Risk Prediction using Machine Learning
- Interactive Streamlit Web App
- Data Cleaning & Preprocessing
- Feature Engineering with One-Hot Encoding
- Random Forest Classification
- Probability Prediction
- Explainable AI using SHAP
- Model Serialization using Joblib

---

## Dataset

**Dataset Used:** German Credit Risk Dataset

- 1,000 customer records
- 9 input features
- Binary Classification target:
  - ✅ Good Risk
  - ❌ Bad Risk

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Explainability | SHAP |
| Web App | Streamlit |
| Visualization | Matplotlib |
| Model Persistence | Joblib |

---

## Machine Learning Pipeline

1. Data Cleaning
2. Missing Value Handling
3. Feature Engineering
4. One-Hot Encoding
5. Train-Test Split
6. Random Forest Model Training
7. Model Evaluation
8. SHAP Explainability
9. Streamlit Deployment

---

## 📊 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **74%** |

**Classification Report**

| Class | Precision | Recall | F1-score |
|--------|-----------|---------|-----------|
| Bad Risk | 0.59 | 0.41 | 0.48 |
| Good Risk | 0.78 | 0.88 | 0.83 |

---

## 🔍 Feature Importance

**Top Important Features:**
- Credit Amount
- Age
- Duration
- Checking Account Status
- Job Category

SHAP was used to explain both global feature importance and individual customer predictions, offering transparency into how the model arrives at each decision.

---

## Project Structure

```
AI-Powered-Financial-Risk-Analyzer
│
├── app.py                 # Streamlit application entry point
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── data/                  # Raw and processed datasets
├── models/                # Serialized trained models (.pkl)
├── notebooks/             # EDA and experimentation notebooks
└── screenshots/           # App screenshots for documentation
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

The app will launch in your default browser at `http://localhost:8501`.

---

## Application Screenshots

### Home Page

![Home Page](screenshots/Home%20Page.png)

### Good Risk Input

![Good Risk Input](screenshots/Good%20risk%20input.png)

### Good Risk Prediction

![Good Risk Prediction](screenshots/Good%20risk%20prediction.png)

### Bad Risk Input

![Bad Risk Input](screenshots/Bad%20risk%20input.png)

### Bad Risk Prediction

![Bad Risk Prediction](screenshots/Bad%20risk%20prediction.png)
![Bad Risk Input](screenshots/bad_risk_input.png)

---

## Future Improvements

- [ ] Hyperparameter tuning
- [ ] Model comparison (XGBoost, LightGBM)
- [ ] Real-time API Integration

---

## Author

**Khushi**
