# 💳 AI-Powered Financial Risk Analyzer

An end-to-end Machine Learning application that predicts whether a loan applicant is a **Good Risk** or **Bad Risk** customer based on their financial and demographic information.

The project covers the full ML lifecycle — data preprocessing, feature engineering, model training, explainable AI (SHAP), model serialization, and an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📑 Table of Contents
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

## 🚀 Features

- 🎯 Credit Risk Prediction using Machine Learning
- 🖥️ Interactive Streamlit Web App
- 🧹 Data Cleaning & Preprocessing
- 🏗️ Feature Engineering with One-Hot Encoding
- 🌲 Random Forest Classification
- 📊 Probability Prediction
- 🔍 Explainable AI using SHAP
- 💾 Model Serialization using Joblib

---

## 📊 Dataset

**Dataset Used:** German Credit Risk Dataset

- 1,000 customer records
- 9 input features
- Binary Classification target:
  - ✅ Good Risk
  - ❌ Bad Risk

---

## 🛠 Tech Stack

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

## 📈 Machine Learning Pipeline

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

## 📂 Project Structure

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

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/AI-Powered-Financial-Risk-Analyzer.git
cd AI-Powered-Financial-Risk-Analyzer
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

The app will launch in your default browser at `http://localhost:8501`.

---

## 📸 Application Screenshots

### Home Page
*(Add screenshot here)*

### Prediction Result
*(Add screenshot here)*

### Bad Risk Prediction
*(Add screenshot here)*

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning
- [ ] Model comparison (XGBoost, LightGBM)
- [ ] Customer Risk Dashboard
- [ ] PDF Report Generation
- [ ] Cloud Deployment
- [ ] Real-time API Integration

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify it for your own work.

---

## 👨‍💻 Author

**Khushi**

If you found this project useful, consider giving it a ⭐ on GitHub!
