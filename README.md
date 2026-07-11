#  AI-Powered Credit Risk Analyzer

Predicts loan applicant risk (Good/Bad) using a Random Forest model trained on the German Credit Dataset, with SHAP-based explainability. Built as a full stack application — FastAPI backend, HTML/CSS/JS frontend, containerized with Docker.

##  Live Demo
- **App:** https://khushis1103.github.io/AI-Powered-Financial-Risk-Analyzer/
- **API docs:** https://ai-powered-financial-risk-analyzer-k7s2.onrender.com/


##  Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python |
| ML Model | scikit-learn (Random Forest) |
| Explainability | SHAP |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker, Render |

## How it works

The frontend collects applicant details through a form and sends them to a `/predict` endpoint. The backend encodes the input, runs it through the trained model, and returns a prediction along with probability scores and the top three features driving that decision (via SHAP)

##  Features

- Predicts credit risk (Good/Bad) with probability scores
- SHAP-based explanation of top 3 contributing factors per prediction
- Clean, responsive web interface
- REST API (`/predict`) usable independently of the frontend
- Fully containerized for consistent deployment

##  Dataset

German Credit Dataset — 1000 records, applicant demographic and financial features (age, credit amount, duration, housing, savings, checking account status, purpose).

##  Architecture

User (Browser)

↓ fetch()

HTML/CSS/JS Frontend

↓ POST /predict

FastAPI Backend

↓

Random Forest Model + SHAP Explainer

↓

JSON Response (prediction + probabilities + top features)

##  Running Locally

```bash
# Clone the repo
git clone https://github.com/khushis1103/AI-Powered-Financial-Risk-Analyzer.git
cd AI-Powered-Financial-Risk-Analyzer

## Screenshots

![App Screenshot](screenshots/ Prediction Demo.png)

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload

# Open index.html in your browser
```

##  Running with Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```


## Project structure
├── main.py              # FastAPI backend

├── index.html            # Frontend

├── models/
     
      └── random_forest.pkl # Trained model

├── requirements.txt

├── Dockerfile

└── README.md


## What's next

- Persisting prediction history
- Downloadable PDF reports
- A small model comparison view (Random Forest vs. other classifiers)

---
Built by Khushi Sahu
