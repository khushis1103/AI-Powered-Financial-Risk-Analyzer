from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/random_forest.pkl")
explainer = shap.TreeExplainer(model)

class CreditInput(BaseModel):
    age: int
    sex: str
    housing: str
    saving_account: str
    job: int
    checking_account: str
    credit_amount: float
    duration: int
    purpose: str

def build_input_df(data: CreditInput) -> pd.DataFrame:
    input_data = {
        'Age': data.age,
        'Job': data.job,
        'Credit amount': data.credit_amount,
        'Duration': data.duration,
        'Sex_male': 1 if data.sex == "male" else 0,
        'Housing_own': 1 if data.housing == "own" else 0,
        'Housing_rent': 1 if data.housing == "rent" else 0,
        'Saving accounts_moderate': 1 if data.saving_account == "moderate" else 0,
        'Saving accounts_quite rich': 1 if data.saving_account == "quite rich" else 0,
        'Saving accounts_rich': 1 if data.saving_account == "rich" else 0,
        'Saving accounts_unknown': 1 if data.saving_account == "unknown" else 0,
        'Checking account_moderate': 1 if data.checking_account == "moderate" else 0,
        'Checking account_rich': 1 if data.checking_account == "rich" else 0,
        'Checking account_unknown': 1 if data.checking_account == "unknown" else 0,
        'Purpose_car': 1 if data.purpose == "car" else 0,
        'Purpose_domestic appliances': 1 if data.purpose == "domestic appliances" else 0,
        'Purpose_education': 1 if data.purpose == "education" else 0,
        'Purpose_furniture/equipment': 1 if data.purpose == "furniture/equipment" else 0,
        'Purpose_radio/TV': 1 if data.purpose == "radio/TV" else 0,
        'Purpose_repairs': 1 if data.purpose == "repairs" else 0,
        'Purpose_vacation/others': 1 if data.purpose == "vacation/others" else 0,
    }
    return pd.DataFrame([input_data])

@app.post("/predict")
def predict(data: CreditInput):
    input_df = build_input_df(data)
    prediction = int(model.predict(input_df)[0])
    probability = model.predict_proba(input_df)[0]
    good_prob = round(probability[1] * 100, 2)
    bad_prob = round(probability[0] * 100, 2)

    shap_values = explainer(input_df)
    customer_shap = shap_values.values[0, :, prediction]

    explanation = pd.DataFrame({
        "feature": input_df.columns,
        "shap_value": customer_shap
    })
    explanation["absolute"] = explanation["shap_value"].abs()
    explanation = explanation.sort_values(by="absolute", ascending=False).head(3)

    top_features = [
        {"feature": row["feature"], "shap_value": float(row["shap_value"])}
        for _, row in explanation.iterrows()
    ]

    return {
        "prediction": prediction,
        "good_risk_probability": good_prob,
        "bad_risk_probability": bad_prob,
        "top_features": top_features
    }