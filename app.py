import streamlit as st
import pandas as pd
import joblib
import shap

# Load the trained model and the exact feature columns used during training
rf = joblib.load("models/random_forest.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Build the SHAP explainer once and reuse it, instead of rebuilding it
# every time the Predict button is clicked (it's slow to build)
@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(rf)

explainer = get_explainer()

st.title("AI Credit Risk Analyzer")
st.write("Enter customer details below to predict loan repayment risk.")

# ---- Input fields (same fields and options as your original HTML form) ----

age = st.number_input("Age", min_value=18, max_value=100, value=18)

sex = st.selectbox("Sex", ["male", "female"])

housing = st.selectbox("Housing", ["own", "rent", "free"])

saving_account = st.selectbox(
    "Saving Account",
    ["little", "moderate", "quite rich", "rich", "unknown"]
)

job_options = {
    "Unskilled (Non-resident)": 0,
    "Unskilled (Resident)": 1,
    "Skilled": 2,
    "Highly Skilled": 3
}
job_label = st.selectbox("Job Category", list(job_options.keys()), index=2)
job = job_options[job_label]

checking_account = st.selectbox(
    "Checking Account",
    ["little", "moderate", "rich", "unknown"]
)

credit_amount = st.number_input("Credit Amount", min_value=0, value=2320)

duration = st.number_input("Duration (Months)", min_value=1, value=18)

purpose = st.selectbox(
    "Purpose",
    ["radio/TV", "education", "furniture/equipment", "car",
     "repairs", "domestic appliances", "vacation/others"]
)

# ---- Predict button ----

if st.button("Predict"):

    # Put the single customer's inputs into a one-row dataframe,
    # same column names as the original dataset
    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_account,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])

    # One-hot encode this row the same way the training data was encoded
    input_encoded = pd.get_dummies(input_data)

    # Line up columns exactly with what the model was trained on.
    # Any column the model expects but this row doesn't have gets filled with 0.
    input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = rf.predict(input_final)[0]
    probabilities = rf.predict_proba(input_final)[0]

    good_risk_probability = round(probabilities[1] * 100, 2)
    bad_risk_probability = round(probabilities[0] * 100, 2)

    if prediction == 1:
        st.success("Prediction: Good Risk")
    else:
        st.error("Prediction: Bad Risk")

    st.write("Good Risk Probability:", good_risk_probability, "%")
    st.write("Bad Risk Probability:", bad_risk_probability, "%")

    # ---- SHAP explanation for this one prediction ----
    # Same logic as the notebook's local explainability section:
    # get SHAP values for this customer, pick the class that was predicted,
    # and show the features that pushed the prediction most.

    shap_values = explainer(input_final)
    customer_shap = shap_values.values[0, :, prediction]

    explanation = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": customer_shap
    })
    explanation["Absolute"] = explanation["SHAP Value"].abs()
    explanation = explanation.sort_values("Absolute", ascending=False)

    top_features = explanation.head(5)

    st.write("### Top Contributing Factors")
    for index, row in top_features.iterrows():
        if row["SHAP Value"] > 0:
            direction = "increases"
        else:
            direction = "decreases"
        st.write("-", row["Feature"], "(" + direction, "good-risk likelihood)")
