#Reasoning behind the defaults
#Age = 33 → median customer
#Job = 2 → most common category
#Credit Amount = 2320 → median loan amount
#Duration = 18 months → median duration
#Remaining categorical fields use the first option because there is no obvious median category without checking frequency counts

import streamlit as st
import pandas as pd
import joblib
import shap

st.set_page_config(page_title="AI Credit Risk Analyst", page_icon="💳")
st.title("AI-Powered Credit Risk Analyzer")

st.markdown("""
Predict whether a loan applicant is likely to be a **Good Risk** or **Bad Risk**
using a Random Forest Machine Learning model trained on the German Credit Dataset.
""")

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=19,
        max_value=75,
        value=33
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    housing = st.selectbox(
        "Housing",
        ["own", "rent", "free"]
    )

    saving_account = st.selectbox(
        "Saving Account",
        ["little", "moderate", "quite rich", "rich", "unknown"]
    )

with col2:

    job_option = st.selectbox(
        "Job Category",
        [
            "0 - Unskilled (Non-resident)",
            "1 - Unskilled (Resident)",
            "2 - Skilled",
            "3 - Highly Skilled"
        ],
        index=2
    )

    job = int(job_option[0])

    checking_account = st.selectbox(
        "Checking Account",
        ["little", "moderate", "rich", "unknown"]
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=250,
        max_value=18424,
        value=2320
    )

    duration = st.number_input(
        "Duration (Months)",
        min_value=4,
        max_value=72,
        value=18
    )

purpose = st.selectbox(
    "Purpose",
    [
        "radio/TV",
        "education",
        "furniture/equipment",
        "car",
        "repairs",
        "domestic appliances",
        "vacation/others"
    ]
)

predict = st.button("🔍 Predict Credit Risk", use_container_width=True)

if predict:
    model = joblib.load("models/random_forest.pkl")
    explainer = shap.TreeExplainer(model)

    input_data = {
        'Age': age,
        'Job': job,
        'Credit amount': credit_amount,
        'Duration': duration,

        'Sex_male': 1 if sex == "male" else 0,

        'Housing_own': 1 if housing == "own" else 0,
        'Housing_rent': 1 if housing == "rent" else 0,

        'Saving accounts_moderate': 1 if saving_account == "moderate" else 0,
        'Saving accounts_quite rich': 1 if saving_account == "quite rich" else 0,
        'Saving accounts_rich': 1 if saving_account == "rich" else 0,
        'Saving accounts_unknown': 1 if saving_account == "unknown" else 0,

        'Checking account_moderate': 1 if checking_account == "moderate" else 0,
        'Checking account_rich': 1 if checking_account == "rich" else 0,
        'Checking account_unknown': 1 if checking_account == "unknown" else 0,

        'Purpose_car': 1 if purpose == "car" else 0,
        'Purpose_domestic appliances': 1 if purpose == "domestic appliances" else 0,
        'Purpose_education': 1 if purpose == "education" else 0,
        'Purpose_furniture/equipment': 1 if purpose == "furniture/equipment" else 0,
        'Purpose_radio/TV': 1 if purpose == "radio/TV" else 0,
        'Purpose_repairs': 1 if purpose == "repairs" else 0,
        'Purpose_vacation/others': 1 if purpose == "vacation/others" else 0
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    good_prob = probability[1] * 100
    bad_prob = probability[0] * 100
    
    # SHAP explanation
    shap_values = explainer(input_df)
    customer_shap = shap_values.values[0, :, prediction]
    explanation = pd.DataFrame({
    "Feature": input_df.columns,
    "SHAP Value": customer_shap})
    explanation["Absolute"] = explanation["SHAP Value"].abs()
    explanation = explanation.sort_values(by="Absolute", ascending=False)
    
    st.markdown("---")
    st.subheader("📊 Prediction Summary")
    def interpret(row):
        if prediction == 1:
            if row["SHAP Value"]>0:
                return "🟢 Increased likelihood of Good Risk"
            else:
                return "🔴 Increased likelihood of Bad Risk"
        
        else:
            if row["SHAP Value"] > 0:
                return "🔴 Increased likelihood of Bad Risk"
            else:
                return "🟢 Increased likelihood of Good Risk"
    explanation["Contribution"] = explanation.apply(interpret, axis=1)
    explanation["Contribution"] = explanation.apply(interpret, axis=1)
    st.markdown("---")
    st.subheader("🔍 Top Factors Influencing This Prediction")
    st.dataframe(explanation[["Feature", "Contribution"]].head(5), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Good Risk Probability", f"{good_prob:.2f}%")
        st.progress(good_prob / 100)
    
    with col2:
        st.metric("Bad Risk Probability", f"{bad_prob:.2f}%")
        st.progress(bad_prob / 100)
    
    st.markdown("### Risk Level")
    if good_prob >= 95:
        st.success("🟢 Very Low Risk")
    elif good_prob >= 80:
        st.success("🟢 Low Risk")
    elif good_prob >= 60:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")
        
    st.markdown("### 💡 Recommendation")
    if prediction == 1:
        st.success(
            "The applicant appears to have a low probability of default. "
        )
    else:
        st.error(
            "The applicant exhibits characteristics associated with higher credit risk. "
            "Additional financial verification or collateral may be required before approval."
        )
  