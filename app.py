#Reasoning behind the defaults
#Age = 33 → median customer
#Job = 2 → most common category
#Credit Amount = 2320 → median loan amount
#Duration = 18 months → median duration
#Remaining categorical fields use the first option because there is no obvious median category without checking frequency counts

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Credit Risk Analyst", page_icon="💳")

st.title("💳 AI Credit Risk Analyst")
st.write("Enter customer details to assess credit risk.")
st.subheader("Customer Information")

age = st.number_input("Age",min_value=19, max_value=75, value=33) #value tells default value that would show on opening app

job = st.selectbox(
    "Job Category",
    options=[
        "0 - Unskilled (Non-resident)",
        "1 - Unskilled (Resident)",
        "2 - Skilled",
        "3 - Highly Skilled"
    ]
)

job = int(job[0])   # Extracts 0,1,2,3 for the model

sex = st.selectbox("Sex",["male", "female"])

housing = st.selectbox("Housing",["own", "rent", "free"])

saving_account = st.selectbox("Saving Account",["little", "moderate", "quite rich", "rich", "unknown"])

checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])

credit_amount = st.number_input("Credit Amount", min_value=250, max_value=18424, value=2320)

duration = st.number_input("Duration (Months)",min_value=4, max_value=72,value=18)

purpose = st.selectbox("Purpose",["radio/TV","education", "furniture/equipment","car","repairs","domestic appliances","vacation/others"
    ])


if st.button("Predict Risk"):

  

    model = joblib.load("models/random_forest.pkl")

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

    if prediction == 1:
        st.success("✅ Good Risk Customer")
    else:
        st.error("⚠️ Bad Risk Customer")

    st.write(f"Good Risk Probability: {probability[1]*100:.2f}%")
    st.write(f"Bad Risk Probability: {probability[0]*100:.2f}%")
