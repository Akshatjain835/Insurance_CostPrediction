# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# scaler=joblib.load("../scaler.pkl")
# le_gender=joblib.load("../encoders/label_encoder_gender.pkl")
# le_diabetic=joblib.load("../encoders/label_encoder_diabetic.pkl")
# le_smoker=joblib.load("../encoders/label_encoder_smoker.pkl")

# model=joblib.load("../models/best_model.pkl")

# st.set_page_config(page_title="Insurance Claim Prediction",layout="centered")
# st.title("Health Insurance Prediction App")
# st.write("Enter the details below to estimate your insurance amount. ")

# with st.form("input_form"):
#     col1,col2=st.columns(2)
#     with col1:
#         age=st.number_input("Age",min_value=0,max_value=100,value=30)
#         bmi=st.number_input("BMI",min_value=10.0,max_value=60.0,value=25.0)
#         children=st.number_input("Number of children",min_value=0,max_value=8,value=0)

    
#     with col2:
#         bloodpressure=st.number_input("Blood Pressure",min_value=60,max_value=200,value=120)
#         gender=st.selectbox("Gender",options=le_gender.classes_)
#         diabetic=st.selectbox("Diabetic",options=le_diabetic.classes_)
#         smoker=st.selectbox("Smoker",options=le_smoker.classes_)

#     submitted=st.form_submit_button("Predict payment")


# if submitted:

#     input_data=pd.DataFrame({
#         "age":[age],
#         "gender":[gender],
#         "bmi":[bmi],
#         "bloodpressure":[bloodpressure],
#         "diabetic":[diabetic],
#         "children":[children],
#         "smoker":[smoker]

#     })

#     input_data["gender"]=le_gender.transform(input_data["gender"])
#     input_data["diabetic"]=le_diabetic.transform(input_data["diabetic"])
#     input_data["smoker"]=le_smoker.transform(input_data["smoker"])

#     num_cols=["age","bmi","bloodpressure","children"]
#     input_data[num_cols]=scaler.transform(input_data[num_cols])

#     prediction=model.predict(input_data)[0]

#     st.success(f"**Estimated Insurance Payment Amount** ${prediction:,.2f}")


import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Insurance Claim Prediction", layout="centered")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
le_gender = joblib.load(os.path.join(BASE_DIR, "encoders", "label_encoder_gender.pkl"))
le_diabetic = joblib.load(os.path.join(BASE_DIR, "encoders", "label_encoder_diabetic.pkl"))
le_smoker = joblib.load(os.path.join(BASE_DIR, "encoders", "label_encoder_smoker.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))

st.title("Health Insurance Prediction App")
st.write("Enter the details below to estimate your insurance amount.")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        children = st.number_input("Number of children", 0, 8, 0)

    with col2:
        bloodpressure = st.number_input("Blood Pressure", 60, 200, 120)
        gender = st.selectbox("Gender", le_gender.classes_)
        diabetic = st.selectbox("Diabetic", le_diabetic.classes_)
        smoker = st.selectbox("Smoker", le_smoker.classes_)

    submitted = st.form_submit_button("Predict payment")

if submitted:
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic],
        "children": [children],
        "smoker": [smoker]
    })

    input_data["gender"] = le_gender.transform(input_data["gender"])
    input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])

    num_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    
    # st.write(type(model))
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Insurance Payment Amount: ${prediction:,.2f}")
