import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("./models/RandomForestClassifier()_final.pkl")  
scaler = joblib.load("./models/scaler.pkl")  

st.title("Prédiction du risque de diabète")
st.write("Entrez vos informations pour estimer votre risque de diabète.")

age = st.number_input("Âge", min_value=1, max_value=120, value=30)
pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=50, max_value=300, value=120)
blood_pressure = st.number_input("Pression artérielle", min_value=40, max_value=200, value=70)
skin_thickness = st.number_input("Épaisseur du pli cutané", min_value=5, max_value=100, value=20)
insulin = st.number_input("Insuline", min_value=0, max_value=1000, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)

user_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})
user_data["DiabetesPedigreeFunction"] = np.log(user_data["DiabetesPedigreeFunction"])
user_data["Insulin"] = np.log(user_data["Insulin"])
user_data["Pregnancies"] = np.log(user_data["Pregnancies"])
user_data["Age"] = np.log1p(user_data["Age"])

if st.button("Prédire le risque"):
    try:
        user_scaled = scaler.transform(user_data)

        
        prediction = model.predict(user_scaled)
        proba = model.predict_proba(user_scaled)[0]

        

        if prediction[0] == 0:
            st.error(f" Risque élevé de diabète ({proba[1]*100:.2f}%)")
        else:
            st.success(f" Risque faible de diabète ({proba[0]*100:.2f}%)")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")


st.write("---")
st.write("⚡ Cette application est un outil éducatif et ne remplace pas un avis médical.")
