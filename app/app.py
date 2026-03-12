import streamlit as st
import joblib

# load trained model
model = joblib.load("model/depression_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("Mental Health Text Analyzer")

user_input = st.text_area("Enter text to analyze")

if st.button("Analyze"):

    vector = vectorizer.transform([user_input])

    prediction = model.predict(vector)
    prob = model.predict_proba(vector)[0][1]

    if prediction[0] == 1:
        st.error("Depression signal detected")
    else:
        st.success("No depression signal detected")

    st.write(f"Confidence: {prob*100:.2f}%")