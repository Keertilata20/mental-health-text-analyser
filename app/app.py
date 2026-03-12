import streamlit as st
import joblib

# load trained model
model = joblib.load("model/depression_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.set_page_config(page_title="Mental Health Analyzer", page_icon="🧠")

st.title("🧠 Mental Health Text Analyzer")

st.write(
"""
This tool analyzes text and detects possible signs of depression.
It is **not a medical diagnosis**, but it can help identify emotional distress.
"""
)

user_input = st.text_area("How are you feeling today?")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector = vectorizer.transform([user_input])
        prob = model.predict_proba(vector)[0][1]

        if prob > 0.5:
            st.error("⚠️ Depression signal detected")
        else:
            st.success("✅ No strong depression signal detected")

        st.write(f"Confidence: {prob*100:.2f}%")

        if prob > 0.5:
            st.info(
            """
            If you're struggling, consider talking to someone you trust.
            Professional support can make a big difference.

            You are not alone. ❤️
            """
            )