import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 AI Fake News Detector")

st.sidebar.header("About")
st.sidebar.write("""
Detect whether a news article is Fake or Real using Machine Learning.
Accuracy: 98%
""")

news = st.text_area("Enter News Article")

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        x = vectorizer.transform([news])

        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0]

        fake_prob = prob[0] * 100
        real_prob = prob[1] * 100

        if pred == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        st.subheader("Confidence")

        st.write(f"Fake Probability: {fake_prob:.2f}%")
        st.progress(int(fake_prob))

        st.write(f"Real Probability: {real_prob:.2f}%")
        st.progress(int(real_prob))