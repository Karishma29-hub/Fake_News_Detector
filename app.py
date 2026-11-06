
import re
from pathlib import Path

import joblib
import streamlit as st


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f4f7ff, #edf2ff);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #172554, #4338ca, #6366f1);
    padding: 35px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(67, 56, 202, 0.2);
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    color: #e0e7ff;
    font-size: 16px;
}

.section-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e0e7ff;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.07);
    margin-bottom: 18px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #dbeafe;
    text-align: center;
    box-shadow: 0 5px 18px rgba(30, 64, 175, 0.08);
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e0e7ff;
    text-align: center;
}

.metric-card h3 {
    color: #334155;
    font-size: 17px;
}

.metric-card p {
    font-size: 28px;
    font-weight: bold;
    margin: 5px;
}

div.stButton > button {
    border-radius: 10px;
    border: 1px solid #c7d2fe;
    background: white;
    color: #3730a3;
    font-weight: 600;
    padding: 10px;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #eef2ff;
    border-color: #6366f1;
    color: #312e81;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 12px;
    border: 1px solid #c7d2fe;
}

h2, h3 {
    color: #1e3a8a;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


if "history" not in st.session_state:
    st.session_state.history = []

if "news_text_input" not in st.session_state:
    st.session_state.news_text_input = ""

if "show_paste_message" not in st.session_state:
    st.session_state.show_paste_message = False


real_news = (
    "The Reserve Bank of India announced that it will keep "
    "the repo rate unchanged after its monetary policy meeting. "
    "The central bank said its decision was based on inflation "
    "trends and the current economic outlook."
)

fake_news = (
    "Scientists have announced that drinking one glass of a "
    "secret herbal mixture every day makes people live for "
    "200 years. The government has reportedly ordered every "
    "hospital to provide this miracle drink to all citizens."
)


def load_article(article):
    st.session_state.news_text_input = article
    st.session_state.show_paste_message = False
    st.session_state.pop("last_prediction", None)


def paste_own_article():
    st.session_state.news_text_input = ""
    st.session_state.show_paste_message = True
    st.session_state.pop("last_prediction", None)


def clear_article():
    st.session_state.news_text_input = ""
    st.session_state.show_paste_message = False
    st.session_state.pop("last_prediction", None)
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


st.markdown("""
<div class="hero">
    <h1>📰 Fake News Detector</h1>
    <p>
        Detect Fake and Real News using Natural Language
        Processing and Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)


with st.expander("🤖 About the Model"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ML Algorithm", "Logistic Regression")

    with col2:
        st.metric("Feature Extraction", "TF-IDF")

    with col3:
        st.metric("Test Accuracy", "98.50%")

    st.write("**Training Articles:** 35,918")
    st.write("**Testing Articles:** 8,980")

st.markdown("### 🧪 Choose Your News Input")

st.write(
    "Select a sample article or paste your own news article."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.button(
        "✅ Real News",
        use_container_width=True,
        on_click=load_article,
        args=(real_news,)
    )

with col2:
    st.button(
        "❌ Fake News",
        use_container_width=True,
        on_click=load_article,
        args=(fake_news,)
    )

with col3:
    st.button(
        "📝 Paste Your Own",
        use_container_width=True,
        on_click=paste_own_article
    )


if st.session_state.show_paste_message:
    st.info("You can now paste your own news article below.")


st.markdown("### 📰 Enter News Article")

news_text = st.text_area(
    "Paste your article here:",
    height=220,
    placeholder="Paste or type your news article here...",
    key="news_text_input"
)

col1, col2 = st.columns(2)

with col1:
    analyze_button = st.button(
        "🔍 Analyze News",
        use_container_width=True
    )

with col2:
    st.button(
        "🗑️ Clear Article",
        use_container_width=True,
        on_click=clear_article
    )



if analyze_button:

    if not news_text.strip():
        st.warning("⚠️ Please enter a news article first.")

    else:

        cleaned_text = clean_text(news_text)

        if not cleaned_text:
            st.warning("Please enter meaningful news text.")

        else:

            try:
                text_tfidf = vectorizer.transform([cleaned_text])

                prediction = model.predict(text_tfidf)[0]

                probabilities = model.predict_proba(text_tfidf)[0]

                classes = list(model.classes_)

                fake_index = classes.index(0)
                real_index = classes.index(1)

                fake_probability = probabilities[fake_index] * 100
                real_probability = probabilities[real_index] * 100

                confidence = max(
                    fake_probability,
                    real_probability
                )

                if prediction == 0:
                    result_text = "FAKE NEWS"
                else:
                    result_text = "REAL NEWS"

                st.session_state.last_prediction = {
                    "prediction": prediction,
                    "fake_probability": fake_probability,
                    "real_probability": real_probability,
                    "confidence": confidence,
                    "result_text": result_text
                }

                st.session_state.history.append({
                    "Article": news_text,
                    "Prediction": result_text,
                    "Confidence": f"{confidence:.2f}%"
                })

            except Exception as e:
                st.error(f"Prediction error: {e}")
                

if "last_prediction" in st.session_state:

    result = st.session_state.last_prediction

    st.divider()

    st.markdown("### 📊 Prediction Result")

    if result["prediction"] == 0:

        st.markdown("""
        <div class="result-card">
            <h2 style="color:#dc2626;">❌ FAKE NEWS</h2>
            <p style="color:#64748b;">
                The model identified this article as Fake News.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">
            <h2 style="color:#16a34a;">✅ REAL NEWS</h2>
            <p style="color:#64748b;">
                The model identified this article as Real News.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("### 🎯 Prediction Confidence")

    st.progress(result["confidence"] / 100)

    st.markdown(
        f"**Confidence: {result['confidence']:.2f}%**"
    )

    st.markdown("### 📈 Probability Breakdown")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <h3>❌ Fake News</h3>
            <p style="color:#dc2626;">
                {result['fake_probability']:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Real News</h3>
            <p style="color:#16a34a;">
                {result['real_probability']:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    chart_data = {
        "Fake News": result["fake_probability"],
        "Real News": result["real_probability"]
    }

    st.bar_chart(chart_data)


if st.session_state.history:

    st.divider()

    st.markdown("### 📋 Recent Prediction History")

    for i, item in enumerate(
        reversed(st.session_state.history[-5:]),
        start=1
    ):

        with st.expander(
            f"{i}. {item['Prediction']} — {item['Confidence']}"
        ):
            st.write(item["Article"])



st.divider()

st.caption(
    "⚠️ This application predicts news based on patterns "
    "learned from its training dataset. The prediction "
    "is not proof of factual accuracy and this application "
    "is not a real-time fact-checking system."
)
