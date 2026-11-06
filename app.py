
import re
import joblib
import streamlit as st

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


st.markdown("""
<style>

/* PAGE BACKGROUND */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #f5f7ff !important;
    color: #1e293b !important;
}

.block-container {
    max-width: 950px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
}


/* TEXT VISIBILITY */

.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"] {
    color: #1e293b !important;
}


/* HERO HEADER */

.hero {
    background: linear-gradient(
        120deg,
        #172554,
        #4338ca,
        #6366f1
    );
    padding: 25px 20px;
    border-radius: 16px;
    color: white !important;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(67, 56, 202, 0.16);
}

.hero h1 {
    color: white !important;
    font-size: 30px !important;
    margin-bottom: 8px;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 14px;
    margin-bottom: 0;
}


/* SECTION HEADINGS */

.stApp h2,
.stApp h3 {
    color: #1e3a8a !important;
}


/* RESULT AND METRIC CARDS */

.result-card,
.metric-card {
    background: white !important;
    padding: 18px;
    border-radius: 13px;
    border: 1px solid #dbeafe;
    box-shadow: 0 3px 12px rgba(30, 64, 175, 0.06);
    margin-bottom: 12px;
}

.result-card {
    text-align: center;
}

.metric-card {
    text-align: center;
}

.metric-card h3 {
    color: #334155 !important;
    font-size: 16px;
}

.metric-card p {
    font-size: 25px;
    font-weight: bold;
    margin: 5px;
}


/* BUTTONS */

div.stButton > button {
    border-radius: 9px;
    border: 1px solid #c7d2fe;
    background: white !important;
    color: #3730a3 !important;
    font-weight: 600;
    padding: 9px;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #eef2ff !important;
    border-color: #6366f1 !important;
    color: #312e81 !important;
}


/* NEWS TEXT AREA */

div[data-testid="stTextArea"] textarea {
    background: white !important;
    color: #1e293b !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 10px;
    font-size: 14px;
}

div[data-testid="stTextArea"] label {
    color: #334155 !important;
}


/* EXPANDER */

[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #dbeafe !important;
    border-radius: 10px;
}

[data-testid="stExpander"] summary {
    color: #1e3a8a !important;
}


/* CHART */

[data-testid="stVegaLiteChart"] {
    background: white !important;
    border-radius: 12px;
    padding: 10px;
}


/* PROGRESS BAR */

.stProgress > div > div > div > div {
    background: #4f46e5 !important;
}


/* DISCLAIMER */

.stCaption,
.stApp small {
    color: #64748b !important;
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
    height=170,
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


# =========================================================
# DISPLAY PREDICTION RESULT
# =========================================================

if "last_prediction" in st.session_state:

    result = st.session_state.last_prediction

    st.divider()

    st.markdown("### 📊 Prediction Result")

    if result["prediction"] == 0:

        st.markdown("""
        <div class="result-card">
            <h2 style="color:#dc2626 !important;">
                ❌ FAKE NEWS
            </h2>
            <p style="color:#475569 !important;">
                The model identified this article as Fake News.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">
            <h2 style="color:#16a34a !important;">
                ✅ REAL NEWS
            </h2>
            <p style="color:#475569 !important;">
                The model identified this article as Real News.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("### 🎯 Prediction Confidence")

    st.progress(result["confidence"] / 100)

    st.write(
        f"**Confidence: {result['confidence']:.2f}%**"
    )

    st.markdown("### 📈 Probability Breakdown")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <h3>❌ Fake News</h3>
            <p style="color:#dc2626 !important;">
                {result['fake_probability']:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Real News</h3>
            <p style="color:#16a34a !important;">
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
