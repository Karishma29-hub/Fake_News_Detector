import streamlit as st
import joblib
import re

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# HEADER
# =========================================================

st.title("📰 Fake News Detector")

st.write(
    "Enter a news article and use the machine learning model "
    "to predict whether it is Fake or Real."
)

st.divider()


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("🤖 About the Model"):

    st.write("**Machine Learning Model:** Logistic Regression")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Training Articles:** 35,918")
    st.write("**Testing Articles:** 8,980")
    st.write("**Test Accuracy:** 98.50%")


st.divider()


# =========================================================
# EXAMPLE NEWS
# =========================================================

st.subheader("🧪 Try an Example")

example1 = (
    "The government announced a new education policy "
    "to improve educational infrastructure and student support."
)

example2 = (
    "A new law will require every college student in India "
    "to attend classes for exactly 12 hours every day, "
    "including Sundays, starting next month."
)

col1, col2 = st.columns(2)

with col1:

    if st.button("Example 1", use_container_width=True):
        st.session_state.news_input = example1

with col2:

    if st.button("Example 2", use_container_width=True):
        st.session_state.news_input = example2


# =========================================================
# NEWS INPUT
# =========================================================

st.subheader("📝 Enter News Article")

news_text = st.text_area(
    "Paste your article here:",
    value=st.session_state.news_input,
    height=220,
    placeholder="Paste a news article here..."
)


# =========================================================
# BUTTONS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    check_button = st.button(
        "🔍 Check News",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


# =========================================================
# CLEAR BUTTON
# =========================================================

if clear_button:

    st.session_state.news_input = ""

    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_fake_probability", None)
    st.session_state.pop("last_real_probability", None)

    st.rerun()


# =========================================================
# CHECK NEWS
# =========================================================

if check_button:

    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article first.")

    else:

        # Clean text
        cleaned_text = clean_text(news_text)

        # Convert to TF-IDF
        text_tfidf = vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(text_tfidf)[0]

        # Probability
        probability = model.predict_proba(text_tfidf)[0]

        fake_probability = probability[0] * 100
        real_probability = probability[1] * 100

        confidence = max(
            fake_probability,
            real_probability
        )

        # Save result
        st.session_state.last_prediction = prediction

        st.session_state.last_fake_probability = fake_probability

        st.session_state.last_real_probability = real_probability

        # Result text
        if prediction == 0:
            result_text = "FAKE NEWS"
        else:
            result_text = "REAL NEWS"

        # Save history
        st.session_state.history.append({
            "Article": news_text,
            "Prediction": result_text,
            "Confidence": f"{confidence:.2f}%"
        })


# =========================================================
# DISPLAY RESULT
# =========================================================

if "last_prediction" in st.session_state:

    st.divider()

    st.subheader("📊 Prediction Result")

    prediction = st.session_state.last_prediction

    fake_probability = st.session_state.last_fake_probability

    real_probability = st.session_state.last_real_probability

    confidence = max(
        fake_probability,
        real_probability
    )

    # Prediction
    if prediction == 0:
        st.error("❌ FAKE NEWS")
    else:
        st.success("✅ REAL NEWS")

    # Confidence
    st.write(
        f"### Confidence: {confidence:.2f}%"
    )

    st.progress(int(confidence))

    # Probability breakdown
    st.write("### Probability Breakdown")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "❌ Fake News",
            f"{fake_probability:.2f}%"
        )

    with col2:

        st.metric(
            "✅ Real News",
            f"{real_probability:.2f}%"
        )

    # Probability chart
    st.write("### 📊 Probability Chart")

    chart_data = {
        "Fake News": fake_probability,
        "Real News": real_probability
    }

    st.bar_chart(chart_data)


# =========================================================
# PREDICTION HISTORY
# =========================================================

if len(st.session_state.history) > 0:

    st.divider()

    st.subheader("📋 Prediction History")

    for i, item in enumerate(
        reversed(st.session_state.history[-5:]),
        start=1
    ):

        st.write(
            f"**{i}. {item['Prediction']}** — "
            f"{item['Confidence']} confidence"
        )

        st.write(item["Article"])


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "⚠️ This application predicts based on patterns "
    "learned from its training dataset. It is not a "
    "real-time fact-checking system."
)