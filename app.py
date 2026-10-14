
from pathlib import Path

import joblib
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Sentinel AI | Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# MODEL PATH
# Model file is in the project root on GitHub
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.joblib"


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background:
            radial-gradient(ellipse at top left, #172957 0%, transparent 42%),
            radial-gradient(ellipse at top right, #292365 0%, transparent 40%),
            #080D1A;
        color: #F5F7FF;
    }

    /* Main page spacing */
    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Global text */
    h1, h2, h3, h4, p, label,
    [data-testid="stMarkdownContainer"] {
        color: #F5F7FF;
    }

    /* Text area: enforce dark background */
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextArea"] textarea:hover,
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextArea"] textarea:active,
    [data-baseweb="textarea"] textarea {
        background: #080F20 !important;
        background-color: #080F20 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        caret-color: #FFFFFF !important;
        border: 1px solid #344968 !important;
        border-radius: 12px !important;
        opacity: 1 !important;
        box-shadow: none;
    }

    /* Text area outer container */
    div[data-testid="stTextArea"] [data-baseweb="textarea"],
    [data-baseweb="textarea"] {
        background: #080F20 !important;
        background-color: #080F20 !important;
        border-radius: 12px !important;
    }

    /* Focus styling */
    div[data-testid="stTextArea"] textarea:focus {
        border: 1px solid #6366F1 !important;
        box-shadow: 0 0 0 1px #6366F1 !important;
    }

    /* Placeholder */
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #AAB8D0 !important;
        -webkit-text-fill-color: #AAB8D0 !important;
        opacity: 1 !important;
    }

    /* Single-line input boxes */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextInput"] input:focus {
        background: #080F20 !important;
        background-color: #080F20 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        caret-color: #FFFFFF !important;
        border: 1px solid #344968 !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    div.stButton > button {
        min-height: 44px;
        border-radius: 10px;
        border: 1px solid #344968;
        background: #111D34;
        color: #F5F7FF !important;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #818CF8;
        background: #1B2A49;
        color: #FFFFFF !important;
    }

    /* Primary analyze button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        border: none;
        color: #FFFFFF !important;
    }

    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.12);
    }

    /* Dashboard cards */
    .sentinel-card {
        padding: 22px;
        border-radius: 16px;
        background: linear-gradient(145deg, #14213A, #10192C);
        border: 1px solid #2B405F;
        margin: 12px 0;
    }

    .result-spam {
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(120deg, #3A1728, #251728);
        border: 1px solid #FB5675;
        margin: 12px 0 20px 0;
    }

    .result-ham {
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(120deg, #10392F, #102A2A);
        border: 1px solid #21C995;
        margin: 12px 0 20px 0;
    }

    .small-label {
        color: #B7C7E0;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .score-number {
        font-size: 28px;
        font-weight: 800;
        margin-top: 10px;
    }

    .muted-text {
        color: #A9B6CC !important;
        font-size: 14px;
    }

    /* Progress bar */
    div[data-testid="stProgressBar"] > div {
        background-color: #25334A;
    }

    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #2196F3, #7C3AED);
    }

    /* Divider */
    hr {
        border-color: #202B40;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #0C1424;
        border: 1px solid #202D43;
        border-radius: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #8493AD;
        font-size: 12px;
        padding-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.is_file():
    st.error(
        f"Trained model not found at: {MODEL_PATH}\n\n"
        "Check that spam_model.joblib is in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================
if "message_text" not in st.session_state:
    st.session_state.message_text = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def set_message(message):
    st.session_state.message_text = message
    st.session_state.last_result = None


def clear_message():
    st.session_state.message_text = ""
    st.session_state.last_result = None


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_message(message):
    prediction = model.predict([message])[0]

    spam_probability = None
    ham_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)

        for index, class_label in enumerate(classes):
            label = str(class_label).strip().lower()

            if label == "spam":
                spam_probability = float(probabilities[index])

            elif label in ("ham", "legitimate", "not spam", "not_spam"):
                ham_probability = float(probabilities[index])

        # Fallback for binary classifiers with different label names
        if spam_probability is None and len(probabilities) == 2:
            spam_probability = float(probabilities[1])

        if ham_probability is None and len(probabilities) == 2:
            ham_probability = float(probabilities[0])

    label = str(prediction).strip().lower()
    is_spam = label in ("spam", "1", "true")

    return {
        "is_spam": is_spam,
        "prediction": str(prediction),
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
        "message": message,
    }


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div style="text-align:center; padding: 8px 0 28px 0;">
        <div style="font-size:48px;">🛡️</div>
        <h1 style="
            font-size:42px;
            font-weight:900;
            margin:0;
            background:linear-gradient(90deg,#60A5FA,#A78BFA,#F472B6);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;">
            SENTINEL AI
        </h1>
        <p style="color:#B7C7E0; font-size:16px;">
            Intelligent Spam Message Detection powered by NLP
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status metrics
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown(
        """
        <div class="sentinel-card">
            <div class="small-label">System status</div>
            <div style="color:#21C995;font-size:21px;font-weight:800;">
                ● Online
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        """
        <div class="sentinel-card">
            <div class="small-label">Model</div>
            <div style="font-size:21px;font-weight:800;">
                Logistic Regression
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric3:
    st.markdown(
        """
        <div class="sentinel-card">
            <div class="small-label">Text features</div>
            <div style="font-size:21px;font-weight:800;">
                TF-IDF
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =========================================================
# MAIN TWO-COLUMN DASHBOARD
# =========================================================
left_col, right_col = st.columns([1.15, 0.85], gap="large")


# =========================================================
# LEFT: MESSAGE ANALYZER
# =========================================================
with left_col:
    st.subheader("✉️ Message Analyzer")

    st.caption(
        "Paste a message into the box below. "
        "The trained NLP pipeline will analyze its text."
    )

    # The key keeps the text area synchronized with quick-test buttons.
    st.text_area(
        "Message text",
        key="message_text",
        height=220,
        placeholder="Type or paste your message here...",
        label_visibility="collapsed",
    )

    message_length = len(st.session_state.message_text)
    st.caption(f"📝 {message_length} characters")

    analyze_col, clear_col = st.columns([1.4, 0.7])

    with analyze_col:
        analyze_clicked = st.button(
            "⚡ ANALYZE MESSAGE",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        st.button(
            "↻ CLEAR",
            on_click=clear_message,
            use_container_width=True,
        )

    if analyze_clicked:
        message = st.session_state.message_text.strip()

        if not message:
            st.warning("Please enter a message before analyzing it.")
        else:
            try:
                st.session_state.last_result = predict_message(message)
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    st.divider()

    st.markdown("### QUICK TESTS")
    st.caption("Choose an example to fill the message box.")

    test1, test2, test3 = st.columns(3)

    with test1:
        st.button(
            "🎁 Prize message",
            on_click=set_message,
            args=("Congratulations! You have won a free prize. Claim now!",),
            use_container_width=True,
        )

    with test2:
        st.button(
            "🔒 Account alert",
            on_click=set_message,
            args=(
                "URGENT! Your account needs verification. "
                "Click here to claim your reward now!",
            ),
            use_container_width=True,
        )

    with test3:
        st.button(
            "💬 Normal chat",
            on_click=set_message,
            args=("Hey, are we meeting at college today?",),
            use_container_width=True,
        )


# =========================================================
# RIGHT: ANALYSIS RESULTS
# =========================================================
with right_col:
    st.subheader("📊 Analysis Results")

    result = st.session_state.last_result

    if result is None:
        st.markdown(
            """
            <div class="sentinel-card" style="padding:32px;">
                <div style="font-size:36px;">🔍</div>
                <h3>Ready to analyze</h3>
                <p class="muted-text">
                    Your latest prediction and classification scores
                    will appear here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sentinel-card">
                <div class="small-label">How it works</div>
                <p style="margin-top:12px;">
                    1. Enter a message<br>
                    2. Convert text using TF-IDF<br>
                    3. Classify using the trained model<br>
                    4. Review the prediction
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        is_spam = result["is_spam"]
        spam_prob = result["spam_probability"]
        ham_prob = result["ham_probability"]

        if is_spam:
            st.markdown(
                """
                <div class="result-spam">
                    <div class="small-label" style="color:#FFB5C3;">
                        ⚠ Classification result
                    </div>
                    <h2 style="margin:12px 0;">🚨 Likely Spam</h2>
                    <p>The model classified this message as spam.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-ham">
                    <div class="small-label" style="color:#82F3CF;">
                        ✓ Classification result
                    </div>
                    <h2 style="margin:12px 0;">✅ Likely Legitimate</h2>
                    <p>The model classified this message as ham.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Display confidence only if the model supports predict_proba.
        if spam_prob is not None and ham_prob is not None:
            confidence = spam_prob if is_spam else ham_prob

            st.markdown("**Prediction confidence**")
            st.progress(max(0.0, min(1.0, confidence)))
            st.markdown(
                f"<div style='font-size:30px;font-weight:900;'>"
                f"{confidence * 100:.1f}%</div>",
                unsafe_allow_html=True,
            )

            score1, score2 = st.columns(2)

            with score1:
                st.markdown(
                    f"""
                    <div class="sentinel-card">
                        <div class="small-label">Legitimate score</div>
                        <div class="score-number"
                             style="color:#21C995;">
                            {ham_prob * 100:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with score2:
                st.markdown(
                    f"""
                    <div class="sentinel-card">
                        <div class="small-label">Spam score</div>
                        <div class="score-number"
                             style="color:#FB7185;">
                            {spam_prob * 100:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "This model does not provide probability scores. "
                "Only the predicted class is shown."
            )

        with st.expander("View analyzed message"):
            st.write(result["message"])

        st.caption(
            "These are model outputs, not a guarantee that a message "
            "is safe or malicious."
        )


# =========================================================
# NLP PIPELINE INFORMATION
# =========================================================
st.divider()

with st.expander("🧠 THE NLP PIPELINE"):
    pipeline1, pipeline2, pipeline3 = st.columns(3)

    with pipeline1:
        st.markdown("### 01 · Input")
        st.write("The user enters an SMS message for classification.")

    with pipeline2:
        st.markdown("### 02 · TF-IDF")
        st.write("Text is converted into numerical feature values.")

    with pipeline3:
        st.markdown("### 03 · Prediction")
        st.write("The trained Logistic Regression model predicts the class.")

st.markdown(
    """
    <div class="footer">
        SENTINEL AI · NLP & Machine Learning Project<br>
        Built with Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
