
from pathlib import Path

import joblib
import streamlit as st

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Sentinel AI | Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_model.joblib"

# =========================================================
# 2. SESSION STATE AND CALLBACKS
# Define callbacks before creating widgets.
# =========================================================
if "message_text" not in st.session_state:
    st.session_state.message_text = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def set_message(text):
    """Populate the input box with a sample message."""
    st.session_state.message_text = text
    st.session_state.last_result = None


def clear_message():
    """Clear the input and previous prediction."""
    st.session_state.message_text = ""
    st.session_state.last_result = None


# =========================================================
# 3. CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080d1a;
        --panel: #111b2e;
        --border: #2b3c59;
        --blue: #38bdf8;
        --purple: #a78bfa;
        --text: #f1f5f9;
        --muted: #9aabc2;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, #172554 0%, transparent 35%),
            radial-gradient(ellipse at 95% 20%, #312e81 0%, transparent 32%),
            var(--bg);
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1500px !important;
        padding: 1.5rem 2.5rem 2rem !important;
    }

    h1, h2, h3, p {
        color: var(--text);
    }

    .hero {
        background: linear-gradient(115deg, #14213d, #18294b 55%, #30235e);
        border: 1px solid #354b72;
        border-radius: 24px;
        padding: 36px 40px;
        margin-bottom: 26px;
        box-shadow: 0 15px 50px #00000030;
    }

    .hero h1 {
        font-size: clamp(30px, 3.5vw, 48px);
        font-weight: 800;
        letter-spacing: -1.4px;
        margin: 12px 0;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.8;
    }

    .eyebrow {
        color: #7dd3fc;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .pill {
        display: inline-block;
        border: 1px solid #36577f;
        background: #132746;
        color: #bae6fd;
        border-radius: 30px;
        padding: 7px 12px;
        font-size: 11px;
        margin: 5px 5px 0 0;
    }

    .panel {
        background: linear-gradient(145deg, #121d32, #0e1729);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 12px 35px #00000020;
        min-height: 100%;
    }

    .panel-title {
        font-size: 21px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .muted {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.8;
    }

    .metric-card {
        background: linear-gradient(145deg, #14233d, #111b2e);
        border: 1px solid #293d5c;
        border-radius: 16px;
        padding: 19px;
        min-height: 112px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #9fb1ca;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 800;
        margin-top: 12px;
        color: #f8fafc;
        overflow-wrap: anywhere;
    }

    .result-card {
        padding: 23px;
        border-radius: 16px;
        margin: 16px 0;
    }

    .spam {
        background: linear-gradient(120deg, #451b2b, #2c1829);
        border: 1px solid #fb7185;
    }

    .legit {
        background: linear-gradient(120deg, #103b35, #102c31);
        border: 1px solid #34d399;
    }

    .result-label {
        font-size: 26px;
        font-weight: 800;
        margin: 8px 0;
    }

    div[data-testid="stTextArea"] textarea {
        background: #0a1222 !important;
        color: #f8fafc !important;
        border: 1px solid #344b6c !important;
        border-radius: 14px !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
        padding: 16px !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px #38bdf833 !important;
    }

    .stButton > button {
        min-height: 45px;
        border-radius: 11px;
        font-weight: 650;
        border: 1px solid #354b6b;
        transition: 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2563eb, #7c3aed);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        border-color: #38bdf8;
        color: white;
    }

    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
    }

    div[data-testid="stMetric"] {
        background: #121d32;
        padding: 14px;
        border: 1px solid #293d5c;
        border-radius: 12px;
    }

    hr {
        border-color: #293954;
    }

    .footer {
        text-align: center;
        color: #7185a5;
        font-size: 12px;
        padding: 25px 0 5px;
        line-height: 2;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }

        .hero {
            padding: 23px;
        }

        .panel {
            padding: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 4. LOAD THE TRAINED MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please run train.py first "
        "and confirm that models/spam_model.joblib exists."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

# =========================================================
# 5. HERO HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ INTELLIGENT MESSAGE SECURITY</div>
        <h1>🛡️ Sentinel <span style="color:#60a5fa">AI</span></h1>
        <p>
            Understand your messages. Detect unwanted content.
            Explore the power of Natural Language Processing
            through an interactive spam classification system.
        </p>
        <div>
            <span class="pill">NLP ENGINE</span>
            <span class="pill">TF-IDF</span>
            <span class="pill">LOGISTIC REGRESSION</span>
            <span class="pill">LIVE PREDICTIONS</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 6. TOP METRICS
# =========================================================
vocab_size = "N/A"

try:
    vocab_size = f"{len(model.named_steps['tfidf'].vocabulary_):,}"
except (AttributeError, KeyError, TypeError):
    pass

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">SYSTEM STATUS</div>
            <div class="metric-value" style="color:#34d399">
                ● Online
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">CLASSIFIER</div>
            <div class="metric-value" style="font-size:21px">
                Logistic Regression
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">VOCABULARY FEATURES</div>
            <div class="metric-value">{vocab_size}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 7. MAIN WORKSPACE
# =========================================================
st.write("")
left, right = st.columns([1.2, 0.9], gap="large")

# ------------------ INPUT PANEL ------------------
with left:
    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-title">✉️ Message Analyzer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="muted">Paste a message into the box below. '
        'The trained NLP pipeline will analyze its text.</p>',
        unsafe_allow_html=True,
    )

    # This widget uses session state. Its value is updated
    # only by callbacks declared before widget creation.
    message = st.text_area(
        "Message text",
        key="message_text",
        placeholder=(
            "Example: Congratulations! You have won a free prize. "
            "Claim your reward now..."
        ),
        height=260,
        max_chars=10000,
        label_visibility="collapsed",
    )

    st.caption(f"📝 {len(message):,} characters")

    analyze_col, clear_col = st.columns([2, 1])

    with analyze_col:
        analyze = st.button(
            "⚡ ANALYZE MESSAGE",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        st.button(
            "↻ CLEAR",
            use_container_width=True,
            on_click=clear_message,
        )

    st.markdown("---")
    st.markdown("**QUICK TESTS**")
    st.caption("Choose an example to fill the message box.")

    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.button(
            "🎁 Prize message",
            use_container_width=True,
            on_click=set_message,
            args=(
                "Congratulations! You have won a free prize. Claim now!",
            ),
        )

    with ex2:
        st.button(
            "🔐 Account alert",
            use_container_width=True,
            on_click=set_message,
            args=(
                "Your account will be suspended. "
                "Verify your credentials immediately.",
            ),
        )

    with ex3:
        st.button(
            "💬 Normal chat",
            use_container_width=True,
            on_click=set_message,
            args=(
                "Hey, are we meeting at the library at 5?",
            ),
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 8. HANDLE PREDICTION
# =========================================================
if analyze:
    current_message = st.session_state.message_text.strip()

    if not current_message:
        st.session_state.last_result = None
        st.warning("Please enter a message before analyzing.")
    else:
        try:
            prediction = int(model.predict([current_message])[0])
            probabilities = model.predict_proba([current_message])[0]

            st.session_state.last_result = {
                "prediction": prediction,
                "confidence": float(max(probabilities)) * 100,
                "legit_probability": float(probabilities[0]) * 100,
                "spam_probability": float(probabilities[1]) * 100,
                "message": current_message,
            }

        except Exception as exc:
            st.session_state.last_result = None
            st.error(f"Prediction failed: {exc}")

# =========================================================
# 9. RESULTS PANEL
# =========================================================
with right:
    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-title">📊 Analysis Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="muted">Your latest prediction and '
        'classification scores appear here.</p>',
        unsafe_allow_html=True,
    )

    result = st.session_state.last_result

    if result is not None:
        if result["prediction"] == 1:
            st.markdown(
                """
                <div class="result-card spam">
                    <div style="font-size:12px;color:#fda4af">
                        ⚠ CLASSIFICATION RESULT
                    </div>
                    <div class="result-label">🚨 Likely Spam</div>
                    <div style="color:#fecdd3">
                        The model classified this message as spam.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card legit">
                    <div style="font-size:12px;color:#6ee7b7">
                        ✓ CLASSIFICATION RESULT
                    </div>
                    <div class="result-label">
                        ✅ Likely Legitimate
                    </div>
                    <div style="color:#a7f3d0">
                        The model classified this message as legitimate.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("**Prediction confidence**")
        st.progress(
            min(1.0, max(0.0, result["confidence"] / 100))
        )

        st.markdown(
            f"""
            <div style="font-size:32px;font-weight:800;
                        margin:5px 0 18px 0">
                {result['confidence']:.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2 = st.columns(2)

        with p1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">LEGITIMATE SCORE</div>
                    <div class="metric-value"
                         style="color:#34d399;font-size:23px">
                        {result['legit_probability']:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">SPAM SCORE</div>
                    <div class="metric-value"
                         style="color:#fb7185;font-size:23px">
                        {result['spam_probability']:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View analyzed message"):
            st.write(result["message"])

        st.caption(
            "These scores are model outputs, not a guarantee "
            "that a message is safe or malicious."
        )

    else:
        st.markdown(
            """
            <div style="text-align:center;padding:42px 10px">
                <div style="font-size:55px">🔎</div>
                <h3 style="color:#cbd5e1">Ready to Analyze</h3>
                <p style="color:#94a3b8;line-height:1.8">
                    Enter a message on the left and click
                    Analyze Message to view its prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**🧠 THE NLP PIPELINE**")

    pipeline_steps = [
        ("01", "Text input", "Receive the message"),
        ("02", "TF-IDF", "Convert text into numerical features"),
        ("03", "Classification", "Apply Logistic Regression"),
        ("04", "Prediction", "Return spam or legitimate"),
    ]

    for number, title, description in pipeline_steps:
        st.markdown(
            f"""
            <div style="padding:11px 0;
                        border-bottom:1px solid #293954">
                <span style="color:#7dd3fc;font-weight:800">
                    {number}
                </span>
                &nbsp; <strong>{title}</strong>
                <div style="color:#94a3b8;font-size:12px;
                            margin:5px 0 0 31px">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 10. EDUCATIONAL SECTION
# =========================================================
st.write("")
st.write("")

st.markdown("## 🧩 Understanding the Results")
st.markdown(
    '<p class="muted">Learn what each prediction means.</p>',
    unsafe_allow_html=True,
)

g1, g2, g3 = st.columns(3, gap="medium")

with g1:
    st.markdown(
        """
        <div class="panel">
            <h3>🚨 Spam</h3>
            <p class="muted">
                The classifier recognizes patterns associated with
                spam messages in its training data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with g2:
    st.markdown(
        """
        <div class="panel">
            <h3>✅ Legitimate</h3>
            <p class="muted">
                The classifier predicts that the text resembles
                legitimate messages in its training data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with g3:
    st.markdown(
        """
        <div class="panel">
            <h3>🛡️ Stay Cautious</h3>
            <p class="muted">
                This is a research prototype, not a security guarantee.
                A legitimate result can still contain malicious links.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 11. FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        <strong>SENTINEL AI</strong>
        <br>
        Built with Python · Streamlit · TF-IDF · scikit-learn
        <br>
        Educational NLP project · Predictions may be incorrect
        <br>
        This model classifies spam versus legitimate messages.
        Dedicated phishing detection requires separate training and evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)
