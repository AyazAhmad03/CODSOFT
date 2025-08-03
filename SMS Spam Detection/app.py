import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Set up page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📨",
    layout="wide"
)

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9f9f9;
    }
    .main-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0,0,0,0.05);
    }
    .header {
        text-align: center;
        padding-bottom: 1rem;
    }
    .stTextArea textarea {
        font-size: 16px;
        padding: 10px;
    }
    .result-box {
        margin-top: 1.5rem;
        padding: 1rem;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .spam {
        background-color: #ffe6e6;
        color: #b30000;
        border: 1px solid #b30000;
    }
    .ham {
        background-color: #e6ffe6;
        color: #006600;
        border: 1px solid #006600;
    }
    .stButton button {
        background-color: #0077b6;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Info ---
st.sidebar.title("📘 About")
st.sidebar.info("""
This app uses a **Linear Support Vector Machine** model trained on SMS data to classify messages as:

- ✅ **Ham** (Not Spam)
- 🚫 **Spam**

Built using **Streamlit**, it processes input using **TF-IDF vectorization** for efficient text classification.
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by: *AYAZ AHMAD*")

# --- Main UI ---
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="header"><h1>📩 SMS Spam Detection </h1><p style="color:gray;">A simple, intelligent system to classify messages</p></div>', unsafe_allow_html=True)

    user_input = st.text_area("✉️ Enter your SMS message below:", height=150, placeholder="Type or paste your SMS message here...")

    if st.button("🧠 Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a message before prediction.")
        else:
            transformed = vectorizer.transform([user_input])
            prediction = model.predict(transformed)[0]

            if prediction == 1:
                st.markdown('<div class="result-box spam">🚫 SPAM: This message is likely unwanted or dangerous.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box ham">✅ HAM: This message appears to be safe.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
>>>>>>> 07b339a (Initial commit)
