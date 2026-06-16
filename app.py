import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from gtts import gTTS
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Student Predictor", layout="wide")

# ---------------- TEXT TO SPEECH (FIXED) ----------------
def speak(text):
    tts = gTTS(text=text, lang="en")

    # unique file name (avoid overwrite issue)
    filename = f"voice_{int(time.time())}.mp3"
    tts.save(filename)

    # Streamlit audio player
    audio_file = open(filename, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- LOGIN PAGE ----------------
def login_page():
    st.title("🔐 Login to AI System")
    st.markdown("### Enter credentials to continue")

    col1, col2 = st.columns(2)

    with col1:
        username = st.text_input("Username")

    with col2:
        password = st.text_input("Password", type="password")

    if st.button("🚀 Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful 🎉")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

# ---------------- DASHBOARD ----------------
def dashboard():

    model = joblib.load("model.pkl")

    st.title("🎓 AI Student Performance Dashboard")
    st.markdown("### 🚀 Professional ML System")

    # ---------------- LOGOUT ----------------
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ---------------- INPUT ----------------
    st.sidebar.header("Student Input")

    studytime = st.sidebar.slider("Study Time", 1, 4, 2)
    failures = st.sidebar.number_input("Failures", 0, 5, 0)
    absences = st.sidebar.number_input("Absences", 0, 50, 5)
    G1 = st.sidebar.number_input("G1 Score", 0, 20, 10)
    G2 = st.sidebar.number_input("G2 Score", 0, 20, 10)

    input_df = pd.DataFrame([[studytime, failures, absences, G1, G2]],
                            columns=["studytime", "failures", "absences", "G1", "G2"])

    # ---------------- PREDICT ----------------
    if st.sidebar.button("🚀 Predict Now"):
        pred = model.predict(input_df)

        if pred[0] == 1:
            result = "Student WILL PASS"
            st.success("🎉 " + result)
        else:
            result = "Student MAY FAIL"
            st.error("⚠️ " + result)

        # 🔊 VOICE OUTPUT (FIXED)
        speak(result)

    # ---------------- CHATBOT ----------------
    st.markdown("---")
    st.subheader("🤖 Smart AI Chatbot (Voice Enabled)")

    user_input = st.chat_input("Ask anything about student performance...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        q = user_input.lower()

        if "study" in q:
            reply = "Study time improves student performance significantly."
        elif "fail" in q:
            reply = "Failures reduce chances of passing."
        elif "absent" in q:
            reply = "High absences negatively affect marks."
        elif "g1" in q or "g2" in q:
            reply = "G1 and G2 are strong predictors of result."
        elif "pass" in q:
            reply = "Good study increases passing chances."
        else:
            reply = "Ask about study, failures, absences, G1 or G2."

        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.write(reply)

        # 🔊 BOT SPEAKS (FIXED)
        speak(reply)

    # ---------------- DATA ----------------
    df = pd.read_csv("student_data.csv")
    df["result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

    X = df[["studytime", "failures", "absences", "G1", "G2"]]
    y = df["result"]

    y_pred = model.predict(X)

    # ---------------- METRICS ----------------
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎯 Accuracy", f"{accuracy_score(y, y_pred)*100:.2f}%")

    with col2:
        st.metric("👨‍🎓 Students", len(df))

    with col3:
        st.metric("📊 Features", len(X.columns))

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ---------------- GRAPH ----------------
    st.subheader("📊 Pass vs Fail Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(x="result", data=df, ax=ax2)
    st.pyplot(fig2)

# ---------------- APP FLOW ----------------
if st.session_state.logged_in:
    dashboard()
else:
    login_page()