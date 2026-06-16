# 🎓 AI Student Performance Prediction System

A complete Machine Learning-based web application that predicts student academic performance (Pass/Fail) using Python, Scikit-Learn, and Streamlit. This project demonstrates end-to-end ML workflow including data preprocessing, model training, evaluation, and deployment with an interactive dashboard.

---

## 🚀 Project Overview

Education systems generate a large amount of student data, which can be analyzed to predict academic outcomes. This project uses machine learning techniques to predict whether a student will **pass or fail** based on important academic and behavioral factors.

The system provides a **user-friendly Streamlit web interface** with real-time prediction, visualization, authentication system, chatbot assistant, and optional voice response feature.

---

## 🎯 Problem Statement

Students’ academic performance depends on multiple factors such as study time, attendance, previous exam scores, and failures. Manual analysis is difficult and time-consuming.

This project aims to:
- Predict student performance using ML models
- Provide instant and accurate results
- Help educators identify at-risk students early

---

## 🧠 Features

- 📊 Predict student performance (Pass / Fail)
- 🤖 Intelligent rule-based AI chatbot
- 🔐 Login authentication system (admin panel)
- 📈 Data visualization using graphs and charts
- 🎯 Model evaluation (Accuracy Score, Confusion Matrix)
- 🖥️ Interactive Streamlit dashboard UI
- 🔊 Voice response feature using gTTS (optional)
- ⚡ Real-time prediction system

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit
- gTTS (Google Text-to-Speech)
- Joblib

---

---

## ⚙️ How It Works

### 1️⃣ Data Collection
The dataset contains student information such as:
- Study time
- Failures
- Absences
- G1 and G2 scores

### 2️⃣ Data Preprocessing
- Missing values handling
- Feature selection
- Label creation (Pass/Fail)

### 3️⃣ Model Training
A Machine Learning model (Random Forest / Logistic Regression) is trained using Scikit-Learn.

### 4️⃣ Model Evaluation
Model performance is evaluated using:
- Accuracy Score
- Confusion Matrix

### 5️⃣ Deployment
The trained model is integrated into a Streamlit web app for real-time predictions.

---

## ⚙️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
python student_prediction.py
streamlit run app.py

## 📂 Project Structure
