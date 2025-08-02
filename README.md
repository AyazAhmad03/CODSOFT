# 📉 Customer Churn Prediction – End-to-End ML Web App

This is an end-to-end machine learning web application that predicts whether a customer is likely to churn based on various features. The model is trained using a dataset from Kaggle and deployed through a Flask backend with a clean frontend built using HTML, CSS, and JavaScript.

---

## 🔍 Project Overview

Customer churn is a critical metric for businesses, especially in subscription-based services. This project uses machine learning to predict churn so businesses can take preventive measures and improve customer retention.

---

## 💡 Features

- 🔬 Predicts customer churn using trained ML model
- 🎨 Interactive and responsive web UI
- ⚙️ Flask-powered backend for real-time prediction
- 📊 Model built with scikit-learn
- 🧠 Dataset sourced from [Kaggle](https://www.kaggle.com/)

---

## 🛠 Tech Stack

| Layer     | Technologies Used              |
|-----------|-------------------------------|
| Frontend  | HTML, CSS, JavaScript          |
| Backend   | Python, Flask                  |
| ML Model  | scikit-learn, pandas, NumPy    |
| Tools     | VS Code, Git, GitHub           |

---

## 📁 Project Structure

CHURN-PREDICTOR/
│
├── app.py # Flask backend
├── index.html # Frontend HTML page
├── churn_model.pkl # Trained ML model
├── Churn_Modelling.csv # Dataset
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── venv/ # Virtual environment (excluded in .gitignore)


---

## ⚙️ How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/codsoft.git
cd codsoft

2. Create and Activate Virtual Environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install Requirements
pip install -r requirements.txt

4. Run the App
python app.py