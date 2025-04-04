# 🧠 Credit Risk Assessment using AI (Supervised Models)

This project focuses on enhancing credit risk assessment using machine learning techniques. It predicts whether a loan applicant is a **low risk** or **high risk**, aiding banks and financial institutions in making informed lending decisions.

---

## 📂 Project Structure

```
Credit Risk Assessment/
│
├── app/
│   ├── app.py             # Flask API for model inference
│   └── dashboard.py       # Streamlit dashboard for user input and prediction
│
├── data/
│   └── credit_risk_dataset.csv
│
├── models/
│   ├── random_forest.pkl  # Trained Random Forest model
│   ├── xgboost.pkl        # Trained XGBoost model
│   └── scaler.pkl         # Standard scaler used in preprocessing
│
├── notebooks/
│   └── eda.ipynb          # EDA, preprocessing, model training and evaluation
│
├── venv64/                # Virtual environment folder (should be in .gitignore)
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <repo_url>
cd "Credit Risk Assessment"
```

### 2. Create Virtual Environment & Activate
```bash
python3 -m venv venv64
source venv64/bin/activate      # Mac/Linux
venv64\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Flask API
```bash
python app/app.py
```
API will be running at: `http://127.0.0.1:5000/`

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```

---

## 🧾 Features Used in the Model
- `person_age`
- `person_income`
- `person_emp_length` *(slider in dashboard)*
- `loan_amnt`
- `loan_int_rate`
- `loan_percent_income`
- `cb_person_cred_hist_length` *(slider in dashboard)*
- One-hot encoded categorical fields:
  - `person_home_ownership`
  - `loan_intent`
  - `loan_grade`
  - `cb_person_default_on_file`

---

## 🤖 Models Implemented
- **Random Forest Classifier**
- **XGBoost Classifier**

The best performing model is saved and used in the backend for predictions.

---

## 🎯 Sample Output
```
✅ Approved: Low Risk
❌ Rejected: High Risk
```
Prediction is shown on the dashboard based on the input values.

---

## 🌐 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Flask
- Streamlit
- Matplotlib, Seaborn

---

## 📌 Future Scope
- Model explainability using SHAP or LIME
- CI/CD pipeline for automated deployment
- Integration with real-time databases (e.g., Firebase, MongoDB)
- Cloud deployment (AWS/GCP/Azure)

---

## 📜 License
**MIT License** — Feel free to use and modify this project.

