import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #111c36 100%); }
    .block-container { padding-top: 1.5rem; max-width: 1150px; }
    html, body, [class*="css"] { color: #e2e8f0; }

    .hero {
        background: linear-gradient(120deg, #4f46e5 0%, #06b6d4 100%);
        padding: 28px 32px; border-radius: 18px; margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(6, 182, 212, 0.25);
    }
    .hero h1 { color: #fff; margin: 0; font-size: 2.1rem; }
    .hero p  { color: #e0f2fe; margin: 6px 0 0 0; font-size: 1.02rem; }

    .section-title {
        display: flex; align-items: center; gap: 10px;
        font-size: 1.15rem; font-weight: 700; color: #f8fafc;
        padding: 10px 14px; margin: 18px 0 8px 0; border-radius: 10px;
        background: rgba(99, 102, 241, 0.15);
        border-left: 5px solid #06b6d4;
    }

    div[data-testid="stForm"] {
        background: #162040; border: 1px solid #263459;
        border-radius: 18px; padding: 10px 24px 24px 24px;
    }

    label, .stSelectbox label, .stNumberInput label { color: #cbd5e1 !important; font-weight: 600; }

    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: #0f172a !important; border-radius: 10px !important;
        border: 1px solid #334155 !important; color: #f1f5f9 !important;
    }

    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%; border: 0; border-radius: 12px; padding: 14px 0;
        font-size: 1.1rem; font-weight: 700; color: #fff;
        background: linear-gradient(120deg, #4f46e5, #06b6d4);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.4);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px); box-shadow: 0 10px 24px rgba(6, 182, 212, 0.5); color: #fff;
    }

    .result { border-radius: 18px; padding: 26px 30px; margin-top: 22px; color: #fff; }
    .result.risk { background: linear-gradient(120deg, #be123c, #f97316); box-shadow: 0 10px 30px rgba(244, 63, 94, .35); }
    .result.safe { background: linear-gradient(120deg, #059669, #22c55e); box-shadow: 0 10px 30px rgba(34, 197, 94, .35); }
    .result h2 { margin: 0 0 6px 0; font-size: 1.6rem; }
    .result p  { margin: 0; opacity: .95; }

    .gauge-wrap { margin-top: 16px; background: rgba(255,255,255,.25); border-radius: 999px; height: 16px; overflow: hidden; }
    .gauge-fill { height: 100%; background: #fff; border-radius: 999px; }

    .kpi { background: #162040; border: 1px solid #263459; border-radius: 14px; padding: 16px 18px; text-align: center; }
    .kpi .v { font-size: 1.5rem; font-weight: 800; color: #22d3ee; }
    .kpi .l { font-size: .85rem; color: #94a3b8; margin-top: 2px; }

    .footer { text-align: center; color: #64748b; font-size: .85rem; margin-top: 30px; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("Best_Model.pkl")


model = load_model()

# ------------------------------------------------------------------ header
st.markdown(
    """
    <div class="hero">
        <h1>📊 Customer Churn Predictor</h1>
        <p>Enter the customer's details and get an instant prediction of whether they are likely to leave.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ form
with st.form("churn_form"):
    st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    gender = a1.selectbox("Gender", ["Male", "Female"])
    senior = a2.selectbox("Senior Citizen", ["No", "Yes"])
    partner = a3.selectbox("Partner", ["Yes", "No"])
    dependents = a4.selectbox("Dependents", ["Yes", "No"])

    st.markdown('<div class="section-title">📡 Services</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    phone = b1.selectbox("Phone Service", ["Yes", "No"])
    multiple = b2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = b3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    c1, c2, c3 = st.columns(3)
    security = c1.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = c2.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = c3.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    d1, d2, d3 = st.columns(3)
    tech = d1.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = d2.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = d3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown('<div class="section-title">💳 Account & Billing</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    contract = e1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = e2.selectbox("Paperless Billing", ["Yes", "No"])
    payment = e3.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

    f1, f2, f3 = st.columns(3)
    tenure = f1.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly = f2.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
    total = f3.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=840.0, step=10.0)

    st.write("")
    submitted = st.form_submit_button("🔮  Predict")

# ------------------------------------------------------------------ prediction
if submitted:
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    pred = model.predict(input_df)[0]
    is_churn = pred in ("Yes", 1)

    churn_prob = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        churn_label = "Yes" if "Yes" in classes else 1
        churn_prob = float(model.predict_proba(input_df)[0][classes.index(churn_label)])

    if is_churn:
        css, title, msg = "risk", "⚠️ This customer is at risk of leaving", "Consider a retention offer or a proactive call."
    else:
        css, title, msg = "safe", "✅ This customer is likely to stay", "No immediate action needed."

    gauge = ""
    if churn_prob is not None:
        level = "High" if churn_prob >= 0.6 else "Medium" if churn_prob >= 0.3 else "Low"
        msg = f"Churn probability: <b>{churn_prob:.1%}</b> &nbsp;|&nbsp; Risk level: <b>{level}</b>. {msg}"
        gauge = f'<div class="gauge-wrap"><div class="gauge-fill" style="width:{churn_prob*100:.0f}%"></div></div>'

    st.markdown(
        f'<div class="result {css}"><h2>{title}</h2><p>{msg}</p>{gauge}</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"{tenure} mo", "Tenure"),
        (k2, f"${monthly:,.2f}", "Monthly Charges"),
        (k3, contract, "Contract"),
        (k4, internet, "Internet Service"),
    ]:
        col.markdown(f'<div class="kpi"><div class="v">{val}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Built with Streamlit · Logistic Regression model</div>', unsafe_allow_html=True)

