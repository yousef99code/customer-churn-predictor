import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------- Page configuration
st.set_page_config(
    page_title="ChurnIQ | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "Best_Model.pkl"

# ----------------------------- Styling
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background: radial-gradient(ellipse at top right, #102d65 0%, #07142d 42%, #050d20 100%);
        color: #eaf2ff;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1500px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07152e 0%, #081027 100%);
        border-right: 1px solid #1d3764;
    }
    [data-testid="stSidebar"] * { color: #dce9ff; }
    .developer {
        background: linear-gradient(110deg, rgba(18,43,91,.96), rgba(10,25,55,.96));
        border: 1px solid #244b85; border-radius: 16px;
        padding: 14px 20px; margin-bottom: 18px;
    }
    .eyebrow { color: #7dd3fc; font-size: .76rem; font-weight: 800; letter-spacing: 2px; }
    .dev-name { color: #fff; font-size: 1.32rem; font-weight: 800; margin-top: 2px; }
    .dev-role { color: #a9c5ed; font-size: .88rem; margin-top: 2px; }
    .hero {
        background: linear-gradient(115deg, #0877e8 0%, #263edb 58%, #5127c9 100%);
        border: 1px solid rgba(125,211,252,.35); border-radius: 20px;
        padding: 25px 28px; margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0,90,220,.18);
    }
    .hero-kicker { color: #c7eaff; font-size: .78rem; font-weight: 800; letter-spacing: 1.8px; }
    .hero h1 { color: white; font-size: clamp(1.8rem, 3vw, 2.55rem); line-height: 1.1; margin: 8px 0; }
    .hero p { color: #e1efff; margin: 0; font-size: 1rem; }
    .panel {
        background: linear-gradient(145deg, rgba(15,34,70,.98), rgba(8,22,49,.98));
        border: 1px solid #234574; border-radius: 17px; padding: 18px 20px;
        margin-bottom: 15px;
    }
    .panel-title { color: #f3f7ff; font-size: 1.13rem; font-weight: 800; margin-bottom: 4px; }
    .panel-subtitle { color: #9fb8dc; font-size: .85rem; margin-bottom: 12px; }
    .section-title {
        color: #a9d9ff; font-size: .96rem; font-weight: 800;
        padding: 9px 12px; margin: 14px 0 10px 0;
        border-left: 4px solid #22c5f5; border-radius: 7px;
        background: rgba(20,91,160,.19);
    }
    label, [data-testid="stWidgetLabel"] p { color: #d9e7fb !important; font-weight: 600 !important; }
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: #07172f !important; border: 1px solid #2a4e7d !important;
        border-radius: 9px !important; color: #f3f7ff !important;
    }
    div[data-testid="stForm"] {
        background: transparent; border: 0; padding: 0;
    }
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%; min-height: 48px; border: 0; border-radius: 11px;
        color: #fff; font-weight: 800; font-size: 1rem;
        background: linear-gradient(100deg, #087be8, #682cf0);
        box-shadow: 0 7px 20px rgba(44,92,230,.25);
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        color: #fff; border: 1px solid #79dcff;
    }
    .result-card {
        padding: 20px; border-radius: 14px; border: 1px solid #287d70;
        background: linear-gradient(135deg, rgba(5,90,80,.38), rgba(7,33,57,.9));
        margin: 12px 0;
    }
    .result-card.risk { border-color: #e65c71; background: linear-gradient(135deg, rgba(143,25,56,.5), rgba(28,27,57,.92)); }
    .result-status { font-size: 1.35rem; font-weight: 900; color: #70f2ce; }
    .risk .result-status { color: #ff9aa9; }
    .prob-number { font-size: 2rem; font-weight: 900; color: #f7fbff; margin: 4px 0; }
    .progress-track { height: 12px; border-radius: 20px; background: #2d486b; overflow: hidden; margin: 8px 0; }
    .progress-fill { height: 100%; border-radius: 20px; background: linear-gradient(90deg,#31e4b0,#20a7ff); }
    .risk .progress-fill { background: linear-gradient(90deg,#fb7185,#fb923c); }
    .stat-card { background: rgba(14,37,77,.88); border: 1px solid #234574; border-radius: 12px; padding: 13px; min-height: 85px; }
    .stat-value { color: #67d7ff; font-size: 1.08rem; font-weight: 850; overflow-wrap: anywhere; }
    .stat-label { color: #a9c0e1; font-size: .78rem; margin-top: 4px; }
    .tip { background: rgba(25,77,125,.23); border: 1px solid #234574; border-radius: 12px; padding: 13px; color: #bfd5f3; font-size: .88rem; }
    .footer { text-align: center; color: #7189ac; font-size: .8rem; margin-top: 24px; padding: 12px; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Load model
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

try:
    if not MODEL_PATH.exists():
        st.error("Model file not found. Place Best_Model.pkl in the same folder as app.py.")
        st.stop()
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The model could not be loaded. Check that Best_Model.pkl matches this app.")
    st.exception(exc)
    st.stop()

# ----------------------------- Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 18px 4px">
          <div style="font-size:2rem">📊</div>
          <div style="font-size:1.55rem;font-weight:900;color:#ffffff">ChurnIQ</div>
          <div style="font-size:.78rem;color:#8fb9ee">Customer Intelligence Platform</div>
        </div>
        <div style="background:linear-gradient(100deg,#1249a5,#173d83);padding:10px 12px;border-radius:9px;font-weight:800;margin-bottom:8px">⌂ &nbsp; Home</div>
        <div style="padding:10px 12px;color:#a8c5ef">ⓘ &nbsp; About Project</div>
        <div style="padding:10px 12px;color:#a8c5ef">⚙ &nbsp; How It Works</div>
        <div style="padding:10px 12px;color:#a8c5ef">〈/〉 &nbsp; Technology</div>
        <hr style="border-color:#213a63">
        <div style="background:#0d1e3d;border:1px solid #203d6a;border-radius:12px;padding:14px;margin-top:25px">
          <div style="font-weight:800;color:#ffffff">💡 Better Insights, Lower Churn</div>
          <div style="font-size:.82rem;color:#a8c5ef;margin-top:7px">Turn customer data into smarter retention decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------- Developer header
st.markdown(
    """
    <div class="developer">
      <div class="eyebrow">DEVELOPED BY</div>
      <div class="dev-name">Eng. Yousef Khaled Madin</div>
      <div class="dev-role">Machine Learning Developer</div>
    </div>
    <div class="hero">
      <div class="hero-kicker">CUSTOMER CHURN PREDICTION</div>
      <h1>Customer Churn Predictor</h1>
      <p>Enter customer details below to estimate whether they may leave the company.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 0.9], gap="large")

with left:
    st.markdown('<div class="panel-title">👤 Customer Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Complete the details to generate a prediction.</div>', unsafe_allow_html=True)

    with st.form("churn_form"):
        st.markdown('<div class="section-title">1. Customer Profile</div>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        gender = a1.selectbox("Gender", ["Male", "Female"])
        senior = a2.selectbox("Senior Citizen", ["No", "Yes"])
        partner = a3.selectbox("Partner", ["Yes", "No"])
        dependents = a4.selectbox("Dependents", ["Yes", "No"])

        st.markdown('<div class="section-title">2. Services</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-title">3. Account & Billing</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        contract = e1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = e2.selectbox("Paperless Billing", ["Yes", "No"])
        payment = e3.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

        st.markdown('<div class="section-title">4. Usage & Charges</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        tenure = f1.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly = f2.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
        total = f3.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=840.0, step=10.0)
        submitted = st.form_submit_button("✨  Predict Customer Churn")

with right:
    st.markdown('<div class="panel-title">🎯 Prediction Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Your model output will appear here.</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown(
            """
            <div class="tip">
              <div style="font-size:1.05rem;font-weight:800;color:#eaf4ff;margin-bottom:7px">Ready when you are</div>
              Fill in the customer information, then select <b>Predict Customer Churn</b> to see the model's result.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### What you'll see")
        st.markdown("- **Prediction:** likely to stay or at risk of leaving")
        st.markdown("- **Probability:** shown only if the model supports it")
        st.markdown("- **Customer summary:** the key values entered in the form")
    else:
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

        try:
            pred = model.predict(input_df)[0]
            is_churn = str(pred).strip().lower() in ("yes", "1", "true", "churn")
            churn_prob = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(getattr(model, "classes_", []))
                churn_index = next(
                    (i for i, cls in enumerate(classes)
                     if str(cls).strip().lower() in ("yes", "1", "true", "churn")),
                    None,
                )
                if churn_index is not None and churn_index < len(probabilities):
                    churn_prob = float(probabilities[churn_index])

            css = "risk" if is_churn else ""
            status = "AT RISK OF LEAVING" if is_churn else "LIKELY TO STAY"
            emoji = "⚠️" if is_churn else "✅"
            message = (
                "Consider a retention offer or a proactive customer call."
                if is_churn else "No immediate retention action is indicated by this prediction."
            )
            st.markdown(
                f'<div class="result-card {css}"><div class="result-status">{emoji} {status}</div>'
                f'<div style="color:#c4d8f2;margin-top:7px">{message}</div></div>',
                unsafe_allow_html=True,
            )

            if churn_prob is not None:
                level = "High" if churn_prob >= 0.6 else "Medium" if churn_prob >= 0.3 else "Low"
                st.markdown(
                    f'<div class="panel"><div style="color:#a9c0e1;font-weight:700">CHURN PROBABILITY</div>'
                    f'<div class="prob-number">{churn_prob:.1%}</div>'
                    f'<div class="progress-track"><div class="progress-fill" style="width:{churn_prob*100:.1f}%"></div></div>'
                    f'<div style="color:#b7cce9;font-size:.86rem">Estimated risk level: <b>{level}</b></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("This model does not expose a probability score; only its predicted class is shown.")

            st.markdown("#### Quick Stats")
            q1, q2 = st.columns(2)
            q1.markdown(f'<div class="stat-card"><div class="stat-value">{tenure} months</div><div class="stat-label">Tenure</div></div>', unsafe_allow_html=True)
            q2.markdown(f'<div class="stat-card"><div class="stat-value">${monthly:,.2f}</div><div class="stat-label">Monthly Charges</div></div>', unsafe_allow_html=True)
            q3, q4 = st.columns(2)
            q3.markdown(f'<div class="stat-card"><div class="stat-value">{contract}</div><div class="stat-label">Contract</div></div>', unsafe_allow_html=True)
            q4.markdown(f'<div class="stat-card"><div class="stat-value">{internet}</div><div class="stat-label">Internet Service</div></div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error("Prediction failed. The input columns or preprocessing may not match the saved model.")
            st.caption("Please check that Best_Model.pkl was trained with the same input features and preprocessing.")
            st.exception(exc)

st.markdown(
    '<div class="footer">ChurnIQ · Customer Churn Prediction Platform · Developed by Eng. Yousef Khaled Madin</div>',
    unsafe_allow_html=True,
)
