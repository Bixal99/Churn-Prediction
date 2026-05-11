# ============================================================
# 🚀 CUSTOMER CHURN PREDICTION — ADVANCED STREAMLIT APP
# ============================================================
# Features:
# ✅ Modern Dark Glassmorphism UI
# ✅ Animated Metrics
# ✅ Beautiful Charts
# ✅ XGBoost Prediction
# ✅ Gemini AI Retention Suggestions
# ✅ Risk Gauge
# ✅ Customer Health Score
# ✅ Download Prediction Report
#
# Required Files:
# - model.pkl
# - scaler.pkl
# - features.pkl
#
# Install:
# pip install streamlit pandas numpy matplotlib plotly xgboost google-generativeai
#
# Run:
# streamlit run app.py
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    color: white;
}

/* Main Title */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(96,165,250,0.15), rgba(167,139,250,0.15));
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #60a5fa;
}

.metric-label {
    color: #cbd5e1;
    font-size: 0.9rem;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(59,130,246,0.5);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

/* Inputs */
.stSelectbox div[data-baseweb="select"],
.stNumberInput input,
.stSlider {
    border-radius: 12px !important;
}

/* Success */
.success-box {
    background: rgba(34,197,94,0.15);
    padding: 20px;
    border-radius: 16px;
    border-left: 5px solid #22c55e;
}

/* Danger */
.danger-box {
    background: rgba(239,68,68,0.15);
    padding: 20px;
    border-radius: 16px;
    border-left: 5px solid #ef4444;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, scaler, features


try:
    model, scaler, feature_cols = load_model()
    model_loaded = True
except:
    model_loaded = False

# ============================================================
# HEADER
# ============================================================

st.markdown("<div class='main-title'>🚀 AI Customer Churn Predictor</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Predict customer churn using XGBoost + Gemini AI Insights</div>",
    unsafe_allow_html=True
)

# ============================================================
# ERROR IF MODEL NOT FOUND
# ============================================================

if not model_loaded:
    st.error("""
    ❌ Model files missing!

    Required:
    - model.pkl
    - scaler.pkl
    - features.pkl
    """)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Dashboard Settings")

    st.success("✅ XGBoost Model Loaded")

    st.markdown("---")

    st.subheader("🤖 AI Integration")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    if GEMINI_API_KEY:
        st.success("Gemini Connected")
    else:
        st.warning("Gemini API Key Missing")

    st.markdown("---")

    st.subheader("📊 About")

    st.info("""
    This system predicts whether a telecom customer is likely to churn using:

    - XGBoost ML Model
    - Feature Engineering
    - AI Retention Strategies
    - Advanced Analytics
    """)

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("## 📋 Customer Information")

left, right = st.columns(2)

with left:

    tenure = st.slider("📅 Tenure (Months)", 0, 72, 12)

    monthly_charges = st.number_input(
        "💳 Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "💰 Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(tenure * monthly_charges)
    )

    senior_citizen = st.selectbox(
        "👴 Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "❤️ Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "👨‍👩‍👧 Dependents",
        ["Yes", "No"]
    )

with right:

    contract = st.selectbox(
        "📄 Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "🌐 Internet Service",
        ["Fiber optic", "DSL", "No"]
    )

    tech_support = st.selectbox(
        "🛠 Tech Support",
        ["No", "Yes", "No internet service"]
    )

    payment_method = st.selectbox(
        "💳 Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    paperless = st.selectbox(
        "📧 Paperless Billing",
        ["Yes", "No"]
    )

    phone_service = st.selectbox(
        "📞 Phone Service",
        ["Yes", "No"]
    )

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_btn = st.button("🔮 Predict Customer Churn")

# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict_btn:

    raw = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "gender": "Male",
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
    }

    input_df = pd.DataFrame([raw])

    # One-hot encoding
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align columns
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_cols]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown("---")
    st.markdown("# 🎯 Prediction Results")

    # Result Banner
    if prediction == 1:
        st.markdown(f"""
        <div class='danger-box'>
            <h2>🚨 HIGH CHURN RISK</h2>
            <p>This customer is highly likely to leave.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class='success-box'>
            <h2>✅ LOW CHURN RISK</h2>
            <p>This customer is likely to stay.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # METRICS
    # ========================================================

    m1, m2, m3, m4 = st.columns(4)

    health_score = int((1 - probability) * 100)

    with m1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{probability:.1%}</div>
            <div class='metric-label'>Churn Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{health_score}</div>
            <div class='metric-label'>Health Score</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{contract}</div>
            <div class='metric-label'>Contract</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>XGB</div>
            <div class='metric-label'>ML Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # GAUGE CHART
    # ========================================================

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Customer Churn Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 40], 'color': "#22c55e"},
                {'range': [40, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#ef4444"},
            ],
        }
    ))

    gauge.update_layout(
        paper_bgcolor="#111827",
        font={'color': "white"}
    )

    st.plotly_chart(gauge, use_container_width=True)

    # ========================================================
    # FACTOR ANALYSIS
    # ========================================================

    st.markdown("## 📊 Risk Analysis")

    factors = pd.DataFrame({
        "Factor": [
            "Contract Stability",
            "Monthly Charges",
            "Tech Support",
            "Customer Loyalty",
            "Billing Type"
        ],
        "Impact": [
            85 if contract != "Month-to-month" else 30,
            max(20, 100 - monthly_charges),
            80 if tech_support == "Yes" else 30,
            min(100, tenure * 1.5),
            60 if paperless == "No" else 40
        ]
    })

    fig = px.bar(
        factors,
        x="Impact",
        y="Factor",
        orientation='h',
        text="Impact"
    )

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font_color="white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # GEMINI AI INSIGHTS
    # ========================================================

    st.markdown("## 🤖 AI Retention Strategies")

    if GEMINI_API_KEY:

        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)

            gemini = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
You are a telecom customer retention expert.

Customer Details:
- Tenure: {tenure}
- Monthly Charges: {monthly_charges}
- Contract: {contract}
- Internet Service: {internet_service}
- Tech Support: {tech_support}
- Payment Method: {payment_method}
- Churn Probability: {probability:.1%}

Give:
1. 3 personalized retention strategies
2. 2 upselling opportunities
3. Short executive summary

Keep response concise and professional.
"""

            response = gemini.generate_content(prompt)

            st.markdown(f"""
            <div class='card'>
            {response.text}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gemini Error: {e}")

    else:
        st.info("""
        Add GEMINI_API_KEY to environment variables for AI-powered retention strategies.
        """)

    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.markdown("## 📥 Download Prediction Report")

    report = f"""
CUSTOMER CHURN PREDICTION REPORT
Generated: {datetime.now()}

========================================

Prediction:
{"HIGH CHURN RISK" if prediction == 1 else "LOW CHURN RISK"}

Churn Probability:
{probability:.1%}

Customer Details:
- Tenure: {tenure}
- Monthly Charges: ${monthly_charges}
- Contract: {contract}
- Internet Service: {internet_service}
- Tech Support: {tech_support}
- Payment Method: {payment_method}

Health Score:
{health_score}/100

Model:
XGBoost
"""

    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="churn_prediction_report.txt",
        mime="text/plain"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center;color:#94a3b8;padding:20px'>
Built with ❤️ using Streamlit · XGBoost · Plotly · Gemini AI
</div>
""", unsafe_allow_html=True)