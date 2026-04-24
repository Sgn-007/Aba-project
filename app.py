import streamlit as st
import numpy as np
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CLV Dashboard",
    page_icon="💰",
    layout="wide"
)

# =========================
# LIGHT FINTECH THEME
# =========================
st.markdown("""
<style>

/* Background with image */
[data-testid="stAppViewContainer"] {
    background: 
        linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)),
        url("https://images.unsplash.com/photo-1554224155-6726b3ff858f");
    background-size: cover;
    background-position: center;
}

/* Remove default top spacing */
.block-container {
    padding-top: 2rem;
}

/* REMOVE WHITE BARS */
div[data-testid="stVerticalBlock"] > div:empty {
    display: none;
}

/* Fix header spacing */
h3 {
    margin-bottom: 0px;
}

/* Titles */
h1 {
    text-align: center;
    color: #0f172a;
}

h2, h3 {
    color: #1e3a8a;
}

/* Card style */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

/* Button */
.stButton>button {
    background-color: #2563eb;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px;
    width: 100%;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

/* Result box */
.metric-box {
    background: #f1f5f9;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("clv_model.pkl", "rb"))

# =========================
# HEADER
# =========================
st.markdown("<h1>💰 Customer Lifetime Value</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Predict Customer Worth Instantly</h3>", unsafe_allow_html=True)

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# INPUT SECTION
# =========================
# INPUT SECTION
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Customer Details")

    total_spent = st.number_input(
        "💳 Total Spent (₹)",
        min_value=0.0,
        step=1.0,
        format="%.2f"
    )

    active_days = st.number_input(
        "📅 Active Days",
        min_value=1,
        max_value=365,
        step=1
    )

    last_transaction = st.number_input(
        "⏳ Days Since Last Transaction",
        min_value=1,
        max_value=365,
        step=1
    )

    satisfaction = st.slider("⭐ Satisfaction Score", 1, 10)

    predict_btn = st.button("🚀 Predict CLV")

    st.markdown('</div>', unsafe_allow_html=True)
# =========================
# OUTPUT SECTION
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if predict_btn:

        if total_spent <= 0:
            st.warning("⚠️ Enter a valid Total Spent amount")
        else:
            # Prepare input
            input_data = np.array([[total_spent, active_days, last_transaction, satisfaction]])

            # Log transform
            input_data[:, 0] = np.log1p(input_data[:, 0])

            # Predict
            pred_log = model.predict(input_data)[0]
            pred_clv = np.expm1(pred_log)

            # Display
            st.markdown(f"""
            <div class="metric-box">
                <h2>💰 ₹{pred_clv:,.2f}</h2>
                <p>Predicted Customer Lifetime Value</p>
            </div>
            """, unsafe_allow_html=True)

            # Classification
            if pred_clv > 500000:
                st.success("🔥 High Value Customer")
            elif pred_clv > 200000:
                st.info("⭐ Medium Value Customer")
            else:
                st.warning("📉 Low Value Customer")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER (NO EXTRA LINES)
# =========================
st.markdown(
    "<p style='text-align:center;color:gray;'>📊 CLV Prediction Dashboard</p>",
    unsafe_allow_html=True
)