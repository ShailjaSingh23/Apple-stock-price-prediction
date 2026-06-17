import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Apple Stock Forecast",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

/* Hero */
.hero{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    padding:40px;
    border-radius:25px;
    border:1px solid rgba(255,255,255,0.1);
    margin-bottom:25px;
}

.hero-title{
    color:white;
    font-size:52px;
    font-weight:800;
}

.hero-sub{
    color:#cbd5e1;
    font-size:20px;
}

/* KPI Cards */
.kpi{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border-radius:20px;
    padding:20px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.1);
}

.kpi-title{
    color:#94a3b8;
    font-size:14px;
}

.kpi-value{
    color:white;
    font-size:30px;
    font-weight:bold;
}

/* Section Titles */
.section{
    color:white;
    font-size:28px;
    font-weight:700;
    margin-top:20px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & DATA
# =========================

with open("apple_stock_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("apple_stock_data.csv")

# =========================
# HERO SECTION
# =========================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        🍎 Apple Stock Forecasting Dashboard
    </div>
    <br>
    <div class="hero-sub">
        Machine Learning Powered 30-Day Future Stock Prediction
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# FORECAST GENERATION
# =========================

history = list(df["Close"])

future_predictions = []

for i in range(30):

    lag_1 = history[-1]
    lag_2 = history[-2]
    lag_3 = history[-3]
    lag_7 = history[-7]

    ma_7 = np.mean(history[-7:])
    ma_30 = np.mean(history[-30:])

    X_future = pd.DataFrame({
        "Lag_1":[lag_1],
        "Lag_2":[lag_2],
        "Lag_3":[lag_3],
        "Lag_7":[lag_7],
        "MA_7":[ma_7],
        "MA_30":[ma_30]
    })

    pred = model.predict(X_future)[0]

    future_predictions.append(pred)
    history.append(pred)

future_df = pd.DataFrame({
    "Day": range(1,31),
    "Predicted_Close": future_predictions
})

# =========================
# KPI CARDS
# =========================

current_price = df["Close"].iloc[-1]

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Current Price</div>
        <div class="kpi-value">${current_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Forecast High</div>
        <div class="kpi-value">${future_df['Predicted_Close'].max():.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Forecast Low</div>
        <div class="kpi-value">${future_df['Predicted_Close'].min():.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Average Forecast</div>
        <div class="kpi-value">${future_df['Predicted_Close'].mean():.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================
# HISTORICAL CHART
# =========================

st.markdown(
    '<div class="section">📈 Historical Stock Performance</div>',
    unsafe_allow_html=True
)

fig_hist = px.line(
    df,
    y="Close",
    template="plotly_dark"
)

fig_hist.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500
)

st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# FORECAST CHART
# =========================

st.markdown(
    '<div class="section">🚀 30-Day Future Forecast</div>',
    unsafe_allow_html=True
)

fig_future = go.Figure()

fig_future.add_trace(
    go.Scatter(
        x=future_df["Day"],
        y=future_df["Predicted_Close"],
        mode="lines+markers",
        name="Forecast"
    )
)

fig_future.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=550
)

st.plotly_chart(fig_future, use_container_width=True)

# =========================
# FORECAST TABLE
# =========================

st.markdown(
    '<div class="section">📅 Forecast Data</div>',
    unsafe_allow_html=True
)

st.dataframe(
    future_df,
    use_container_width=True
)

# =========================
# DOWNLOAD BUTTON
# =========================

csv = future_df.to_csv(index=False)

st.download_button(
    "⬇ Download Forecast",
    csv,
    "Apple_30_Day_Forecast.csv",
    "text/csv"
)
