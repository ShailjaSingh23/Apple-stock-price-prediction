import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Apple Stock Price Prediction",
    page_icon="🍎",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.big-title {
    text-align:center;
    font-size:48px;
    font-weight:700;
    color:#111827;
}

.sub-title {
    text-align:center;
    font-size:18px;
    color:#6b7280;
    margin-bottom:20px;
}

.metric-box {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

footer {
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD FILES
# -----------------------------
with open("apple_stock_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("apple_stock_data.csv")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<div class='big-title'>🍎 Apple Stock Price Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>30-Day Future Stock Forecast using Machine Learning</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
    width=120
)

st.sidebar.title("Project Information")

st.sidebar.info(
"""
Model Used: Linear Regression

Features:
• Lag_1
• Lag_2
• Lag_3
• Lag_7
• MA_7
• MA_30

Forecast Horizon:
30 Days
"""
)

# -----------------------------
# PROJECT OVERVIEW
# -----------------------------
st.subheader("📌 Project Overview")

st.write("""
This project predicts Apple's future stock prices using
Machine Learning and Time-Series Feature Engineering.

The model was trained using:

- Previous stock closing prices (Lag Features)
- Moving averages
- Historical stock trends

After comparing multiple models, Linear Regression
provided the best performance and was selected for
future forecasting.
""")

st.markdown("---")

# -----------------------------
# HISTORICAL CHART
# -----------------------------
st.subheader("📈 Historical Stock Trend")

fig_hist = px.line(
    df,
    y="Close",
    title="Historical Apple Closing Price"
)

fig_hist.update_layout(
    height=500,
    template="plotly_white"
)

st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------
# FUTURE FORECAST
# -----------------------------
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
        "Lag_1": [lag_1],
        "Lag_2": [lag_2],
        "Lag_3": [lag_3],
        "Lag_7": [lag_7],
        "MA_7": [ma_7],
        "MA_30": [ma_30]
    })

    pred = model.predict(X_future)[0]

    future_predictions.append(pred)
    history.append(pred)

future_df = pd.DataFrame({
    "Day": range(1, 31),
    "Predicted_Close": future_predictions
})

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("📊 Forecast Insights")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Highest Forecast",
    f"${future_df['Predicted_Close'].max():.2f}"
)

col2.metric(
    "Lowest Forecast",
    f"${future_df['Predicted_Close'].min():.2f}"
)

col3.metric(
    "Average Forecast",
    f"${future_df['Predicted_Close'].mean():.2f}"
)

st.markdown("---")

# -----------------------------
# FORECAST CHART
# -----------------------------
st.subheader("🚀 30-Day Forecast")

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
    title="Predicted Apple Stock Prices (Next 30 Days)",
    template="plotly_white",
    height=550
)

st.plotly_chart(fig_future, use_container_width=True)

# -----------------------------
# FORECAST TABLE
# -----------------------------
st.subheader("📅 Forecast Table")

st.dataframe(
    future_df,
    use_container_width=True
)

# -----------------------------
# DOWNLOAD BUTTON
# -----------------------------
csv = future_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Forecast CSV",
    data=csv,
    file_name="Apple_30Day_Forecast.csv",
    mime="text/csv"
)

st.markdown("---")

# -----------------------------
# CONCLUSION
# -----------------------------
st.subheader("✅ Conclusion")

st.success(
"""
The model forecasts Apple's stock prices for the next
30 days using historical patterns and engineered features.

This project demonstrates:
✔ Time Series Forecasting
✔ Feature Engineering
✔ Machine Learning
✔ Data Visualization
✔ Streamlit Deployment
"""
)

st.markdown(
    """
    <center>
    <h4>Developed by Shailja Singh</h4>
    </center>
    """,
    unsafe_allow_html=True
)