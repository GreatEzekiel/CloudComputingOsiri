import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Layout
st.set_page_config(page_title="Yahoo Stock Predictor", page_icon="📈")

st.title("📈 Stock Transaction Predictor (CSIS505)")
st.markdown("""
### OSIRI UNIVERSITY - APPLICATION ASSESSMENT
This application uses the **Extra Trees Classifier** to predict transaction types.
""")

# Sidebar Inputs
st.sidebar.header("Input Features")
def get_user_input():
    amount = st.sidebar.number_input("Amount", value=5000.0)
    price = st.sidebar.number_input("Reported Price", value=150.0)
    usd_val = st.sidebar.number_input("USD Value", value=750000.0)
    
    features = pd.DataFrame({
        'amount': [amount],
        'reportedPrice': [price],
        'usdValue': [usd_val]
    })
    return features

input_data = get_user_input()

# Load Model & Scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Scale and Predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)

    # Display Results
    st.subheader("Prediction Analysis")
    label = "BUY" if prediction[0] == 1 else "SELL"
    st.success(f"The predicted transaction type is: **{label}**")
    
    # Confidence Metrics
    st.write(f"Confidence Level: **{np.max(prob)*100:.2f}%**")
    st.bar_chart(pd.DataFrame(prob, columns=['SELL', 'BUY']).T)

except FileNotFoundError:
    st.error("Error: Model files not found. Run 'train_best_model.py' first.")