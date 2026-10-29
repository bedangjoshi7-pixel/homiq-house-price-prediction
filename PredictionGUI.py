
import streamlit as st
from model import predict_price

st.title("House Price Predictor")

st.write(
    "Enter the size of a house to estimate its price."
)

size = st.number_input(
    "House size (square feet)",
    min_value=100,
    max_value=100000,
    value=1500,
    step=100
)

if st.button("Predict Price"):
    prediction = predict_price(size)

    st.success(
        f"Estimated price: {prediction:.2f} thousand Dollars or {prediction*1000*152.19} Nepali Rupees"
        "in the dataset's currency units"
    )
