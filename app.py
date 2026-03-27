
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Fixed ML Streamlit App")

# Sample training data (replace with real data.csv if available)
data = pd.DataFrame({
    "age":[1,2,3,4],
    "income":[1,2,3,4],
    "spend":[0,1,1,0]
})

X = data[["age","income"]]
y = data["spend"]

model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.header("Input Features")

age = st.sidebar.slider("Age",0,5,1)
income = st.sidebar.slider("Income",0,5,1)

input_df = pd.DataFrame({
    "age":[age],
    "income":[income]
})

# Align columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(input_df)

st.write("### Prediction:", prediction[0])
