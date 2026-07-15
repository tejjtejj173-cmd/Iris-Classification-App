import streamlit as st
import joblib
classifier = joblib.load("dt_model.pkl")

st.title("Iris Classification App")


st.write("used for iris classification")

sl=st.number_input(label="Enter Sepal Length")
sw=st.number_input(label="Enter Sepal Width")
pl=st.number_input(label="Enter Petal Length")
pw=st.number_input(label="Enter Petal Width")


if st.button("Predict"):
    result=classifier.predict([[sl,sw,pl,pw]])
    st.success(result[0])
