import streamlit as st
import joblib

st.title("📊 PredXion MVP")

model = joblib.load("model.pkl")

st.success("✅ Modèle chargé avec succès")
