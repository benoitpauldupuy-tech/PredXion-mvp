import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("📊 PredXion MVP")

mode = st.radio("Mode de prédiction", ["📊 Manuel", "📁 Excel"])

# =========================
# LISTES (à adapter si besoin)
# =========================
cat_list = sorted(pd.DataFrame().columns)  # temporaire safe
typ_list = []
gamme_list = []
saison_list = []

# =========================
# MODE MANUEL
# =========================
if mode == "📊 Manuel":

    st.write("Mode manuel")

    cat = st.text_input("Catégorie Produit")
    typ = st.text_input("Typologie Produit")
    gamme = st.text_input("Gamme PV")
    saison = st.text_input("Saison")
    prix = st.number_input("Prix de Vente", min_value=0.0)

    if st.button("Prédire"):

        input_df = pd.DataFrame([{
            "Catégorie Produit": cat,
            "Typologie Produit": typ,
            "Gamme PV": gamme,
            "Saison": saison,
            "Prix de Vente": prix
        }])

        pred_log = model.predict(input_df)
        pred = np.expm1(pred_log)

        st.success(f"📦 Ventes prévues : {int(pred[0])}")

# =========================
# MODE EXCEL
# =========================
elif mode == "📁 Excel":

    file = st.file_uploader("Upload Excel", type=["xlsx"])

    if file is not None:

        df = pd.read_excel(file)

        st.write(df.head())

        pred_log = model.predict(df)
        pred = np.expm1(pred_log)

        df["prediction"] = pred

        st.write(df)
