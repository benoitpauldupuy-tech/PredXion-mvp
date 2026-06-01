import streamlit as st
import pandas as pd
import numpy as np
import joblib
st.title("📊 PredXion MVP")
model = joblib.load("model.pkl")

mode = st.radio(
    "Sources Data pour prédiction",
    ["📊 Manuel", "📁 Excel"]
)

if mode == "📊 Manuel":

    cat = st.selectbox("Catégorie Produit", X["Catégorie Produit"].unique())
    typ = st.selectbox("Typologie Produit", X["Typologie Produit"].unique())
    gamme = st.selectbox("Gamme PV", X["Gamme PV"].unique())
    saison = st.selectbox("Saison", X["Saison"].unique())
    prix = st.number_input("Prix de Vente", min_value=0.0)

    input_df = pd.DataFrame([{
        "Catégorie Produit": cat,
        "Typologie Produit": typ,
        "Gamme PV": gamme,
        "Saison": saison,
        "Prix de Vente": prix
    }])

    if st.button("Prédire"):

        pred_log = model.predict(input_df)
        pred = np.expm1(pred_log)

        st.write("📦 Ventes prévues :", int(pred[0]))

  if mode == "📁 Excel":

    file = st.file_uploader("Upload fichier Excel", type=["xlsx"])

    if file is not None:

        df_input = pd.read_excel(file)

        st.write("Aperçu :", df_input.head())

        pred_log = model.predict(df_input)
        pred = np.expm1(pred_log)

        df_input["prediction"] = pred

        st.write(df_input)

        st.download_button(
            "Télécharger résultats",
            df_input.to_csv(index=False),
            "predictions.csv"
        )
