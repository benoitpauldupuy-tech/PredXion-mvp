import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
df = pd.read_excel("BDD_Ventes_NafNaf_MachineLearning.xlsx") # ou csv
st.title("📊 PredXion MVP")

def get_list(col):
    return sorted(df[col].dropna().unique())

saison_list = get_list("Saison")
annees_list = get_list("Années")
reconduit_list = get_list("Reconduit")

cat_list = get_list("Catégorie Produit")
subcat_list = get_list("Sous-Catégorie Produit")
typ_list = get_list("Typologie Produit")

matiere_list = get_list("Matière")
groupe_couleur_list = get_list("Groupe Couleur")
type_couleur_list = get_list("Type Couleur")

mois_list = get_list("Mois Implantation")
gamme_list = get_list("Gamme PV")

mode = st.radio("Mode", ["📊 Manuel", "📁 Excel"])

if mode == "📊 Manuel":

cat = st.selectbox("Catégorie Produit", ["Sélectionnez"] + cat_list)
subcat = st.selectbox("Sous-Catégorie Produit", ["Sélectionnez"] + subcat_list)
typ = st.selectbox("Typologie Produit", ["Sélectionnez"] + typ_list)

matiere = st.selectbox("Matière", ["Sélectionnez"] + matiere_list)
groupe_couleur = st.selectbox("Groupe Couleur", ["Sélectionnez"] + groupe_couleur_list)
type_couleur = st.selectbox("Type Couleur", ["Sélectionnez"] + type_couleur_list)

mois = st.selectbox("Mois Implantation", ["Sélectionnez"] + mois_list)
gamme = st.selectbox("Gamme PV", ["Sélectionnez"] + gamme_list)

parc = st.selectbox("Parc Magasin", ["Sélectionnez"] + parc_list)

annees = st.text_input("Année (ex: 2026)")
prix = st.number_input("Prix de Vente (€)", min_value=0.0)


    input_df = pd.DataFrame([{
        "Saison": saison,
        "Années": annees,
        "Reconduit": reconduit,
        "Catégorie Produit": cat,
        "Sous-Catégorie Produit": subcat,
        "Typologie Produit": typ,
        "Matière": matiere,
        "Groupe Couleur": groupe_couleur,
        "Type Couleur": type_couleur,
        "Mois Implantation": mois,
        "Gamme PV": gamme,
        "Prix de Vente": prix,
        "Parc Magasin": parc
    }])


    if st.button("Prédire"):

        pred_log = model.predict(input_df)
        pred = np.expm1(pred_log)

        st.success(f"📦 Ventes prévues : {int(pred[0])}")
parc_list = get_list("Parc Magasin")

