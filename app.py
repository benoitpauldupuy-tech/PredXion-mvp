import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================

# CHARGEMENT

# =========================

model = joblib.load("model.pkl")

df = pd.read_excel("BDD_Ventes_NafNaf_MachineLearning.xlsx")

st.title("📊 PredXion MVP")

# =========================

# FONCTION LISTES

# =========================

def get_list(col):
    return sorted(df[col].dropna().astype(str).unique())

# =========================

# LISTES DÉROULANTES

# =========================

saison_list = get_list("Saison")
reconduit_list = get_list("Reconduit")

cat_list = get_list("Catégorie Produit")
subcat_list = get_list("Sous-Catégorie Produit")
typ_list = get_list("Typologie Produit")

matiere_list = get_list("Matière")
groupe_couleur_list = get_list("Groupe Couleur")
type_couleur_list = get_list("Type Couleur")

mois_list = get_list("Mois Implantation")
gamme_list = get_list("Gamme PV")
parc_list = get_list("Parc Magasin")

# =========================

# MODE

# =========================

mode = st.radio("Mode", ["📊 Manuel", "📁 Excel"])

# =========================

# MODE MANUEL

# =========================

if mode == "📊 Manuel":

    saison = st.selectbox(
        "Saison",
        ["Sélectionnez"] + saison_list
    )
    
    annees = st.text_input(
        "Année (ex : 2026)"
    )
    
    reconduit = st.selectbox(
        "Reconduit",
        ["Sélectionnez"] + reconduit_list
    )
    
    cat = st.selectbox(
        "Catégorie Produit",
        ["Sélectionnez"] + cat_list
    )
    
    subcat = st.selectbox(
        "Sous-Catégorie Produit",
        ["Sélectionnez"] + subcat_list
    )
    
    typ = st.selectbox(
        "Typologie Produit",
        ["Sélectionnez"] + typ_list
    )
    
    matiere = st.selectbox(
        "Matière",
        ["Sélectionnez"] + matiere_list
    )
    
    groupe_couleur = st.selectbox(
        "Groupe Couleur",
        ["Sélectionnez"] + groupe_couleur_list
    )
    
    type_couleur = st.selectbox(
        "Type Couleur",
        ["Sélectionnez"] + type_couleur_list
    )
    
    couleur = st.text_input(
        "Couleur"
    )
    
    theme = st.text_input(
        "Thème"
    )
    
    mois = st.selectbox(
        "Mois Implantation",
        ["Sélectionnez"] + mois_list
    )
    
    gamme = st.selectbox(
        "Gamme PV",
        ["Sélectionnez"] + gamme_list
    )
    
    prix = st.number_input(
        "Prix de Vente (€)",
        min_value=0.0
    )
    
    parc = st.selectbox(
        "Parc Magasin",
        ["Sélectionnez"] + parc_list
    )

if st.button("Prédire"):

    champs_select = [
        saison,
        reconduit,
        cat,
        subcat,
        typ,
        matiere,
        groupe_couleur,
        type_couleur,
        mois,
        gamme,
        parc
    ]

    if "Sélectionnez" in champs_select:

        st.warning(
            "⚠️ Merci de compléter tous les champs."
        )

    else:

        input_df = pd.DataFrame([{
            "Saison": str(saison),
            "Années": str(annees),
            "Reconduit": str(reconduit),
            "Catégorie Produit": str(cat),
            "Sous-Catégorie Produit": str(subcat),
            "Typologie Produit": str(typ),
            "Matière": str(matiere),
            "Groupe Couleur": str(groupe_couleur),
            "Type Couleur": str(type_couleur),
            "Couleur": str(couleur),
            "Thème": str(theme),
            "Mois Implantation": str(mois),
            "Gamme PV": str(gamme),
            "Prix de Vente": float(prix),
            "Parc Magasin": str(parc)
        }])

        # =========================
        # FEATURES DÉRIVÉES
        # =========================

        input_df["Cat_Saison"] = (
            input_df["Catégorie Produit"]
            + "_"
            + input_df["Saison"]
        )

        input_df["Typo_Saison"] = (
            input_df["Typologie Produit"]
            + "_"
            + input_df["Saison"]
        )

        input_df["Cat_GammePV"] = (
            input_df["Catégorie Produit"]
            + "_"
            + input_df["Gamme PV"]
        )

        # =========================
        # ORDRE EXACT DU TRAINING
        # =========================

        expected_cols = [
            "Saison",
            "Années",
            "Reconduit",
            "Catégorie Produit",
            "Sous-Catégorie Produit",
            "Typologie Produit",
            "Matière",
            "Groupe Couleur",
            "Type Couleur",
            "Couleur",
            "Thème",
            "Mois Implantation",
            "Gamme PV",
            "Prix de Vente",
            "Parc Magasin",
            "Cat_Saison",
            "Typo_Saison",
            "Cat_GammePV"
        ]

        input_df = input_df[expected_cols]

        pred_log = model.predict(input_df)

        pred = np.expm1(pred_log)

        st.success(
            f"📦 Ventes prévisionnelles : {int(pred[0])}"
        )
