import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CHARGEMENT
# =========================

model = joblib.load("model.pkl")
df = pd.read_excel("BDD_Ventes_NafNaf_MachineLearning.xlsx")

st.markdown(
"""
<h1>PREDXION</h1>
<p style='text-align:center;color:yellow>
Retail Forecasting Platform
</p>
""",
unsafe_allow_html=True
)

st.markdown("""
<style>

.main {
    background-color: white;
}

h1 {
    text-align:center;
    color:black;
}

.block-container {
    padding-top:2rem;
    max-width:1200px;
}

[data-testid="stMetric"] {
    border:1px solid #EAEAEA;
    border-radius:12px;
    padding:20px;
}

.stButton > button {
    width:100%;
    border-radius:8px;
    height:50px;
    background-color:black;
    color:white;
    border:none;
}

.stButton > button:hover {
    background-color:#222222;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 40%,
        #334155 100%
    );
}

h1 {
    text-align: center;
    color: white;
}

p {
    color: #d1d5db;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 20px;
}

.stButton > button {
    background: white;
    color: black;
    border-radius: 10px;
    border: none;
    height: 50px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)
# =========================
# FONCTIONS
# =========================

def get_list(col):
    return sorted(df[col].dropna().astype(str).unique())

def build_features(data):
    data["Cat_Saison"] = data["Catégorie Produit"].astype(str) + "_" + data["Saison"].astype(str)
    data["Typo_Saison"] = data["Typologie Produit"].astype(str) + "_" + data["Saison"].astype(str)
    data["Cat_GammePV"] = data["Catégorie Produit"].astype(str) + "_" + data["Gamme PV"].astype(str)
    return data

# =========================
# LISTES
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
# FEATURES ATTENDUES MODEL
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

# =========================
# MODE MANUEL
# =========================

if mode == "📊 Manuel":

    saison = st.selectbox("Saison", ["Sélectionnez"] + saison_list)
    annees = st.text_input("Année (ex: 2026)")
    reconduit = st.selectbox("Reconduit", ["Sélectionnez"] + reconduit_list)

    cat = st.selectbox("Catégorie Produit", ["Sélectionnez"] + cat_list)
    subcat = st.selectbox("Sous-Catégorie Produit", ["Sélectionnez"] + subcat_list)
    typ = st.selectbox("Typologie Produit", ["Sélectionnez"] + typ_list)

    matiere = st.selectbox("Matière", ["Sélectionnez"] + matiere_list)
    groupe_couleur = st.selectbox("Groupe Couleur", ["Sélectionnez"] + groupe_couleur_list)
    type_couleur = st.selectbox("Type Couleur", ["Sélectionnez"] + type_couleur_list)

    couleur = st.text_input("Couleur")
    theme = st.text_input("Thème")

    mois = st.selectbox("Mois Implantation", ["Sélectionnez"] + mois_list)
    gamme = st.selectbox("Gamme PV", ["Sélectionnez"] + gamme_list)

    prix = st.number_input("Prix de Vente (€)", min_value=0.0)
    parc = st.selectbox("Parc Magasin", ["Sélectionnez"] + parc_list)

    if st.button("Prédire"):

        required = [saison, reconduit, cat, subcat, typ, matiere,
                    groupe_couleur, type_couleur, mois, gamme, parc]

        if "Sélectionnez" in required:
            st.warning("⚠️ Merci de compléter tous les champs")

        else:

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
                "Couleur": couleur,
                "Thème": theme,
                "Mois Implantation": mois,
                "Gamme PV": gamme,
                "Prix de Vente": prix,
                "Parc Magasin": parc
            }])

            input_df = build_features(input_df)
            input_df = input_df[expected_cols]

            pred_log = model.predict(input_df)
            pred = np.expm1(pred_log)

            st.metric(
                label="Prévision de ventes",
                value=f"{int(pred[0]):,}".replace(",", " ")
)

# =========================
# MODE EXCEL
# =========================

if mode == "📁 Excel":

    file = st.file_uploader("Uploader un fichier Excel", type=["xlsx"])

    if file is not None:

        df_upload = pd.read_excel(file)

        st.write("Aperçu")
        st.dataframe(df_upload.head())

        if st.button("Lancer prédictions"):

            df_upload = build_features(df_upload)
            df_upload = df_upload[expected_cols]

            preds_log = model.predict(df_upload)
            preds = np.expm1(preds_log)

            df_upload["Prediction"] = preds

            st.success("Prédictions terminées")

            st.dataframe(df_upload)

            csv = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger résultat", csv, "predictions.csv", "text/csv")
