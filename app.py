import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="PredXion MVP", layout="wide")

model = joblib.load("model.pkl")
df = pd.read_excel("BDD_Ventes_NafNaf_MachineLearning.xlsx")

# =========================
# STYLE UI
# =========================

st.markdown("""
<h1 style='text-align:center;color:#D4AF37;font-size:60px;font-weight:300;letter-spacing:4px;margin-top:40px;'>
PredXion
</h1>
<p style='text-align:center;color:#ccc;font-size:16px;letter-spacing:2px;'>
RETAIL FORECASTING PLATFORM
</p>
""", unsafe_allow_html=True)

# =========================
# FUNCTIONS
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
# FEATURES MODEL
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
# ML SCORE BASE
# =========================

mae_cat = {
    "T-Shirts": 2696.23,
    "Jupes": 2029.69,
    "Vestes": 2016.68,
    "Chemises": 1930.53,
    "Pulls": 1889.54,
    "Pantalons": 1869.48,
    "Robes": 1620.77,
    "Manteaux": 841.84
}

mae_typo = {
    "Essentiel": 2107.97,
    "Mode": 1516.86,
    "Image": 365.82
}

def norm(x, mn, mx):
    return 100 * (1 - (x - mn) / (mx - mn + 1e-9))

def ml_score(cat, typ):
    cat_s = norm(mae_cat.get(cat, np.mean(list(mae_cat.values()))),
                 min(mae_cat.values()), max(mae_cat.values()))

    typ_s = norm(mae_typo.get(typ, np.mean(list(mae_typo.values()))),
                 min(mae_typo.values()), max(mae_typo.values()))

    return 0.5 * cat_s + 0.5 * typ_s

# =========================
# MANUAL MODE
# =========================

if mode == "📊 Manuel":

    saison = st.selectbox("Saison", ["Sélectionnez"] + saison_list)
    annees = st.text_input("Année")
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

        required = [saison, reconduit, cat, subcat, typ,
                    matiere, groupe_couleur, type_couleur,
                    mois, gamme, parc]

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

            # =========================
            # BUSINESS SCORE (SAISON)
            # =========================

            avg_same = df[
                (df["Catégorie Produit"] == cat) &
                (df["Saison"] == saison)
            ]["Quantités Vendues"].mean()

            if np.isnan(avg_same):
                business_score = 60
            else:
                diff = abs(pred[0] - avg_same)

                if diff <= 1100:
                    business_score = 90
                elif diff <= 1900:
                    business_score = 60
                elif diff <= 3000:
                    business_score = 35
                else:
                    business_score = 15

            # =========================
            # FINAL SCORE
            # =========================

            final = 0.5 * ml_score(cat, typ) + 0.5 * business_score
            final = max(0, min(100, final))

            if final >= 70:
                label = "🟢 Achat sécurisé"
            elif final >= 45:
                label = "🟠 À vérifier"
            else:
                label = "🔴 Risque élevé"

            st.metric("📦 Ventes prévues", f"{int(pred[0]):,}".replace(",", " "))
            st.metric("🎯 Score confiance", f"{int(final)}/100")
            st.markdown(f"### {label}")

# =========================
# EXCEL MODE
# =========================

if mode == "📁 Excel":

    file = st.file_uploader("Uploader Excel", type=["xlsx"])

    if file:

        df_up = pd.read_excel(file)

        if st.button("Lancer prédictions"):

            df_up = build_features(df_up)
            df_up = df_up[expected_cols]

            preds = np.expm1(model.predict(df_up))
            df_up["Prediction"] = preds

            st.success("Prédictions terminées")
            st.dataframe(df_up)

            csv = df_up.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger", csv, "predictions.csv", "text/csv")
