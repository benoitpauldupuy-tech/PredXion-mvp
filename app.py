import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIG UI
# =========================

st.set_page_config(page_title="PredXion MVP", layout="wide")

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #172554 40%,
        #1e293b 100%
    );
}

.block-container{
    padding-top:2rem;
}

h1,h2,h3,p,label {
    color:white !important;
}

[data-testid="stMetric"]{
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:16px;
    padding:12px;
}

.stButton>button{
    background:#D4AF37;
    color:black;
    font-weight:bold;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL + DATA
# =========================

model = joblib.load("model.pkl")
df = pd.read_excel("BDD_Ventes_NafNaf_MachineLearning.xlsx")

# =========================
# HEADER
# =========================

st.markdown("""
<h1 style='text-align:center;color:#D4AF37;font-size:55px;margin-top:30px;'>
PredXion
</h1>
<p style='text-align:center;color:#d1d5db;margin-bottom:30px;'>
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

def ml_score(cat, typ):

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

    cat_s = norm(mae_cat.get(cat, np.mean(list(mae_cat.values()))),
                 min(mae_cat.values()), max(mae_cat.values()))

    typ_s = norm(mae_typo.get(typ, np.mean(list(mae_typo.values()))),
                 min(mae_typo.values()), max(mae_typo.values()))

    return 0.5 * cat_s + 0.5 * typ_s

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
    "Saison","Années","Reconduit","Catégorie Produit",
    "Sous-Catégorie Produit","Typologie Produit","Matière",
    "Groupe Couleur","Type Couleur","Couleur","Thème",
    "Mois Implantation","Gamme PV","Prix de Vente",
    "Parc Magasin","Cat_Saison","Typo_Saison","Cat_GammePV"
]

# =========================
# MANUEL
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
            st.warning("Merci de compléter tous les champs")

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

            pred = np.expm1(model.predict(input_df))[0]

            # =========================
            # HISTORIQUE 3 ANS PONDÉRÉ
            # =========================

            hist = df[
                (df["Catégorie Produit"] == cat) &
                (df["Saison"] == saison)
            ]

            hist["Années"] = pd.to_numeric(hist["Années"], errors="coerce")

            v25 = hist[hist["Années"] == 2025]["Quantités Vendues"].mean()
            v24 = hist[hist["Années"] == 2024]["Quantités Vendues"].mean()
            v23 = hist[hist["Années"] == 2023]["Quantités Vendues"].mean()

            historique = (
                (0 if np.isnan(v25) else v25 * 0.5) +
                (0 if np.isnan(v24) else v24 * 0.3) +
                (0 if np.isnan(v23) else v23 * 0.2)
            )

            if np.isnan(historique) or historique == 0:
                historique = pred

            # =========================
            # METRICS
            # =========================

            ratio = pred / (historique + 1e-9)
            ecart_pct = abs(pred - historique) / (historique + 1e-9) * 100

            ml_s = ml_score(cat, typ)

            # =========================
            # SCORE
            # =========================

            if ecart_pct < 10:
                business = 95
            elif ecart_pct < 20:
                business = 85
            elif ecart_pct < 30:
                business = 70
            elif ecart_pct < 40:
                business = 55
            else:
                business = 35

            final_score = 0.55 * business + 0.45 * ml_s

            # =========================
            # FEU
        # =========================
        # FEU (RULE STRICTE)
        # =========================
        
        if final_score >= 75:
            feu, label = "🟢", "Achat sécurisé"
        
        elif final_score >= 40:
            feu, label = "🟠", "À vérifier"
        
        else:
            feu, label = "🔴", "Achat risqué"
        
        # =========================
        # RECOMMANDATION
        # =========================
        
        if feu == "🟢":
            reco = pred
        
        elif feu == "🟠":
            reco = (pred + historique) / 2
        
        else:
            reco = historique * (0.9 if pred < historique else 1.1)
        
        # =========================
        # COMMENTAIRE MÉTIER
        # =========================
        
        if final_score >= 75:
            commentaire = "Prévision cohérente avec le marché et la saisonnalité."
        
        elif final_score >= 40:
        
            if pred > historique:
                commentaire = "Signal de hausse vs historique saisonnier. Vérifier capacité stock."
            else:
                commentaire = "Baisse vs historique saisonnier. Risque de surstock potentiel."
        
        else:
        
            if pred > historique:
                commentaire = "Désalignement fort avec historique. Forte demande attendue vs capacité planifiée."
            else:
                commentaire = "Chute significative vs historique. Risque élevé de surstock."
        
        # =========================
        # UI (RENOMMAGE IMPORTANT)
        # =========================
        
        c1, c2, c3, c4 = st.columns(4)
        
        c1.metric("Ventes prévues", f"{int(pred):,}")
        c2.metric("Score confiance", f"{int(final_score)}/100")
        c3.metric("Référence saison 3 ans", f"{int(historique):,}")
        c4.metric("Ratio marché", f"{ratio:.2f}")
        
        c5, c6 = st.columns([1, 2])
        
        with c5:
            st.markdown(f"## {feu} {label}")
        
        with c6:
            st.metric("Recommandation achat", f"{int(reco):,}")
        
        # COMMENTAIRE (IMPORTANT UX)
        st.markdown("### 💬 Analyse métier")
        st.info(commentaire)

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

            st.dataframe(df_up)

            csv = df_up.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger", csv, "predictions.csv", "text/csv")
