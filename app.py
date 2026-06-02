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

[data-testid="stMetric"]{
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:16px;
    padding:15px;
}

h1,h2,h3,p,label{
    color:white !important;
}

.stButton>button{
    background:#D4AF37;
    color:black;
    font-weight:bold;
    border:none;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD
# =========================

model = joblib.load("model.pkl")
df = pd.read_excel("BDD_Ventes_NafNaf_MachineLearning.xlsx")

# =========================
# HEADER
# =========================

st.markdown("""
<h1 style='text-align:center;color:#D4AF37;font-size:60px;margin-top:40px;'>
PredXion
</h1>
<p style='text-align:center;color:#ccc;'>
RETAIL FORECASTING PLATFORM
</p>
""", unsafe_allow_html=True)

# =========================
# FUNCTIONS
# =========================

def get_list(col):
    return sorted(df[col].dropna().astype(str).unique())

def build_features(data):
    data = data.copy()
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

expected_cols = [
    "Saison","Années","Reconduit","Catégorie Produit","Sous-Catégorie Produit",
    "Typologie Produit","Matière","Groupe Couleur","Type Couleur","Couleur",
    "Thème","Mois Implantation","Gamme PV","Prix de Vente","Parc Magasin",
    "Cat_Saison","Typo_Saison","Cat_GammePV"
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

        if "Sélectionnez" in [saison, reconduit, cat, subcat, typ, matiere,
                              groupe_couleur, type_couleur, mois, gamme, parc]:
            st.warning("⚠️ Complète tous les champs")
            st.stop()

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

        pred = float(np.expm1(model.predict(input_df)[0]))

        avg_last_year = df[
            (df["Catégorie Produit"] == cat) &
            (df["Saison"] == saison)
        ]["Quantités Vendues"].mean()

        if np.isnan(avg_last_year):
            ratio = 1
            ecart_pct = 0
        else:
            ratio = pred / avg_last_year
            ecart_pct = abs(pred - avg_last_year) / avg_last_year * 100

        # =========================
        # SCORE STABLE (IMPORTANT FIX)
        # =========================

        ml_s = ml_score(cat, typ)

        business_score = 100 - min(100, ecart_pct * 1.2)

        final_score = (0.6 * ml_s + 0.4 * business_score)

        # =========================
        # FEU METIER CORRIGÉ
        # =========================

        if final_score > 75 or ecart_pct < 15:
            feu = "🟢"
            label = "Achat sécurisé"
            mode_reco = "GREEN"
        
        elif final_score < 40 or ecart_pct > 30:
            feu = "🔴"
            label = "Achat risqué"
            mode_reco = "RED"
        
        else:
            feu = "🟠"
            label = "À vérifier"
            mode_reco = "ORANGE"
        
        # =========================
        # RECOMMANDATION ACHAT (RULE ENGINE)
        # =========================
        
        if np.isnan(avg_last_year):
            avg_last_year = 0
        
        if mode_reco == "GREEN":
            reco = pred
        
        elif mode_reco == "ORANGE":
            reco = (pred + avg_last_year) / 2
        
        else:  # RED
            reco = avg_last_year * 0.90  # 90% du marché historique (prudence)
        
        commentaire = (
            f"{label} | "
            f"écart vs marché: {ecart_pct:.1f}% | "
            f"ratio: {ratio:.2f}"
        )
        
        # =========================
        # UI - LIGNE 1 (ALIGNÉE)
        # =========================
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("📦 Ventes prévues", f"{int(pred):,}".replace(",", " "))
        col2.metric("🎯 Score confiance", f"{int(final_score)}/100")
        col3.metric("📊 Historique N-1", "N/A" if np.isnan(avg_last_year) else f"{int(avg_last_year):,}")
        col4.metric("📈 Ratio marché", f"{ratio:.2f}")
        
        # =========================
        # UI - LIGNE 2 (ALIGNÉE)
        # =========================
        
        col1, col2 = st.columns([1, 2])
        
        col1.markdown(f"## {feu} {label}")
        
        col2.metric(
            "🧠 Recommandation achat",
            f"{int(reco):,}".replace(",", " ")
        )
        
        st.markdown(
            f"""
            <div style="
                margin-top:10px;
                padding:12px;
                border-radius:12px;
                background:rgba(255,255,255,0.08);
                color:white;
                font-size:14px;
            ">
                💬 {commentaire}
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# EXCEL MODE
# =========================

if mode == "📁 Excel":

    file = st.file_uploader("Upload Excel", type=["xlsx"])

    if file:

        df_up = pd.read_excel(file)

        if st.button("Lancer prédictions"):

            df_up = build_features(df_up)
            df_up = df_up[expected_cols]

            preds = np.expm1(model.predict(df_up))
            df_up["Prediction"] = preds

            df_up["Feu"] = np.where(preds > preds.mean(), "🟢", "🟠")

            st.dataframe(df_up)

            csv = df_up.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger", csv, "predictions.csv")
