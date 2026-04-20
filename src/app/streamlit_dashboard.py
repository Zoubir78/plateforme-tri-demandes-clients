from __future__ import annotations

import ast
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

# Permet d'importer proprement le package src depuis Streamlit
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from google.cloud import storage
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from wordcloud import WordCloud

from src.app.ml.predict import TicketModel
from src.app.pipeline.pipeline_service import run_pipeline


# -----------------------------
# CONFIG
# -----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "clean_complaints")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "logreg-v1")

st.set_page_config(page_title="Signal Conso DASHBOARD", page_icon="📊", layout="wide")
#st.caption("Signalements, catégories, mots-clés, performance modèle et pipeline GCS.")


# -----------------------------
# UTILS
# -----------------------------
STOPWORDS_FR = {
    "a",
    "alors",
    "au",
    "aucuns",
    "aussi",
    "autre",
    "avant",
    "avec",
    "avoir",
    "bon",
    "car",
    "ce",
    "cela",
    "ces",
    "ceux",
    "chaque",
    "ci",
    "comme",
    "comment",
    "dans",
    "des",
    "du",
    "dedans",
    "dehors",
    "depuis",
    "devrait",
    "doit",
    "donc",
    "dos",
    "début",
    "elle",
    "elles",
    "en",
    "encore",
    "essai",
    "est",
    "et",
    "eu",
    "fait",
    "faites",
    "fois",
    "font",
    "hors",
    "ici",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "là",
    "ma",
    "maintenant",
    "mais",
    "mes",
    "mine",
    "moins",
    "mon",
    "mot",
    "même",
    "ni",
    "nommés",
    "notre",
    "nous",
    "nouveaux",
    "ou",
    "où",
    "par",
    "parce",
    "parole",
    "pas",
    "personnes",
    "peu",
    "peut",
    "pièce",
    "plupart",
    "pour",
    "pourquoi",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "sans",
    "ses",
    "seulement",
    "si",
    "sien",
    "son",
    "sont",
    "sous",
    "soyez",
    "sujet",
    "sur",
    "ta",
    "tandis",
    "te",
    "tes",
    "ton",
    "tous",
    "tout",
    "trop",
    "très",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "vu",
    "ça",
    "étaient",
    "état",
    "étions",
    "été",
    "être",
}

BOOL_TRUE_VALUES = {"1", "true", "t", "yes", "y", "oui", "vrai", "on"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_multivalue(value: Any) -> list[str]:
    """Parse des valeurs de type liste, tuple, chaîne JSON-like ou chaîne simple."""
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        raw = str(value).strip()
        if not raw:
            return []

        if raw.startswith("[") or raw.startswith("(") or raw.startswith("{"):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple, set)):
                    items = list(parsed)
                else:
                    items = [parsed]
            except (ValueError, SyntaxError):
                items = [part.strip() for part in raw.strip("[](){} ").split(",") if part.strip()]
        else:
            items = [raw]

    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _bool_series(series: pd.Series) -> pd.Series:
    def _to_bool(value: Any) -> bool:
        if _is_missing(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in BOOL_TRUE_VALUES

    return series.apply(_to_bool)


def _normalize_label(text: str) -> str:
    return " ".join(str(text).strip().split())


def _department_label(row: pd.Series) -> str:
    dep_code = row.get("dep_code")
    dep_name = row.get("dep_name")

    dep_code_txt = _normalize_label(dep_code) if not _is_missing(dep_code) else ""
    dep_name_txt = _normalize_label(dep_name) if not _is_missing(dep_name) else ""

    if dep_code_txt and dep_name_txt:
        return f"{dep_code_txt} - {dep_name_txt}"
    if dep_code_txt:
        return dep_code_txt
    if dep_name_txt:
        return dep_name_txt
    return "Inconnu"


def _frequency_from_cells(df: pd.DataFrame, column: str, limit: int = 15) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=int)

    counter: Counter[str] = Counter()
    display_name: dict[str, str] = {}

    for raw in df[column].dropna():
        values = _parse_multivalue(raw)
        if not values:
            values = [str(raw)]

        for value in values:
            label = _normalize_label(value)
            if not label:
                continue
            key = label.casefold()
            counter[key] += 1
            display_name.setdefault(key, label)

    if not counter:
        return pd.Series(dtype=int)

    ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)[:limit]
    return pd.Series({display_name[key]: count for key, count in ordered})


def _keyword_frequency(df: pd.DataFrame, limit: int = 15) -> pd.Series:
    """Calcule les mots-clés les plus fréquents à partir de tags, subcategories et clean_text."""
    counter: Counter[str] = Counter()
    display_name: dict[str, str] = {}

    candidate_columns = [c for c in ["tags", "subcategories", "clean_text"] if c in df.columns]
    if not candidate_columns:
        return pd.Series(dtype=int)

    for _, row in df[candidate_columns].iterrows():
        tokens: list[str] = []

        if "tags" in row and not _is_missing(row.get("tags")):
            tokens.extend(_parse_multivalue(row.get("tags")))
        if "subcategories" in row and not _is_missing(row.get("subcategories")):
            tokens.extend(_parse_multivalue(row.get("subcategories")))

        if not tokens and "clean_text" in row and not _is_missing(row.get("clean_text")):
            raw_text = str(row.get("clean_text"))
            tokens = [
                tok
                for tok in raw_text.lower().split()
                if len(tok) > 2 and tok not in STOPWORDS_FR
            ]

        for tok in tokens:
            label = _normalize_label(tok)
            if not label:
                continue
            key = label.casefold()
            counter[key] += 1
            display_name.setdefault(key, label)

    if not counter:
        return pd.Series(dtype=int)

    ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)[:limit]
    return pd.Series({display_name[key]: count for key, count in ordered})

# -----------------------------
# 🎨 STYLE GLOBAL
# -----------------------------
PRIMARY_COLOR = "#1f77b4"

st.markdown("""
<style>
.main-title {font-size:32px;font-weight:700;color:#1f77b4;}
.section-title {font-size:22px;font-weight:600;margin-top:20px;}
.card {padding:15px;border-radius:10px;background:#f9f9f9;box-shadow:0px 2px 6px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Dashboard Signal Conso</div>', unsafe_allow_html=True)
st.caption("Analyse data + IA + monitoring avancé")


# -----------------------------
# GCS HELPERS
# -----------------------------
@st.cache_resource
def get_gcs_client() -> storage.Client:
    return storage.Client()


@st.cache_data(ttl=60)
def list_blob_names(prefix: str) -> list[str]:
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]


@st.cache_data(ttl=60)
def get_latest_blob_name(prefix: str) -> str | None:
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        return None

    fallback_dt = datetime.min.replace(tzinfo=timezone.utc)
    latest = max(blobs, key=lambda blob: blob.updated or fallback_dt)
    return latest.name


@st.cache_data(ttl=60)
def load_latest_dataset() -> tuple[pd.DataFrame, str | None]:
    latest_name = get_latest_blob_name("processed/")
    if not latest_name:
        return pd.DataFrame(), None

    blob = get_gcs_client().bucket(GCS_BUCKET_NAME).blob(latest_name)
    data = blob.download_as_bytes()
    return pd.read_csv(BytesIO(data)), latest_name


def download_model(blob_name: str, local_path: str = "tmp_model.joblib") -> str:
    bucket = get_gcs_client().bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path


# -----------------------------
# API HELPERS
# -----------------------------
def predict_api(text: str) -> dict[str, Any]:
    payload = {"text": text}
    r = requests.post(f"{API_URL.rstrip('/')}/predictions", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🌐 API")
    st.code(API_URL)

    st.markdown("### ☁️ Bucket")
    st.code(GCS_BUCKET_NAME)

    st.markdown("### 🤖 Modèle")
    st.code(DEFAULT_MODEL_VERSION)

    st.markdown("---")

    if st.button("🔄 Refresh global"):
        st.cache_data.clear()
        st.rerun()


# -----------------------------
# DATASET CHARGÉ UNE FOIS
# -----------------------------
df, source = load_latest_dataset()

if not df.empty:
    if "creationdate" in df.columns:
        df["creationdate"] = pd.to_datetime(df["creationdate"], errors="coerce")
    if "dep_name" in df.columns or "dep_code" in df.columns:
        df["department_label"] = df.apply(_department_label, axis=1)
    else:
        df["department_label"] = "Inconnu"


# -----------------------------
# TABS
# -----------------------------
tab_signalconso, tab_predict, tab_data, tab_ml, tab_compare, tab_monitoring, tab_gcs, tab_pipeline = st.tabs(
    [
        "📋 Signal Conso",
        "🤖 Prédictions",
        "📊 Data",
        "🧠 ML",
        "⚖️ Comparaison",
        "📉 Monitoring",
        "☁️ GCS",
        "⚙️ Pipeline",
    ]
)


# -----------------------------
# TAB 1 - SIGNAL CONSO DASHBOARD
# -----------------------------
with tab_signalconso:
    st.subheader("Tableau de bord Signal Conso")

    if df.empty:
        st.warning("Aucun dataset trouvé dans le dossier processed/ du bucket GCS.")
    else:
        available_dates = (
            df["creationdate"].dropna().dt.date
            if "creationdate" in df.columns
            else pd.Series(dtype=object)
        )
        reference_date = available_dates.max() if not available_dates.empty else date.today()

        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.4, 1.4])

        with c1:
            selected_date = st.date_input(
                "Sélection d'une autre date",
                value=reference_date,
                format="DD/MM/YYYY",
            )

        with c2:
            period_choice = st.selectbox(
                "Période",
                ["Depuis le début du mois", "7 derniers jours", "Toutes les données"],
                index=0,
            )

        # --------- REGION ---------
        with c3:
            region_options = ["Toutes les régions"]
            if "reg_name" in df.columns:
                region_options.extend(
                    sorted(
                        df["reg_name"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                )

            selected_region = st.selectbox(
                "Région",
                region_options,
            )

        # --------- DATA FILTRÉE PAR REGION (IMPORTANT) ---------
        df_region_filtered = df.copy()

        if selected_region != "Toutes les régions" and "reg_name" in df.columns:
            df_region_filtered = df_region_filtered[
                df_region_filtered["reg_name"].astype(str) == selected_region
            ]

        # --------- DEPARTEMENT (DÉPENDANT) ---------
        with c4:
            dept_options = ["Tous les départements"]

            if "department_label" in df_region_filtered.columns:
                dept_options.extend(
                    sorted(
                        df_region_filtered["department_label"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                )

            selected_department = st.selectbox(
                "Département",
                dept_options,
            )

        # Filtrage temporel
        filtered_df = df.copy()

        # ---- Temps
        if "creationdate" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["creationdate"].notna()].copy()

            if period_choice == "Depuis le début du mois":
                start_date = selected_date.replace(day=1)
                end_date = selected_date
                filtered_df = filtered_df[
                    (filtered_df["creationdate"].dt.date >= start_date)
                    & (filtered_df["creationdate"].dt.date <= end_date)
                ]

            elif period_choice == "7 derniers jours":
                start_date = selected_date - timedelta(days=6)
                end_date = selected_date
                filtered_df = filtered_df[
                    (filtered_df["creationdate"].dt.date >= start_date)
                    & (filtered_df["creationdate"].dt.date <= end_date)
                ]

        # ---- Région
        if selected_region != "Toutes les régions" and "reg_name" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["reg_name"].astype(str) == selected_region
            ]

        # ---- Département (après région)
        if selected_department != "Tous les départements" and "department_label" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["department_label"] == selected_department
            ]

        selected_label_date = selected_date.strftime("%d/%m/%Y")
        st.markdown(f"### Tableau de bord Signal Conso au {selected_label_date}")
        st.write(f"**Période :** {period_choice}")
        st.write(f"**Département sélectionné :** {selected_department}")

        if filtered_df.empty:
            st.info("Aucune donnée ne correspond aux filtres sélectionnés.")
        else:
            # --------- KPI ---------
            total_signalements = len(filtered_df)
            transmitted = (
                int(_bool_series(filtered_df["signalement_transmis"]).sum())
                if "signalement_transmis" in filtered_df.columns
                else 0
            )
            read = (
                int(_bool_series(filtered_df["signalement_lu"]).sum())
                if "signalement_lu" in filtered_df.columns
                else 0
            )
            response = (
                int(_bool_series(filtered_df["signalement_reponse"]).sum())
                if "signalement_reponse" in filtered_df.columns
                else 0
            )

            transmitted_rate = (transmitted / total_signalements) if total_signalements else 0.0
            read_rate = (read / transmitted) if transmitted else 0.0
            response_rate = (response / read) if read else 0.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Nombre de signalements", f"{total_signalements}")
            k2.metric("Part de signalements transmis", f"{transmitted_rate:.2%}")
            k3.metric("Part des signalements transmis lus", f"{read_rate:.2%}")
            k4.metric("Part des signalements lus ayant une réponse", f"{response_rate:.2%}")

            st.progress(min(max(transmitted_rate, 0.0), 1.0))
            st.caption(
                f"{transmitted} signalements transmis sur un total de {total_signalements} signalements"
            )
            st.progress(min(max(read_rate, 0.0), 1.0))
            st.caption(f"{read} signalements lus sur un total de {transmitted} signalements transmis")
            st.progress(min(max(response_rate, 0.0), 1.0))
            st.caption(f"{response} signalements ayant reçu une réponse sur un total de {read} signalements lus")

            st.divider()

            # -----------------------------
            # CARTE FRANCE PAR DÉPARTEMENT
            # -----------------------------
            if "dep_code" in filtered_df.columns:
                st.subheader("🗺️ Carte des signalements par département")

                try:
                    import json
                    import plotly.express as px
                    from urllib.request import urlopen

                    with urlopen("https://france-geojson.gregoiredavid.fr/repo/departements.geojson") as response:
                        geojson = json.load(response)

                    dep_counts = (
                        filtered_df["dep_code"]
                        .astype(str)
                        .str.strip()
                        .str.zfill(2)
                        .value_counts()
                        .reset_index()
                    )
                    dep_counts.columns = ["code", "count"]

                    fig = px.choropleth(
                        dep_counts,
                        geojson=geojson,
                        locations="code",
                        featureidkey="properties.code",
                        color="count",
                        color_continuous_scale="Reds",
                        labels={"count": "Nombre de signalements"},
                    )

                    fig.update_geos(fitbounds="locations", visible=False)
                    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur affichage carte : {e}")

                st.divider()
            else:
                st.info("La colonne 'dep_code' est absente du dataset.")

            # --------- Répartition par catégories ---------
            st.markdown("### Répartition par catégories")
            if "category" in filtered_df.columns:
                cat_counts = _frequency_from_cells(filtered_df, "category", limit=15)
                if not cat_counts.empty:
                    st.bar_chart(cat_counts)
                else:
                    st.info("Aucune catégorie exploitable.")
            else:
                st.info("La colonne 'category' est absente du dataset.")

            st.divider()

            # --------- Mots-clés les plus populaires ---------
            st.markdown("### Mots-clés les plus populaires")
            keyword_counts = _keyword_frequency(filtered_df, limit=15)
            if not keyword_counts.empty:
                st.bar_chart(keyword_counts)
            else:
                st.info("Aucun mot-clé exploitable.")

            st.divider()

            # --------- Aperçu ---------
            st.markdown("### Aperçu des données filtrées")
            preview_cols = [
                c for c in ["creationdate", "department_label", "reg_name", "category", "tags", "clean_text"]
                if c in filtered_df.columns
            ]
            st.dataframe(filtered_df[preview_cols].head(20), use_container_width=True)


# -----------------------------
# TAB 2 - PRÉDICTION
# -----------------------------
with tab_predict:
    st.markdown('<div class="section-title">🤖 Prédiction intelligente</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])

    with col1:
        text = st.text_area("✍️ Saisir un texte client", height=160)

    with col2:
        st.markdown("### ⚡ Action")
        run = st.button("🚀 Prédire")

    if run:
        if not text.strip():
            st.warning("Texte vide")
        else:
            with st.spinner("Analyse IA..."):
                result = predict_api(text)

            st.success(f"📌 {result.get('predicted_category')}")
            st.metric("Confiance", f"{result.get('confidence',0):.2%}")
            st.json(result)


# -----------------------------
# TAB 3 - DATA
# -----------------------------
with tab_data:
    st.markdown('<div class="section-title">📊 Data Explorer</div>', unsafe_allow_html=True)

    if not df.empty:
        c1,c2,c3 = st.columns(3)
        c1.metric("📄 Lignes", len(df))
        c2.metric("📊 Colonnes", len(df.columns))
        c3.metric("🏷️ Catégories", df['category'].nunique() if 'category' in df else 0)

        st.markdown("---")

        colA, colB = st.columns(2)

        with colA:
            if "category" in df.columns:
                st.markdown("### Top catégories")
                st.bar_chart(df["category"].value_counts().head(10))

        with colB:
            if "creationdate" in df.columns:
                st.markdown("### Volume temporel")
                timeline = df["creationdate"].dropna().dt.date.value_counts().sort_index()
                st.line_chart(timeline)

        if "clean_text" in df.columns:
            st.markdown("### ☁️ Nuage de mots")
            text_all = " ".join(df["clean_text"].dropna().astype(str))
            if text_all:
                from wordcloud import WordCloud
                wc = WordCloud(width=800,height=400).generate(text_all)
                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Aucun texte exploitable pour le wordcloud.")
    else:
        st.warning("Aucun dataset trouvé dans le dossier processed/ du bucket GCS.")


# -----------------------------
# TAB 4 - ML
# -----------------------------
with tab_ml:
    st.markdown('<div class="section-title">🧠 Analyse modèle</div>', unsafe_allow_html=True)

    if not df.empty and "category" in df.columns:
        if st.button("📊 Évaluer"):
            with st.spinner("Calcul métriques..."):
                eval_df = df.dropna(subset=["clean_text","category"])
                model = TicketModel(DEFAULT_MODEL_PATH)
                model.load()

                eval_df["pred"] = eval_df["clean_text"].apply(model.predict)

                acc = accuracy_score(eval_df["category"], eval_df["pred"])
                f1 = f1_score(eval_df["category"], eval_df["pred"], average="weighted")

                c1,c2 = st.columns(2)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("F1", f"{f1:.3f}")

                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(eval_df["category"], eval_df["pred"])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                st.pyplot(fig)
    else:
        st.info("Le dataset n'est pas disponible ou ne contient pas les colonnes attendues.")


# -----------------------------
# TAB 5 - COMPARAISON
# -----------------------------
with tab_compare:
    st.markdown('<div class="section-title">⚖️ Benchmark modèles</div>', unsafe_allow_html=True)

    models = list_blob_names("models/")
    selected = st.multiselect("Choisir modèles", models)

    if selected and not df.empty:
        results = []

        for m in selected:
            try:
                path = download_model(m)
                model = TicketModel(path)
                model.load()

                eval_df = df.dropna(subset=["clean_text","category"])
                eval_df["pred"] = eval_df["clean_text"].apply(model.predict)

                acc = accuracy_score(eval_df["category"], eval_df["pred"])
                f1 = f1_score(eval_df["category"], eval_df["pred"], average="weighted")

                results.append({"model": m, "accuracy": acc, "f1": f1})
            except Exception as e:
                results.append({"model": m, "error": str(e)})

        res = pd.DataFrame(results)
        st.dataframe(res)

        if "accuracy" in res.columns:
            st.bar_chart(res.set_index("model")[["accuracy","f1"]])

        if not res.empty and "f1" in res.columns:
            best = res.sort_values("f1", ascending=False).iloc[0]
            st.success(f"🏆 Best model: {best['model']}")
    else:
        st.info("Sélectionnez un ou plusieurs modèles pour lancer la comparaison.")


# -----------------------------
# TAB 6 - MONITORING
# -----------------------------
with tab_monitoring:
    st.subheader("Monitoring et dérive")

    if not df.empty and "category" in df.columns:
        current_distribution = _frequency_from_cells(df, "category", limit=20)
        if not current_distribution.empty:
            st.write("### Distribution actuelle des catégories")
            st.bar_chart(current_distribution)

            st.write("### Part des catégories")
            current_share = (current_distribution / current_distribution.sum()).sort_values(ascending=False)
            st.bar_chart(current_share)
        else:
            st.info("Aucune distribution exploitable.")

        if "predicted_category" in df.columns:
            st.write("### Distribution des prédictions enregistrées")
            st.bar_chart(df["predicted_category"].value_counts().head(15))
    else:
        st.info("Colonne 'category' absente ou dataset vide.")


# -----------------------------
# TAB 7 - GCS
# -----------------------------
with tab_gcs:
    st.subheader("Explorateur GCS")

    prefix = st.selectbox("Dossier", ["raw/", "processed/", "predictions/", "models/"])
    blob_names = list_blob_names(prefix)

    if blob_names:
        st.write(blob_names[:20])
    else:
        st.info("Aucun fichier trouvé.")


# -----------------------------
# TAB 8 - PIPELINE
# -----------------------------
with tab_pipeline:
    st.subheader("Pipeline ingestion + transformation + upload")
    st.write("Exécute le pipeline complet : extract → transform → upload vers GCS.")

    if st.button("🚀 Lancer le pipeline"):
        try:
            result = run_pipeline(st.write)
            st.success("Pipeline exécuté avec succès")
            st.json(result)
        except Exception as e:
            st.error(str(e))


# -----------------------------
# FOOTER
# -----------------------------
st.caption("Dashboard PRO — Data + ML + Monitoring 🚀")
