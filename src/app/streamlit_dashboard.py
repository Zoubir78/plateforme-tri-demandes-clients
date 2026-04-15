from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
from io import BytesIO
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from wordcloud import WordCloud

from src.app.ml.predict import TicketModel
from src.app.pipeline.pipeline_service import run_pipeline

# -----------------------------
# CONFIG
# -----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "clean_complaints")

st.set_page_config(page_title="Dashboard IA", layout="wide")
st.title("📊 Plateforme IA - Tri des demandes clients")

# -----------------------------
# GCS HELPERS
# -----------------------------
@st.cache_resource
def get_gcs_client():
    return storage.Client()


@st.cache_data(ttl=60)
def list_blob_names(prefix: str) -> list[str]:
    """
    Retourne uniquement des noms de fichiers (types sérialisables),
    compatibles avec st.cache_data.
    """
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]


def get_latest_blob_name(prefix: str) -> str | None:
    """
    Ne renvoie pas d'objet Blob en cache.
    On récupère les blobs ici sans cache, puis on retourne seulement le nom.
    """
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        return None

    fallback_dt = datetime.min.replace(tzinfo=timezone.utc)
    latest = max(blobs, key=lambda b: b.updated or fallback_dt)
    return latest.name


def load_latest_dataset():
    latest_name = get_latest_blob_name("processed/")

    if not latest_name:
        return pd.DataFrame(), None

    blob = get_gcs_client().bucket(GCS_BUCKET_NAME).blob(latest_name)
    data = blob.download_as_bytes()
    return pd.read_csv(BytesIO(data)), latest_name


def download_model(blob_name: str, local_path="tmp_model.joblib"):
    blob = get_gcs_client().bucket(GCS_BUCKET_NAME).blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path


# -----------------------------
# API
# -----------------------------
def predict_api(text: str):
    r = requests.post(f"{API_URL}/predictions", json={"text": text})
    r.raise_for_status()
    return r.json()


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Config")
    st.write(f"API: {API_URL}")
    st.write(f"Bucket: {GCS_BUCKET_NAME}")

    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()


# -----------------------------
# LOAD DATA ONCE
# -----------------------------
df, source = load_latest_dataset()

# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs([
    "🤖 Prédire",
    "📊 Data",
    "🧠 ML",
    "⚖️ Comparaison",
    "📉 Monitoring",
    "☁️ GCS",
    "⚙️ Pipeline"
])

# -----------------------------
# TAB 1 - PREDICT
# -----------------------------
with tabs[0]:
    st.subheader("Prédiction temps réel")

    text = st.text_area("Texte client")

    if st.button("Prédire"):
        try:
            result = predict_api(text)
            st.success(result["predicted_category"])
            st.json(result)
        except Exception as e:
            st.error(e)

# -----------------------------
# TAB 2 - DATA
# -----------------------------
with tabs[1]:
    st.subheader("Analyse des données")

    if source is not None and not df.empty:
        st.success(f"Dataset: {source}")
        st.metric("Nb lignes", len(df))

        if "category" in df.columns:
            st.bar_chart(df["category"].value_counts().head(10))

        if "token_count" in df.columns:
            st.line_chart(df["token_count"].value_counts().sort_index())

        if "creationdate" in df.columns:
            df["creationdate"] = pd.to_datetime(df["creationdate"], errors="coerce")
            timeline = df.groupby(df["creationdate"].dt.date).size()
            st.line_chart(timeline)

        if "clean_text" in df.columns:
            st.subheader("Wordcloud")
            text_all = " ".join(df["clean_text"].dropna().astype(str))
            if text_all.strip():
                wc = WordCloud(width=800, height=400).generate(text_all)

                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Aucun texte disponible pour générer le wordcloud.")
    else:
        st.warning("Aucun dataset trouvé dans le dossier processed/.")

# -----------------------------
# TAB 3 - ML
# -----------------------------
with tabs[2]:
    st.subheader("Analyse modèle")

    if not df.empty and "clean_text" in df.columns and "category" in df.columns:
        model = TicketModel("models/model.joblib")
        model.load()

        if st.button("📊 Évaluer modèle"):
            df["pred"] = df["clean_text"].astype(str).apply(lambda x: model.predict(x))

            acc = accuracy_score(df["category"], df["pred"])
            f1 = f1_score(df["category"], df["pred"], average="weighted")

            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("F1", f"{f1:.3f}")

            cm = confusion_matrix(df["category"], df["pred"])

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

            errors = df[df["category"] != df["pred"]]
            st.dataframe(errors.head(20))
    else:
        st.info("Le dataset n'est pas disponible ou ne contient pas les colonnes attendues.")

# -----------------------------
# TAB 4 - COMPARAISON
# -----------------------------
with tabs[3]:
    st.subheader("Comparaison modèles")

    model_names = list_blob_names("models/")
    selected = st.multiselect("Choisir modèles", model_names)

    results = []

    if not df.empty and "clean_text" in df.columns and "category" in df.columns:
        for m in selected:
            path = download_model(m)
            model = TicketModel(path)
            model.load()

            df["pred"] = df["clean_text"].astype(str).apply(lambda x: model.predict(x))
            acc = accuracy_score(df["category"], df["pred"])

            results.append({"model": m, "accuracy": acc})

        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df)
            st.bar_chart(res_df.set_index("model"))
    else:
        st.info("Impossible de comparer les modèles sans dataset valide.")

# -----------------------------
# TAB 5 - MONITORING
# -----------------------------
with tabs[4]:
    st.subheader("Monitoring dérive")

    if not df.empty and "category" in df.columns:
        current = df["category"].value_counts(normalize=True)

        st.write("Distribution actuelle")
        st.bar_chart(current)
    else:
        st.info("Colonne 'category' absente ou dataset vide.")

# -----------------------------
# TAB 6 - GCS
# -----------------------------
with tabs[5]:
    st.subheader("Explorateur GCS")

    prefix = st.selectbox("Dossier", ["raw/", "processed/", "models/"])
    blob_names = list_blob_names(prefix)

    st.write(blob_names[:20] if blob_names else "Aucun fichier trouvé.")

# -----------------------------
# TAB 7 - PIPELINE
# -----------------------------
with tabs[6]:
    st.subheader("Pipeline")

    if st.button("🚀 Lancer pipeline"):
        try:
            result = run_pipeline(st.write)
            st.success("Pipeline OK")
            st.json(result)
        except Exception as e:
            st.error(e)