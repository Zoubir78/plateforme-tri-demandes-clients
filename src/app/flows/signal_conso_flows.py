"""
Orchestration Prefect des KPIs Signal Conso depuis Google Cloud Storage.

Flows orchestrés :
  1. nombre_signalements       — Comptage total des signalements
  2. signalements_transmis     — Part des signalements transmis aux entreprises
  3. signalements_transmis_lus — Part des signalements transmis qui ont été lus
  4. signalements_lus_reponse  — Part des signalements lus ayant reçu une réponse

Usage :
  # Lancement ponctuel
  python signal_conso_flows.py

  # Déploiement avec schedule (toutes les 5 minutes)
  prefect deployment build signal_conso_flows.py:kpi_pipeline_flow \
      --name signal-conso-realtime --interval 300
  prefect deployment apply kpi_pipeline_flow-deployment.yaml
  prefect agent start --pool default-agent-pool
"""

from __future__ import annotations

import ast
import os
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from typing import Any

import pandas as pd
from google.cloud import storage

# ── Prefect ──────────────────────────────────────────────────────────────────
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.states import Completed, Failed

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "clean_complaints")
GCS_PROCESSED_PREFIX: str = os.getenv("GCS_PROCESSED_PREFIX", "processed/")
BOOL_TRUE_VALUES: frozenset[str] = frozenset(
    {"1", "true", "t", "yes", "y", "oui", "vrai", "on"}
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _to_bool(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in BOOL_TRUE_VALUES


def _bool_series(series: pd.Series) -> pd.Series:
    return series.apply(_to_bool)


# ═══════════════════════════════════════════════════════════════════════════════
# TASKS GCS
# ═══════════════════════════════════════════════════════════════════════════════

@task(
    name="get-gcs-client",
    description="Initialise le client Google Cloud Storage.",
    retries=2,
    retry_delay_seconds=5,
    tags=["gcs", "infra"],
)
def get_gcs_client_task() -> storage.Client:
    logger = get_run_logger()
    logger.info("Initialisation du client GCS.")
    return storage.Client()


@task(
    name="find-latest-blob",
    description="Trouve le blob le plus récent dans le prefix GCS.",
    retries=3,
    retry_delay_seconds=10,
    tags=["gcs", "extract"],
)
def find_latest_blob_task(client: storage.Client, bucket_name: str, prefix: str) -> str | None:
    logger = get_run_logger()
    logger.info(f"Recherche du blob le plus récent dans gs://{bucket_name}/{prefix}")

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logger.warning(f"Aucun blob trouvé dans gs://{bucket_name}/{prefix}")
        return None

    fallback_dt = datetime.min.replace(tzinfo=timezone.utc)
    latest = max(blobs, key=lambda b: b.updated or fallback_dt)

    logger.info(f"Blob le plus récent : {latest.name} (mis à jour le {latest.updated})")
    return latest.name


@task(
    name="download-dataset",
    description="Télécharge le dataset CSV depuis GCS et le charge en DataFrame.",
    retries=3,
    retry_delay_seconds=15,
    tags=["gcs", "extract"],
)
def download_dataset_task(client: storage.Client, bucket_name: str, blob_name: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Téléchargement de gs://{bucket_name}/{blob_name}")

    blob = client.bucket(bucket_name).blob(blob_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data))

    logger.info(f"Dataset chargé : {len(df)} lignes × {len(df.columns)} colonnes")
    return df


@task(
    name="preprocess-dataframe",
    description="Normalise les types (dates, booléens) du DataFrame brut.",
    tags=["transform"],
)
def preprocess_task(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    if "creationdate" in df.columns:
        df["creationdate"] = pd.to_datetime(df["creationdate"], errors="coerce")
        logger.info("Colonne 'creationdate' convertie en datetime.")

    for bool_col in ["signalement_transmis", "signalement_lu", "signalement_reponse"]:
        if bool_col in df.columns:
            df[bool_col] = _bool_series(df[bool_col])

    logger.info("Pré-traitement terminé.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TASKS FILTRAGE
# ═══════════════════════════════════════════════════════════════════════════════

@task(
    name="apply-temporal-filter",
    description="Filtre le DataFrame sur une période temporelle.",
    tags=["filter"],
)
def apply_temporal_filter_task(
    df: pd.DataFrame,
    reference_date: date | None = None,
    period: str = "Depuis le début du mois",
) -> pd.DataFrame:
    logger = get_run_logger()

    if "creationdate" not in df.columns:
        logger.warning("Colonne 'creationdate' absente — pas de filtre temporel.")
        return df

    ref = reference_date or date.today()
    df = df[df["creationdate"].notna()].copy()

    if period == "Depuis le début du mois":
        start = ref.replace(day=1)
        end = ref
    elif period == "7 derniers jours":
        start = ref - timedelta(days=6)
        end = ref
    else:  # Toutes les données
        logger.info("Période = 'Toutes les données' — pas de filtre temporel.")
        return df

    filtered = df[
        (df["creationdate"].dt.date >= start)
        & (df["creationdate"].dt.date <= end)
    ]

    logger.info(
        f"Filtre temporel [{start} → {end}] : {len(df)} → {len(filtered)} lignes."
    )
    return filtered


@task(
    name="apply-geo-filter",
    description="Filtre le DataFrame par région et/ou département.",
    tags=["filter"],
)
def apply_geo_filter_task(
    df: pd.DataFrame,
    region: str | None = None,
    department_label: str | None = None,
) -> pd.DataFrame:
    logger = get_run_logger()

    if region and "reg_name" in df.columns:
        before = len(df)
        df = df[df["reg_name"].astype(str) == region]
        logger.info(f"Filtre région '{region}' : {before} → {len(df)} lignes.")

    if department_label and "department_label" in df.columns:
        before = len(df)
        df = df[df["department_label"] == department_label]
        logger.info(f"Filtre département '{department_label}' : {before} → {len(df)} lignes.")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TASKS KPI (1 task = 1 KPI)
# ═══════════════════════════════════════════════════════════════════════════════

@task(
    name="kpi-nombre-signalements",
    description="Calcule le nombre total de signalements.",
    tags=["kpi"],
)
def kpi_nombre_signalements_task(df: pd.DataFrame) -> dict[str, Any]:
    logger = get_run_logger()
    total = len(df)
    logger.info(f"[KPI] Nombre de signalements = {total}")
    return {
        "kpi": "nombre_signalements",
        "label": "Nombre de signalements",
        "value": total,
        "unit": "signalements",
        "computed_at": datetime.utcnow().isoformat(),
    }


@task(
    name="kpi-signalements-transmis",
    description="Calcule la part des signalements transmis aux entreprises.",
    tags=["kpi"],
)
def kpi_signalements_transmis_task(df: pd.DataFrame) -> dict[str, Any]:
    logger = get_run_logger()

    total = len(df)
    if "signalement_transmis" not in df.columns:
        logger.warning("Colonne 'signalement_transmis' absente.")
        return {
            "kpi": "signalements_transmis",
            "label": "Part de signalements transmis",
            "value": None,
            "numerator": None,
            "denominator": total,
            "error": "Colonne manquante",
            "computed_at": datetime.utcnow().isoformat(),
        }

    transmitted = int(df["signalement_transmis"].sum())
    rate = transmitted / total if total else 0.0

    logger.info(
        f"[KPI] Signalements transmis = {transmitted}/{total} = {rate:.2%}"
    )
    return {
        "kpi": "signalements_transmis",
        "label": "Part de signalements transmis",
        "value": round(rate, 6),
        "value_pct": f"{rate:.2%}",
        "numerator": transmitted,
        "denominator": total,
        "computed_at": datetime.utcnow().isoformat(),
    }


@task(
    name="kpi-signalements-transmis-lus",
    description="Calcule la part des signalements transmis qui ont été lus.",
    tags=["kpi"],
)
def kpi_signalements_transmis_lus_task(df: pd.DataFrame) -> dict[str, Any]:
    logger = get_run_logger()

    missing_cols = [c for c in ["signalement_transmis", "signalement_lu"] if c not in df.columns]
    if missing_cols:
        logger.warning(f"Colonnes manquantes : {missing_cols}")
        return {
            "kpi": "signalements_transmis_lus",
            "label": "Part des signalements transmis lus",
            "value": None,
            "error": f"Colonnes manquantes : {missing_cols}",
            "computed_at": datetime.utcnow().isoformat(),
        }

    transmitted = int(df["signalement_transmis"].sum())
    transmitted_df = df[df["signalement_transmis"]]
    read = int(transmitted_df["signalement_lu"].sum())
    rate = read / transmitted if transmitted else 0.0

    logger.info(
        f"[KPI] Signalements transmis lus = {read}/{transmitted} = {rate:.2%}"
    )
    return {
        "kpi": "signalements_transmis_lus",
        "label": "Part des signalements transmis lus",
        "value": round(rate, 6),
        "value_pct": f"{rate:.2%}",
        "numerator": read,
        "denominator": transmitted,
        "computed_at": datetime.utcnow().isoformat(),
    }


@task(
    name="kpi-signalements-lus-reponse",
    description="Calcule la part des signalements lus ayant reçu une réponse.",
    tags=["kpi"],
)
def kpi_signalements_lus_reponse_task(df: pd.DataFrame) -> dict[str, Any]:
    logger = get_run_logger()

    missing_cols = [
        c for c in ["signalement_lu", "signalement_reponse"] if c not in df.columns
    ]
    if missing_cols:
        logger.warning(f"Colonnes manquantes : {missing_cols}")
        return {
            "kpi": "signalements_lus_reponse",
            "label": "Part des signalements lus ayant une réponse",
            "value": None,
            "error": f"Colonnes manquantes : {missing_cols}",
            "computed_at": datetime.utcnow().isoformat(),
        }

    read = int(df["signalement_lu"].sum())
    read_df = df[df["signalement_lu"]]
    response = int(read_df["signalement_reponse"].sum())
    rate = response / read if read else 0.0

    logger.info(
        f"[KPI] Signalements lus avec réponse = {response}/{read} = {rate:.2%}"
    )
    return {
        "kpi": "signalements_lus_reponse",
        "label": "Part des signalements lus ayant une réponse",
        "value": round(rate, 6),
        "value_pct": f"{rate:.2%}",
        "numerator": response,
        "denominator": read,
        "computed_at": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK PUBLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@task(
    name="publish-kpi-results",
    description="Publie les résultats KPI sous forme d'artifact Prefect et JSON.",
    tags=["publish"],
)
def publish_kpi_results_task(kpis: list[dict[str, Any]], source_blob: str) -> dict[str, Any]:
    logger = get_run_logger()

    summary = {
        "source": source_blob,
        "computed_at": datetime.utcnow().isoformat(),
        "kpis": kpis,
    }

    # Artifact tabulaire visible dans l'UI Prefect
    table_rows = []
    for k in kpis:
        row: dict[str, Any] = {
            "KPI": k.get("label", k.get("kpi", "?")),
        }
        if "value_pct" in k:
            row["Valeur"] = k["value_pct"]
        elif "value" in k and k["value"] is not None:
            row["Valeur"] = str(k["value"])
        else:
            row["Valeur"] = k.get("error", "N/A")

        if "numerator" in k and "denominator" in k:
            row["Détail"] = f"{k['numerator']} / {k['denominator']}"
        else:
            row["Détail"] = "—"

        table_rows.append(row)

    create_table_artifact(
        key="signal-conso-kpis",
        table=table_rows,
        description=f"KPIs Signal Conso — source : `{source_blob}`",
    )

    logger.info(f"KPIs publiés : {[k['kpi'] for k in kpis]}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# SOUS-FLOWS SPÉCIALISÉS (1 par KPI)
# ═══════════════════════════════════════════════════════════════════════════════

@flow(
    name="flow-nombre-signalements",
    description="Flow dédié au KPI : Nombre total de signalements.",
    log_prints=True,
)
def flow_nombre_signalements(
    df: pd.DataFrame,
) -> dict[str, Any]:
    return kpi_nombre_signalements_task(df)


@flow(
    name="flow-signalements-transmis",
    description="Flow dédié au KPI : Part des signalements transmis.",
    log_prints=True,
)
def flow_signalements_transmis(
    df: pd.DataFrame,
) -> dict[str, Any]:
    return kpi_signalements_transmis_task(df)


@flow(
    name="flow-signalements-transmis-lus",
    description="Flow dédié au KPI : Part des signalements transmis lus.",
    log_prints=True,
)
def flow_signalements_transmis_lus(
    df: pd.DataFrame,
) -> dict[str, Any]:
    return kpi_signalements_transmis_lus_task(df)


@flow(
    name="flow-signalements-lus-reponse",
    description="Flow dédié au KPI : Part des signalements lus ayant une réponse.",
    log_prints=True,
)
def flow_signalements_lus_reponse(
    df: pd.DataFrame,
) -> dict[str, Any]:
    return kpi_signalements_lus_reponse_task(df)


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW PRINCIPAL (PIPELINE COMPLET)
# ═══════════════════════════════════════════════════════════════════════════════

@flow(
    name="kpi-pipeline-flow",
    description=(
        "Pipeline Prefect complet Signal Conso : "
        "extraction GCS → filtrage → calcul des 4 KPIs en parallèle → publication."
    ),
    log_prints=True,
)
def kpi_pipeline_flow(
    bucket_name: str = GCS_BUCKET_NAME,
    prefix: str = GCS_PROCESSED_PREFIX,
    reference_date: date | None = None,
    period: str = "Depuis le début du mois",
    region: str | None = None,
    department_label: str | None = None,
) -> dict[str, Any]:
    """
    Paramètres
    ----------
    bucket_name      : nom du bucket GCS
    prefix           : dossier dans le bucket (ex: "processed/")
    reference_date   : date de référence (défaut : aujourd'hui)
    period           : "Depuis le début du mois" | "7 derniers jours" | "Toutes les données"
    region           : filtre optionnel sur reg_name
    department_label : filtre optionnel sur department_label (ex: "75 - Paris")
    """
    logger = get_run_logger()
    logger.info(
        f"🚀 Démarrage pipeline KPI Signal Conso | bucket={bucket_name} | période={period}"
    )

    # ── 1. Extraction GCS ────────────────────────────────────────────────────
    client = get_gcs_client_task()
    blob_name = find_latest_blob_task(client, bucket_name, prefix)

    if blob_name is None:
        logger.error("Aucun fichier trouvé dans GCS — arrêt du pipeline.")
        return {"error": "Aucun fichier GCS trouvé", "kpis": []}

    raw_df = download_dataset_task(client, bucket_name, blob_name)

    # ── 2. Pré-traitement ─────────────────────────────────────────────────────
    df = preprocess_task(raw_df)

    # ── 3. Filtrage ───────────────────────────────────────────────────────────
    df = apply_temporal_filter_task(df, reference_date=reference_date, period=period)
    df = apply_geo_filter_task(df, region=region, department_label=department_label)

    if df.empty:
        logger.warning("DataFrame vide après filtrage — KPIs non calculables.")
        return {"error": "Aucune donnée après filtrage", "kpis": [], "source": blob_name}

    # ── 4. Calcul des 4 KPIs (sous-flows en parallèle via submit) ────────────
    # Note : les sous-flows sont appelés directement ; pour un vrai parallélisme
    # Prefect, utiliser .submit() via ConcurrentTaskRunner ou DaskTaskRunner.

    kpi_total = flow_nombre_signalements(df)
    kpi_transmis = flow_signalements_transmis(df)
    kpi_transmis_lus = flow_signalements_transmis_lus(df)
    kpi_lus_reponse = flow_signalements_lus_reponse(df)

    kpis = [kpi_total, kpi_transmis, kpi_transmis_lus, kpi_lus_reponse]

    # ── 5. Publication ────────────────────────────────────────────────────────
    summary = publish_kpi_results_task(kpis, source_blob=blob_name)

    logger.info("✅ Pipeline terminé avec succès.")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT LOCAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = kpi_pipeline_flow(
        period="Depuis le début du mois",
        # Décommenter pour filtrer :
        # region="Île-de-France",
        # department_label="75 - Paris",
    )
    import json
    print("\n──────────────────────────────────────────")
    print("📊 Résultats KPI Signal Conso")
    print("──────────────────────────────────────────")
    print(json.dumps(result, indent=2, ensure_ascii=False))