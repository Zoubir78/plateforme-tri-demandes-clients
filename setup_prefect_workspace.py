"""
setup_prefect_workspace.py
==========================
Création programmatique du workspace Prefect Cloud pour Signal Conso.
Alternative Python au script shell — utile sur Windows ou CI/CD.

Usage :
    # Avec variables d'environnement
    export PREFECT_API_KEY=pnu_xxxx
    export GCS_BUCKET_NAME=clean_complaints
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
    python setup_prefect_workspace.py

    # Ou tout en une ligne
    PREFECT_API_KEY=pnu_xxxx python setup_prefect_workspace.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
WORKSPACE_NAME   = os.getenv("PREFECT_WORKSPACE_NAME", "signal-conso")
ACCOUNT_HANDLE   = os.getenv("PREFECT_ACCOUNT_HANDLE", "")
WORK_POOL_NAME   = os.getenv("PREFECT_WORK_POOL", "signal-conso-pool")
GCS_BUCKET       = os.getenv("GCS_BUCKET_NAME", "clean_complaints")
GOOGLE_CREDS     = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
PREFECT_API_KEY  = os.getenv("PREFECT_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"{GREEN}[OK]{RESET}    {msg}")
def warn(msg: str) -> None: print(f"{YELLOW}[WARN]{RESET}  {msg}")
def err(msg: str)  -> None: print(f"{RED}[ERROR]{RESET} {msg}")
def info(msg: str) -> None: print(f"{CYAN}[INFO]{RESET}  {msg}")
def section(title: str) -> None:
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Exécute une commande shell et retourne le résultat."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Installation / vérification des packages
# ══════════════════════════════════════════════════════════════════════════════
def step_install() -> None:
    section("1. Vérification des dépendances")

    packages = {
        "prefect":              "prefect>=2.14.0",
        "prefect_gcp":         "prefect-gcp>=0.5.0",
        "google.cloud.storage": "google-cloud-storage>=2.10.0",
        "pandas":               "pandas>=2.0.0",
    }

    for module, pip_name in packages.items():
        try:
            __import__(module.replace("-", "_"))
            ok(f"{module} déjà installé.")
        except ImportError:
            info(f"Installation de {pip_name}…")
            run([sys.executable, "-m", "pip", "install", "--quiet", pip_name])
            ok(f"{pip_name} installé.")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Authentification Prefect Cloud
# ══════════════════════════════════════════════════════════════════════════════
def step_auth() -> None:
    section("2. Authentification Prefect Cloud")

    api_key = PREFECT_API_KEY
    if not api_key:
        warn("PREFECT_API_KEY non définie.")
        info("Récupérez votre clé sur : https://app.prefect.cloud/my/api-keys")
        api_key = input("  → Clé API : ").strip()
        os.environ["PREFECT_API_KEY"] = api_key

    result = run(["prefect", "cloud", "login", "--key", api_key, "--no-interactive"], check=False)
    if result.returncode != 0:
        run(["prefect", "cloud", "login", "--key", api_key])

    ok("Authentification Prefect Cloud réussie.")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Sélection / création du workspace
# ══════════════════════════════════════════════════════════════════════════════
def step_workspace() -> None:
    section("3. Workspace Prefect Cloud")

    # Lister les workspaces disponibles
    result = run(["prefect", "cloud", "workspace", "ls"], check=False)
    info("Workspaces disponibles :")
    print(result.stdout)

    if WORKSPACE_NAME in result.stdout:
        warn(f"Workspace '{WORKSPACE_NAME}' déjà existant.")
    else:
        info(f"Création du workspace '{WORKSPACE_NAME}'…")
        create_result = run(
            ["prefect", "cloud", "workspace", "create", "--name", WORKSPACE_NAME],
            check=False,
        )
        if create_result.returncode == 0:
            ok(f"Workspace '{WORKSPACE_NAME}' créé.")
        else:
            warn("Création via CLI non supportée — faites-le sur https://app.prefect.cloud")

    # Sélectionner le workspace
    if ACCOUNT_HANDLE:
        ws_slug = f"{ACCOUNT_HANDLE}/{WORKSPACE_NAME}"
        set_result = run(
            ["prefect", "cloud", "workspace", "set", "--workspace", ws_slug],
            check=False,
        )
        if set_result.returncode == 0:
            ok(f"Workspace '{ws_slug}' sélectionné.")
        else:
            warn("Sélection automatique échouée — sélection interactive :")
            run(["prefect", "cloud", "workspace", "set"])
    else:
        warn("PREFECT_ACCOUNT_HANDLE non défini — sélection interactive :")
        run(["prefect", "cloud", "workspace", "set"])


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Work Pool
# ══════════════════════════════════════════════════════════════════════════════
def step_work_pool() -> None:
    section("4. Work Pool")

    result = run(["prefect", "work-pool", "ls"], check=False)

    if WORK_POOL_NAME in result.stdout:
        warn(f"Work pool '{WORK_POOL_NAME}' déjà existant — ignoré.")
    else:
        run(["prefect", "work-pool", "create", WORK_POOL_NAME, "--type", "process"])
        ok(f"Work pool '{WORK_POOL_NAME}' créé.")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Variables Prefect
# ══════════════════════════════════════════════════════════════════════════════
def step_variables() -> None:
    section("5. Variables Prefect")

    variables: dict[str, str] = {
        "gcs_bucket_name":      GCS_BUCKET,
        "gcs_processed_prefix": "processed/",
        "work_pool":            WORK_POOL_NAME,
        "default_period":       "Depuis le début du mois",
    }

    for name, value in variables.items():
        result = run(
            ["prefect", "variable", "set", name, "--value", value, "--overwrite"],
            check=False,
        )
        if result.returncode == 0:
            ok(f"Variable '{name}' = '{value}'")
        else:
            result2 = run(
                ["prefect", "variable", "set", name, value],
                check=False,
            )
            if result2.returncode == 0:
                ok(f"Variable '{name}' = '{value}'")
            else:
                warn(f"Variable '{name}' non créée (vérifiez la version Prefect).")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Blocks GCS (GcpCredentials + GCSBucket)
# ══════════════════════════════════════════════════════════════════════════════
def step_gcs_blocks() -> None:
    section("6. Blocks GCS")

    if not GOOGLE_CREDS:
        warn("GOOGLE_APPLICATION_CREDENTIALS non défini.")
        warn("Créez le block GCS manuellement dans l'UI Prefect :")
        warn("  https://app.prefect.cloud → Blocks → GCS Bucket")
        return

    creds_path = Path(GOOGLE_CREDS)
    if not creds_path.exists():
        err(f"Fichier credentials introuvable : {creds_path}")
        return

    try:
        from prefect_gcp import GcpCredentials, GCSBucket  # type: ignore

        with open(creds_path) as f:
            sa_info = json.load(f)

        # Block credentials
        gcp_creds = GcpCredentials(service_account_info=sa_info)
        gcp_creds.save("signal-conso-gcp-creds", overwrite=True)
        ok("Block GcpCredentials 'signal-conso-gcp-creds' créé.")

        # Block bucket
        gcs_block = GCSBucket(
            bucket=GCS_BUCKET,
            gcp_credentials=gcp_creds,
        )
        gcs_block.save("signal-conso-gcs-bucket", overwrite=True)
        ok("Block GCSBucket 'signal-conso-gcs-bucket' créé.")

    except ImportError:
        err("prefect-gcp non installé. Relancez l'étape 1.")
    except Exception as exc:
        err(f"Erreur lors de la création des blocks GCS : {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Déploiement des flows
# ══════════════════════════════════════════════════════════════════════════════
def step_deploy() -> None:
    section("7. Déploiement des flows")

    for fname in ("prefect.yaml", "signal_conso_flows.py"):
        if not Path(fname).exists():
            err(f"Fichier '{fname}' introuvable dans le répertoire courant.")
            err("Assurez-vous que prefect.yaml et signal_conso_flows.py sont présents.")
            sys.exit(1)

    run(["prefect", "deploy", "--all", "--prefect-file", "prefect.yaml"])
    ok("Tous les deployments Signal Conso ont été créés.")


# ══════════════════════════════════════════════════════════════════════════════
# RÉCAPITULATIF
# ══════════════════════════════════════════════════════════════════════════════
def step_summary() -> None:
    section("Récapitulatif")

    print(f"  {BOLD}Workspace{RESET}   : {WORKSPACE_NAME}")
    print(f"  {BOLD}Work Pool{RESET}   : {WORK_POOL_NAME}")
    print(f"  {BOLD}Bucket GCS{RESET}  : {GCS_BUCKET}")
    print()
    ok("Workspace Prefect configuré avec succès !")
    print()
    print(f"  {BOLD}Démarrer le worker :{RESET}")
    print(f"    {CYAN}prefect worker start --pool {WORK_POOL_NAME}{RESET}")
    print()
    print(f"  {BOLD}Lancer le pipeline manuellement :{RESET}")
    print(f"    {CYAN}prefect deployment run 'kpi-pipeline-flow/signal-conso-pipeline'{RESET}")
    print()
    print(f"  {BOLD}Dashboard Prefect Cloud :{RESET}")
    print(f"    {CYAN}https://app.prefect.cloud{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  🚀 Setup Workspace Prefect — Signal Conso{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    step_install()
    step_auth()
    step_workspace()
    step_work_pool()
    step_variables()
    step_gcs_blocks()
    step_deploy()
    step_summary()


if __name__ == "__main__":
    main()
