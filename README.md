# Plateforme intelligente de tri et de priorisation des demandes clients

## Contexte
Les entreprises reçoivent chaque jour un volume important de demandes clients via plusieurs canaux : formulaires web, e-mails, tickets support ou messageries internes. Le traitement manuel de ces demandes est souvent lent, hétérogène et source d’erreurs de routage ou de priorisation.

Ce projet propose une plateforme d’intelligence artificielle capable de centraliser ces demandes, de les préparer, puis de les classer automatiquement afin d’aider les équipes support à gagner du temps et à traiter les cas les plus urgents en priorité.

## Objectif métier
L’objectif est de concevoir une solution applicative qui permet de :
- collecter des demandes depuis plusieurs sources de données ;
- nettoyer, normaliser et structurer les informations ;
- stocker les données dans une base relationnelle conforme aux bonnes pratiques ;
- classer automatiquement chaque demande par catégorie métier ;
- estimer son niveau de priorité ;
- exposer les prédictions via une API REST ;
- préparer le projet à une logique d’industrialisation et de MLOps.

## Fonctionnalités attendues
- ingestion de données depuis fichiers CSV, base de données ou API ;
- transformation et nettoyage des données ;
- stockage dans PostgreSQL ;
- mise à disposition via une API REST développée avec FastAPI ;
- modèle de machine learning pour la classification de texte ;
- évaluation du modèle et suivi des performances ;
- tests automatisés sur les composants critiques ;
- conteneurisation avec Docker ;
- journalisation et bases de monitorage.

## Périmètre fonctionnel
Le système vise principalement trois usages :
1. **Qualifier** une demande client à partir de son sujet et de sa description.
2. **Prioriser** la demande selon des règles métier et/ou un modèle IA.
3. **Exposer** les résultats à d’autres composants ou à une interface applicative.

## Cas d’usage principal
Un agent saisit ou importe une demande client. La plateforme analyse le texte, détecte la catégorie probable (par exemple : facturation, incident technique, commande, réclamation), estime une priorité (basse, moyenne, haute) et retourne un résultat exploitable par l’équipe support.

## Données utilisées
Les données pourront provenir de :
- fichiers CSV ou Excel simulés ;
- jeux de données publics ;
- export de tickets support ;
- API ou scraping de contenu de FAQ pour enrichir les catégories.

Les jeux de données seront organisés dans le dépôt pour distinguer :
- `data/raw/` : données brutes ;
- `data/processed/` : données nettoyées et préparées ;
- `data/external/` : sources externes ;
- `data/samples/` : exemples réduits pour les tests et la démonstration.

## Architecture technique
L’architecture du projet est organisée autour de quatre blocs principaux :

- **Ingestion / Data Engineering** : extraction, nettoyage, transformation et chargement des données ;
- **Base de données** : stockage des demandes et des métadonnées dans PostgreSQL ;
- **API applicative** : exposition des données et des prédictions via FastAPI ;
- **Machine Learning / MLOps** : entraînement, évaluation, export du modèle, tests et monitorage.

### Technologies envisagées
- Python 3.11+
- FastAPI
- PostgreSQL
- SQLAlchemy
- pandas
- scikit-learn
- pytest
- Docker
- Uvicorn

## Structure du dépôt
```text
plateforme-tri-demandes-clients/
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── samples/
├── notebooks/
│   └── 01_exploration_donnees.ipynb
├── docs/
│   ├── architecture.md
│   ├── cahier_des_charges.md
│   ├── modele_donnees.md
│   ├── api_spec.md
│   └── soutenance.md
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes/
│       │   │   ├── health.py
│       │   │   ├── tickets.py
│       │   │   └── predictions.py
│       │   └── schemas/
│       │       ├── ticket.py
│       │       └── prediction.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── session.py
│       │   ├── models.py
│       │   └── migrations/
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── extract.py
│       │   ├── transform.py
│       │   └── load.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── text_preprocessing.py
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── evaluate.py
│       │   └── model_registry.py
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── drift.py
│       │   └── metrics.py
│       └── utils/
│           ├── __init__.py
│           └── logger.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_ingestion.py
│   ├── test_transformation.py
│   ├── test_ml.py
│   └── test_db.py
├── scripts/
│   ├── seed_db.py
│   ├── run_pipeline.py
│   ├── train_model.py
│   └── export_predictions.py
├── models/
│   ├── .gitkeep
│   └── README.md
└── .github/
    └── workflows/
        ├── ci.yml
        └── cd.yml
```

## Installation et démarrage
### 1. Cloner le dépôt
```bash
git clone <repo-url>
cd plateforme-tri-demandes-clients
```

### 2. Préparer l’environnement
```bash

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Only if you don't have .env yet
cp -n .env.example .env
pip install -r requirements.txt
```

### 3. Lancer l’API en local
```bash
uvicorn src.app.main:app --reload
```

### 4. Lancer avec Docker
```bash
docker compose up --build
```

## Commandes utiles
```bash
make setup
make install_precommit
make format
make run_tests
```

## API
### Endpoints principaux
- `GET /health` : vérification de l’état du service ;
- `GET /` : page d’accueil de l’API ;
- `POST /tickets/predict` : prédiction de catégorie et de priorité à partir d’une demande client.

### Exemple de requête
```json
{
  "subject": "Problème de facture",
  "description": "Je n'arrive pas à télécharger ma facture du mois dernier."
}
```

### Exemple de réponse
```json
{
  "category": "facturation",
  "priority": "moyenne",
  "confidence": 0.78
}
```

## Entraînement du modèle
Le modèle de base peut être entraîné à partir d’un fichier CSV contenant au minimum :
- une colonne `text` ;
- une colonne `category`.

L’entraînement est prévu dans `src/app/ml/train.py`, avec une approche simple de classification de texte basée sur `scikit-learn`.

## Tests
Les tests automatisés couvrent les composants clés du projet :
- API ;
- ingestion ;
- transformation des données ;
- base de données ;
- modèle de machine learning.

Exécution :
```bash
pytest
```

## Déploiement Docker
Le projet est prévu pour fonctionner dans des conteneurs Docker afin de faciliter :
- la reproductibilité de l’environnement ;
- le déploiement local ;
- l’intégration continue ;
- la préparation à un déploiement plus industrialisé.

## MLOps et monitorage
Le projet intègre une logique MLOps autour de :
- la version des données et du modèle ;
- les tests de validation avant mise en production ;
- le suivi des performances du modèle ;
- la détection d’une dérive éventuelle des données ou des prédictions ;
- la journalisation des événements et erreurs.

## Livrables de soutenance
- cahier des charges fonctionnel ;
- schéma d’architecture technique ;
- modèle conceptuel / physique des données ;
- pipeline d’ingestion et de transformation ;
- API REST documentée ;
- modèle IA entraîné et évalué ;
- tests automatisés ;
- conteneur Docker ;
- documentation technique ;
- démonstration fonctionnelle.

## Auteurs
Projet réalisé dans le cadre de la certification de développeur en intelligence artificielle – Data Engineering.
