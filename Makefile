.PHONY: build up down restart logs shell db pipeline

# Construire les images Docker
build:
	docker compose build

# Lancer les conteneurs
up:
	docker compose up -d

# Arrêter les conteneurs
down:
	docker compose down

# Redémarrer les services
restart:
	docker compose down
	docker compose up -d

# Voir les logs
logs:
	docker compose logs -f

# Entrer dans le conteneur API
shell:
	docker exec -it tri-demandes-api /bin/bash

# Entrer dans PostgreSQL
db:
	docker exec -it tri-demandes-db psql -U postgres -d SignalConso

# Lancer le pipeline ETL
pipeline:
	docker exec -it tri-demandes-api python -m scripts.run_pipeline

# Rebuild complet (utile si problème)
reset:
	docker compose down -v
	docker compose build
	docker compose up -d