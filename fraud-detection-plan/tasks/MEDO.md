# MEDO : le système et le dashboard

**Mission :** faire tourner le modèle d'Anas en temps réel et le montrer.

## Semaine 1 : fondations
- [ ] Repo GitHub, structure de dossiers, `Makefile`
- [ ] MLflow en Docker
- [ ] API squelette avec le faux modèle (voir contrat)
- [ ] CI GitHub Actions (lint + tests)

## Semaine 2 : API
- [ ] `POST /predict` et `GET /health`
- [ ] Charge le modèle et le seuil depuis MLflow
- [ ] Renvoie verdict, score et top variables SHAP
- [ ] Tests (`pytest`), Dockerfile
- [ ] Test de charge Locust : latence visée < 100 ms

## Semaine 3 : temps réel
- [ ] Redpanda en Docker
- [ ] Simulateur : rejoue la partie test du dataset
- [ ] Consumer : lit, appelle l'API, enregistre en SQLite
- [ ] Réessais et file d'erreurs si l'API échoue

## Semaine 4 : dashboard, drift, démo
- [ ] Dashboard Streamlit (voir `docs/01_systeme_et_interface.md`)
- [ ] Evidently : détection de drift
- [ ] Grafana : latence et volume
- [ ] `docker-compose.yml` : tout démarre avec une commande
- [ ] `make demo` et `make demo-drift`

## Livrables
1. Repo propre, CI verte, README de reproduction
2. API dockerisée testée, résultat du test de charge
3. Flux temps réel fonctionnel
4. Dashboard (5 zones)
5. Schéma d'architecture

## Critères de fin
- `git clone` puis `docker compose up` démarre tout
- Latence p95 < 100 ms mesurée
- Démo répétée au moins 2 fois
