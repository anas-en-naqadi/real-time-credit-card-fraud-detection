# 05. Planning (4 semaines, à adapter)

| Semaine | Anas | Medo | Jalon |
|---|---|---|---|
| 1 | Données, split, baseline | Repo, MLflow, API squelette, CI | Contrat validé, baseline dans MLflow |
| 2 | Features, boosting, déséquilibre | API complète, tests, Docker, test de charge | L'API sert le vrai modèle |
| 3 | Isolation Forest, K-Means, Optuna, seuil, SHAP | Redpanda, simulateur, consumer, SQLite | Flux de bout en bout |
| 4 | Rapport, modèle final | Dashboard, drift, Grafana, démo | Démo répétée |

## Si vous prenez du retard, couper dans cet ordre
Grafana → drift → K-Means → Optuna. Ne jamais couper : split temporel, seuil par coût, SHAP, API, dashboard, démo.

## Rythme
Point de 15 minutes tous les 2 jours. Chacun relit les pull requests de l'autre. Pas de push direct sur `main`.
