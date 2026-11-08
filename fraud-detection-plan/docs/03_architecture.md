# 03. Architecture

```
Simulateur → Redpanda → Consumer → API → Base de données → Dashboard
                                   ↑                          ↑
                          modèle d'Anas (MLflow)     Evidently (drift) + Grafana
```

## Les composants

| Composant | Rôle | Responsable |
|---|---|---|
| Entraînement + MLflow | Produire et sauvegarder le modèle et son seuil | Anas |
| Simulateur | Rejoue la partie test du dataset, transaction par transaction | Medo |
| Redpanda | Transporte les transactions (file d'attente) | Medo |
| Consumer | Lit une transaction et appelle l'API | Medo |
| API (FastAPI) | Calcule le score, applique le seuil, renvoie le verdict et le SHAP | Medo |
| Base de données (SQLite) | Stocke tous les résultats | Medo |
| Dashboard (Streamlit) | Affiche tout en direct | Medo |
| Evidently | Compare les données live à celles de l'entraînement (drift) | Medo |
| Grafana | Affiche latence et volume | Medo |
| Docker Compose | Lance tout avec une commande | Medo |

## Flux d'une transaction
1. Le simulateur envoie la transaction à Redpanda.
2. Le consumer la lit et appelle l'API.
3. L'API renvoie `{score, verdict, top_features}`.
4. Le consumer enregistre le résultat en base. Si l'API échoue, il réessaie puis envoie le message dans une file d'erreurs.
5. Le dashboard lit la base et se met à jour.

## Décisions
- Le modèle est un **pipeline scikit-learn complet** : même traitement à l'entraînement et en production.
- Le **seuil** est stocké à part : on peut le changer sans réentraîner.
- Une seule commande pour tout démarrer : `docker compose up`.
