# CONTEXTE DU PROJET — Source unique pour générer les slides

> **Règle absolue pour le générateur de slides :** n'utiliser QUE les chiffres et faits de ce fichier.
> Ne rien inventer. Chaque chiffre ci-dessous a été vérifié dans le code ou dans les artefacts du dépôt
> (`data/*.json`, `data/model_comparison.csv`, `src/`, `docker-compose.yml`, tests).

---

## 0. Identité du projet

| Élément | Valeur |
|---|---|
| Titre | Système industriel et MLOps de détection de fraude bancaire en temps réel |
| Sous-titre | Modélisation asymétrique des coûts, explicabilité SHAP et ingestion streaming événementielle |
| Formation | Master IAII — Module Data Mining — Projet industriel MLOps |
| Année | 2026 |
| Équipe | **Anas EN-NAQADI** — Pôle Machine Learning, optimisation et explicabilité |
| | **Mohammed MAHMOUD (Medo)** — Pôle architecture MLOps, streaming et dashboard |
| Dataset | Credit Card Fraud Detection — ULB / Worldline (Kaggle) |

**Le système en une phrase :** il reçoit des transactions en continu et répond pour chacune `FRAUD` ou `LEGIT`, avec un score de 0 à 1 et les 3 variables qui ont le plus pesé (SHAP), en quelques millisecondes.

---

## 1. Problématique métier

- La directive européenne **DSP2** oblige la banque émettrice à rembourser les transactions non autorisées : chaque fraude ratée est une **perte directe**.
- **Trois verrous :**
  1. **Déséquilibre extrême** : moins de 2 transactions sur 1 000 sont frauduleuses.
  2. **Le piège de l'accuracy** : un modèle qui répond toujours `LEGIT` obtient **99,827 %** d'accuracy, rate **100 %** des fraudes et laisse passer **60 127,97 €** de pertes sur le dataset.
  3. **Temps réel + auditabilité** : décision en **< 100 ms**, avec une explication de chaque refus (RGPD, droit à l'explication).

---

## 2. Données

| Élément | Valeur |
|---|---|
| Transactions | 284 807 |
| Fraudes | 492 (**0,1727 %**, soit 1 fraude pour 577 transactions légitimes) |
| Valeurs manquantes | 0 |
| Colonnes | `Time`, `V1`…`V28` (composantes ACP anonymisées), `Amount`, `Class` (0 = légitime, 1 = fraude) |
| Période | 48 heures (2 jours) |
| Montant des fraudes | moyenne 122,21 € · médiane 9,25 € · max 2 125,87 € · total 60 127,97 € |
| Montant des légitimes | moyenne 88,29 € · médiane 22,00 € · max 25 691,16 € |
| Variables les plus corrélées à la fraude | V17 (−0,327), V14 (−0,303), V12 (−0,261), V10 (−0,217), V16 (−0,197) |
| Versionnage | `data/raw/creditcard.csv` suivi par **DVC** (`data.dvc`), jamais dans Git |

**Limites du dataset (à afficher honnêtement) :**
1. Pas d'identifiant client → impossible de construire l'historique d'un porteur (ex. « nombre de transactions dans l'heure »).
2. Variables anonymisées → l'explication dit « V14 », pas « pays inhabituel ».
3. Seulement 2 jours → aucune saisonnalité observable.
4. Peu de fraudes (492, dont 52 en test) → résultats à interpréter avec prudence.

---

## 3. Protocole expérimental (zéro fuite de données)

Découpage **chronologique** selon `Time`, sans mélange aléatoire (pas de biais d'anticipation) :

| Partition | Part | Transactions | Fraudes | Taux | Usage |
|---|---|---|---|---|---|
| Entraînement (passé) | 70 % | 199 364 | 384 | 0,193 % | Apprentissage des modèles et des transformateurs |
| Validation (milieu) | 15 % | 42 721 | 56 | 0,131 % | Choix du modèle, hyperparamètres, seuil θ* |
| Test (futur) | 15 % | 42 722 | 52 | 0,122 % | **Évaluation unique** + rejeu par le simulateur streaming |

- Transformations et SMOTE ajustés **sur l'entraînement uniquement**.
- Seed fixée à **42**.
- Le taux de fraude baisse d'une partition à l'autre → la distribution évolue dans le temps (justifie la surveillance du drift).

**Feature engineering** — transformateur scikit-learn `FraudFeatureEngineer` (`src/fraud/data/features.py`), intégré au pipeline (même traitement à l'entraînement et en production) :
- `Hour = (Time div 3600) mod 24` → cycle jour/nuit.
- `log_amount = ln(1 + Amount)` → compresse une distribution très asymétrique.
- Sortie : 31 variables (V1–V28 + Hour + log_amount + Amount).

**Métriques :** PR-AUC (principale), ROC-AUC, Recall, Precision, F1, **coût financier total**. Accuracy bannie.

---

## 4. Fonction de coût métier (contrat Anas ↔ Medo)

$$C(\theta) = \sum_{i \in FN(\theta)} \text{Amount}_i + 5\,€ \times (|FP(\theta)| + |TP(\theta)|)$$

| Issue | Situation | Coût |
|---|---|---|
| Faux négatif (FN) | Fraude ratée | Montant intégral de la transaction |
| Faux positif (FP) | Fausse alerte sur un client légitime | 5,00 € |
| Vrai positif (TP) | Fraude interceptée (gestion du dossier) | 5,00 € |
| Vrai négatif (TN) | Transaction légitime validée | 0,00 € |

- Exemple d'asymétrie : bloquer une fraude de 500 € = gain net **+495 €** ; bloquer un micro-paiement suspect de 2 € = **−3 €**.
- θ* = argmin C(θ) sur une grille de **200 seuils**, calculé sur la **validation**.
- Le seuil est stocké à part (`threshold.json`, `configs/serving.yaml`) → modifiable **sans réentraîner**.

---

## 5. Benchmark des modèles (jeu de validation)

Source : `data/model_comparison.csv` (chaque essai tracé dans **MLflow**).

| Modèle | Stratégie de déséquilibre | PR-AUC | ROC-AUC | Recall | Precision | F1 | Coût | θ* |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** ✅ retenu | class_weight balanced | **0,8595** | 0,9851 | 78,57 % | **100 %** | **0,880** | 623,56 € | **0,4631** |
| LightGBM | Under-sampling (0,1) | 0,8477 | 0,9868 | 82,14 % | 85,19 % | 0,836 | **435,30 €** | 0,9851 |
| Régression logistique (baseline) | class_weight | 0,8456 | 0,9823 | 87,50 % | 44,95 % | 0,594 | 706,55 € | 0,9900 |
| LightGBM | SMOTE (0,1) | 0,8311 | 0,9882 | 80,36 % | 83,33 % | 0,818 | 436,30 € | 0,6945 |
| LightGBM | Optuna (tuning) | 0,8070 | 0,9830 | 78,57 % | 84,62 % | 0,815 | 750,89 € | 0,5468 |
| LightGBM | scale_pos_weight brut (518) | 0,0294 | 0,8796 | 83,93 % | 2,56 % | 0,050 | 9 315,91 € | 0,9900 |

**Messages clés :**
- Random Forest : meilleure PR-AUC et zéro fausse alerte en validation → retenu pour la production.
- LightGBM + under-sampling : coût le plus bas, mais seuil extrême (0,985) et variance plus forte.
- `scale_pos_weight` brut sur LightGBM : explosion des faux positifs (precision 2,56 %) → leçon : le poids de classe ne se transpose pas naïvement au boosting.
- Stratégies de déséquilibre comparées : poids de classe vs SMOTE vs under-sampling.

**Modèles non supervisés complémentaires** (`src/fraud/models/anomaly.py`) :
- **Isolation Forest** (150 arbres, contamination = 0,002) : score d'anomalie sans utiliser les labels.
- **K-Means (k = 5)** — profils réels (`data/kmeans_profiles.json`) :

| Cluster | Transactions | Montant moyen | Heure moyenne | Fraudes | Taux de fraude |
|---|---|---|---|---|---|
| 0 | 5 187 | 1 170,50 € | 13,0 h | 13 | 0,25 % |
| 1 | 145 441 | 57,48 € | 13,6 h | 128 | 0,09 % |
| 2 | 28 466 | 87,04 € | 13,2 h | 4 | 0,01 % |
| 3 (nuit) | 20 034 | 48,30 € | **3,4 h** | 43 | 0,21 % |
| **4** | **236** | 87,76 € | 8,7 h | **196** | **83,05 %** |

→ Le cluster 4 isole 196 fraudes sur 236 transactions : K-Means retrouve une « signature » de fraude sans supervision.

---

## 6. Évaluation finale sur le test (utilisé une seule fois)

Source : `data/final_model_summary.json`. Modèle : pipeline Random Forest, θ = 0,4631.

| Métrique | Valeur |
|---|---|
| PR-AUC | 0,7664 (−0,093 vs validation) |
| ROC-AUC | 0,9836 |
| Recall | **75,00 %** → 39 fraudes interceptées sur 52 |
| Precision | **86,67 %** → 6 fausses alertes (45 alertes au total) |
| F1 | 0,8041 |
| Matrice de confusion | TP = 39 · FP = 6 · FN = 13 · TN = 42 664 |

**Bilan économique sur le test (vérifié sur `data/test.csv`) :**

| Poste | Montant |
|---|---|
| Pertes sans aucun modèle (somme des 52 fraudes) | **6 168,88 €** |
| Pertes dues aux 13 fraudes ratées | 2 372,40 € |
| Frais d'investigation (45 alertes × 5 €) | 225,00 € |
| **Coût total avec le modèle** | **2 597,40 €** |
| **Économie nette** | **3 571,48 € (−57,9 %)** |

> ⚠️ Le rapport PDF et `REPORT_ANAS.md` annoncent « 17 640 € de pertes brutes » et « −85,3 % ». Ce chiffre **ne correspond pas** aux données : la somme réelle des fraudes du test est 6 168,88 €. Utiliser **−57,9 %** dans les slides (ou corriger le rapport avant la soutenance).

---

## 7. Explicabilité (SHAP TreeExplainer)

`src/fraud/models/explain.py` — Complexité O(T·L·D²).

**Importance globale** (`data/shap_analysis.json`, mean |SHAP|) :
V14 (0,0865) · V12 (0,0776) · V4 (0,0623) · V10 (0,0568) · V11 (0,0556) · V17 (0,0381)

**Cas locaux réels :**
| Cas | Score | Verdict | Explication |
|---|---|---|---|
| Vrai positif | 94,96 % | FRAUD | V14 (+0,136), V10 (+0,083), V12 (+0,073) très anormaux |
| Faux négatif | 2,93 % | LEGIT | Fraude de 2,47 € imitant un micro-paiement ; V12 (−0,102), V3 (−0,081). Perte (2,47 €) < coût d'une investigation (5 €) |
| Faux positif | 51,77 % | FRAUD | Transaction légitime inhabituelle, franchit de peu le seuil (V14 +0,136) |

En production, l'API renvoie les **3 variables SHAP principales** pour chaque transaction.

---

## 8. Architecture temps réel

```
Producer → Redpanda (topic "transactions") → Consumer → API FastAPI → SQLite → Dashboard Streamlit
                                   ↘ transactions.dlq          ↑
                                                       MLflow Registry
                                                  models:/fraud-detector/Production
```

**Docker Compose — 6 services** (`docker-compose.yml`) :

| Service | Image / build | Port | Rôle |
|---|---|---|---|
| mlflow | mlflow v2.10.2 | 5000 | Tracking + Model Registry |
| api | Dockerfile.api | 8000 | Scoring, seuil, SHAP, métriques |
| redpanda | redpanda (API Kafka) | 9092 / 19092 / 9644 | Bus de messages |
| producer | Dockerfile.streaming | — | Rejoue `data/test.csv` à 5 tx/s |
| consumer | Dockerfile.streaming | — | Lit, appelle l'API, écrit en base |
| dashboard | Dockerfile.dashboard | 8501 | Supervision en direct |

Healthchecks sur l'API et Redpanda ; le consumer attend que les deux soient « healthy ».

**Flux d'une transaction :**
1. Le producer publie la transaction sur `transactions` (5 tx/s, `FRAUD_INJECTION_RATE=0.05`, burst toutes les 100 transactions via `FRAUD_BURST_EVERY`).
2. Le consumer (groupe `fraud-detector-consumer`) la lit.
3. Il appelle `POST /predict` avec réessais exponentiels.
4. Il enregistre le résultat en SQLite, puis commite l'offset Kafka.
5. Le dashboard lit la base et se rafraîchit automatiquement.

**API FastAPI** (`src/fraud/serving/api.py`) :

| Route | Rôle |
|---|---|
| `POST /predict` | Score + verdict + seuil + version du modèle + top 3 SHAP + latence |
| `POST /predict_batch` | Même chose pour une liste de transactions |
| `GET /health` | État du service, modèle chargé, version, URI MLflow, mode fallback |
| `GET /metrics` | Prometheus : `fraud_api_requests_total`, `fraud_api_inference_latency_seconds`, `fraud_api_fraud_detected_total` |

- Validation **Pydantic v2** stricte (`extra="forbid"`, `Amount ≥ 0`) → HTTP **422** si montant négatif, champ manquant ou null.
- Charge le modèle depuis `models:/fraud-detector/Production` ; **fallback** sécurisé si MLflow est indisponible.

Exemple de réponse :
```json
{"transaction_id": "tx_43", "fraud_probability": 0.94, "decision": "FRAUD",
 "threshold": 0.4631, "model_version": "1",
 "top_features": [{"feature":"V14","shap_value":0.136}, {"feature":"V10","shap_value":0.083}, {"feature":"V12","shap_value":0.073}],
 "latency_ms": 3.4}
```

---

## 9. Résilience

1. **Idempotence** : table `predictions` avec `transaction_id PRIMARY KEY` + `INSERT OR IGNORE` → aucun doublon si Kafka relivre un message. SQLite en mode WAL.
2. **Réessais exponentiels** : `API_RETRY_ATTEMPTS=5`, délai initial 0,25 s doublé à chaque tentative ; déclenchés sur erreurs HTTP 5xx et erreurs réseau.
3. **Dead Letter Queue** `transactions.dlq` : JSON invalide, transaction sans ID ou API définitivement indisponible → message mis en quarantaine, le flux ne bloque pas.
4. **Commit manuel des offsets** (`enable_auto_commit=False`) : l'offset n'est validé qu'après traitement ou envoi en DLQ.

Schéma SQLite : `transaction_id, probability, decision, latency_ms, timestamp, amount, top_driver, shap_json`.

---

## 10. Performance

| Mesure | Valeur | Source |
|---|---|---|
| Latence d'inférence unitaire | < 5 ms | Rapport + réponses API |
| Test de charge Locust | 50 utilisateurs, 10/s de montée, 60 s | `tests/load/locustfile.py` |
| Latence médiane mesurée | 3,4 ms | Rapport (pas de fichier de résultats stocké dans le dépôt) |
| p95 mesuré | < 8,2 ms | Rapport |
| Objectif contractuel | p95 < 100 ms | Plan projet |

---

## 11. Surveillance du drift

`src/fraud/monitoring/drift.py` compare la référence (`data/train.csv`) au flux (`data/test.csv`) :

$$TVD(P,Q) = \tfrac{1}{2}\sum_b |P(b) - Q(b)|$$

(histogrammes à 10 intervalles, par variable V1–V28 + Amount). Seuil d'alerte : **0,20** (`DRIFT_THRESHOLD`).
Concept drift : alerte si le taux de fraude varie de plus de 5 points.

| Scénario | Résultat | Fichier |
|---|---|---|
| Normal (train vs test) | Aucune alerte · max TVD = 0,140 (V15), puis V11 0,128 | `data/test_drift_report.json` |
| Démo `DEMO_DRIFT=true`, force 2,0 | **Alerte** · V4 = **0,529** | `data/drift_report.json` |

Le mode démo décale V1–V5 de +2,0 et multiplie Amount par 1,2.
Pourquoi c'est important : les fraudes sont confirmées 30 à 60 jours plus tard → le drift des entrées est la seule alerte précoce.

---

## 12. Dashboard « FraudOps Sentinel » (Streamlit, port 8501)

Thème sombre, rafraîchissement automatique (intervalle réglable dans la barre latérale), seuil lu depuis `threshold.json`.

| Onglet | Contenu |
|---|---|
| 🛰️ Live Stream Operations | Courbe des scores en direct, nuage Montant vs Score, flux des dernières transactions (FRAUD en rouge **et** en texte) |
| 🔍 SHAP Forensic Investigation | Décomposition des contributions SHAP d'une transaction, disposition analyste |
| ⚡ System Telemetry & SLA | Latence, volume, état de l'exporteur Prometheus `/metrics` |
| 📊 Evidently Drift Governance | Scores de drift par variable, état OK / Alerte |

En haut : chiffres clés (transactions analysées, alertes fraude, latence).

---

## 13. Qualité, tests et CI/CD

- **17 tests pytest, 17 réussis** (vérifié) :
  - `test_api.py` (6) : health, metrics, predict, 422 sur montant négatif, champs manquants, batch.
  - `test_features.py` (3) : forme de sortie, calcul de Hour, log_amount.
  - `test_metrics.py` (3) : métriques et seuil de coût.
  - `test_models.py` (5) : intégration MLflow → API (modèle chargé, pas de fallback), seuil = `threshold.json`, transaction légitime → LEGIT, transaction frauduleuse → top features attendues, drift sur 42 722 lignes.
- **GitHub Actions** (`.github/workflows/ci.yml`) : sur push/PR vers `main` → install, `black --check`, `ruff check`, `pytest tests/unit`, build de l'image Docker de l'API.
- `pre-commit`, Ruff (88 colonnes), Black.
- **Makefile** : `make setup`, `make lint`, `make test`, `make up`.

---

## 14. Répartition du travail

| Anas EN-NAQADI — ML | Mohammed MAHMOUD — MLOps |
|---|---|
| EDA (`notebooks/01_exploratory_data_analysis.ipynb`) | Redpanda, topics, DLQ |
| Split temporel sans fuite | Producer / consumer résilients |
| `FraudFeatureEngineer` | API FastAPI + Prometheus |
| Benchmark 6 modèles + déséquilibre | Persistance SQLite idempotente |
| Isolation Forest, K-Means, Optuna | Dashboard Streamlit (4 onglets) |
| Fonction de coût + seuil θ* | Drift monitoring |
| SHAP global et local | Docker Compose (6 services), CI, Locust |
| MLflow Registry + DVC | Makefile, tests API |

Planning en 4 semaines : S1 données/baseline + repo/CI → S2 modèles + API → S3 non supervisé/seuil/SHAP + streaming → S4 rapport + dashboard/drift/démo.

---

## 15. Démo (5 à 7 minutes)

1. `docker compose up -d` → montrer les 6 services.
2. Dashboard (http://localhost:8501) : le flux défile, la majorité des lignes sont LEGIT.
3. Une ligne FRAUD apparaît en rouge → ouvrir l'onglet SHAP.
4. Swagger (http://localhost:8000/docs) : envoyer une transaction, puis une avec `Amount = -1` → 422.
5. `/metrics` : compteurs Prometheus.
6. Drift : `DEMO_DRIFT=true DRIFT_STRENGTH=2.0 python -m src.fraud.monitoring.drift --reference data/train.csv --current data/test.csv --output data/drift_report.json` → l'onglet drift passe en Alerte (V4 = 0,529).
7. MLflow (http://localhost:5000) : comparaison des runs, modèle `fraud-detector` en Production.

Plan de secours : vidéo de démo enregistrée.

---

## 16. Écarts honnêtes plan ↔ réalisation (pour la slide limites)

| Prévu dans le plan | État réel |
|---|---|
| Grafana pour latence et volume | Non déployé : métriques Prometheus exposées et affichées dans l'onglet Telemetry du dashboard |
| `make demo` / `make demo-drift` | Absents du Makefile : la démo passe par `make up` et la commande drift ci-dessus |
| Evidently | Dépendance installée ; le calcul de drift est une TVD maison dans `drift.py` |
| Dashboard « sans bouton » | Le dashboard final a une barre latérale (filtres, rafraîchissement) et une zone de disposition analyste |

---

## 17. Conclusion et perspectives

- Un Random Forest avec seuil calibré par le coût (0,4631) intercepte 75 % des fraudes du futur avec 6 fausses alertes sur 42 722 transactions, et réduit le coût de **57,9 %**.
- Chaîne complète : données versionnées → modèle tracé → API < 5 ms → streaming résilient → supervision et drift.
- **Perspectives :** feature store (Feast) pour l'historique des porteurs, réentraînement automatique déclenché par le drift, déploiement Kubernetes, Grafana, calibration des probabilités.
