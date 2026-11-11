# 🛡️ Plateforme MLOps Temps Réel de Détection de Fraude Bancaire

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-22c55e?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3b82f6?logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688?logo=fastapi&logoColor=white)](src/fraud/serving/api.py)
[![MLflow Registry](https://img.shields.io/badge/Registry-MLflow_v2.10-0194E2?logo=mlflow&logoColor=white)](http://localhost:5000)
[![Redpanda Streaming](https://img.shields.io/badge/Streaming-Redpanda%2FKafka-FD4659?logo=apachekafka&logoColor=white)](docker-compose.yml)
[![Streamlit Operations](https://img.shields.io/badge/Dashboard-Streamlit_FraudOps-FF4B4B?logo=streamlit&logoColor=white)](http://localhost:8501)
[![Tests Passing](https://img.shields.io/badge/Tests-17%2F17_Passing-success?logo=pytest&logoColor=white)](tests/)
[![DVC Versioned](https://img.shields.io/badge/Data-DVC_Tracked-945DD6?logo=dvc&logoColor=white)](data.dvc)

---

## 📌 Présentation du Projet

Ce projet implémente une **plateforme MLOps industrielle de bout en bout** pour la détection et la surveillance en temps réel de transactions bancaires frauduleuses par carte bancaire. 

Il combine un pipeline scientifique d'entraînement et d'explicabilité sans aucune fuite temporelle (*zero data leakage*), un bus de messagerie distribué résilient (Redpanda/Kafka), une API de serving haute performance (< 5 ms d'inférence), une base analytique SQLite idempotente, un moteur de surveillance de dérive (*data & concept drift*), et un tableau de bord opérationnel immersif (FraudOps Sentinel).

### 🎯 Le système en une phrase
> *Chaque transaction arrivant sur le réseau est ingérée, analysée et scorée en quelques millisecondes, renvoyant un verdict `FRAUD` ou `LEGIT`, un score de risque probabiliste, et ses 3 facteurs discriminants majeurs (SHAP) pour les analystes de risque.*

---

## 👥 Répartition des Rôles & Responsabilités (Binôme)

Conformément au contrat d'interface défini dans [`fraud-detection-plan/`](fraud-detection-plan/docs/04_contrat_anas_medo.md) :

| Rôle | Responsable | Périmètre & Livrables |
|---|---|---|
| **Pôle Machine Learning & Science des Données** | **Anas** | • Analyse exploratoire (EDA) & détection de biais.<br>• Découpage temporel strict chronologique (*Zero Data Leakage* : 70% train / 15% val / 15% test).<br>• Benchmark algorithmique (Régression Logistique, Random Forest, LightGBM, Optuna).<br>• Gestion du déséquilibre extrême (Poids de classe vs SMOTE vs Random Under-Sampling).<br>• Modèles non-supervisés (Isolation Forest pour anomalies, K-Means pour 5 profils comportementaux).<br>• Optimisation du seuil décisionnel via **fonction de coût financier métier** (minimisation des pertes directes).<br>• Explicabilité mathématique globale et cas locaux réels via **SHAP TreeExplainer**.<br>• Publication du modèle sous l'alias `Production` dans **MLflow Model Registry** et export de `threshold.json`. |
| **Pôle MLOps, Streaming & Système Temps Réel** | **Medo** | • Orchestration multi-conteneurs via **Docker Compose** (6 microservices isolés).<br>• Infrastructure de streaming distribuée sous **Redpanda** (cluster Kafka, topics `transactions` et `transactions.dlq`).<br>• Ingestion streaming : **Producer** (rejeu de test / génération synthétique, débit et injection de bursts configurables) & **Consumer** groupe résilient.<br>• Moteur de serving **FastAPI** asynchrone (`/predict`, `/predict_batch`, `/health`, `/metrics` Prometheus, validation stricte Pydantic v2, fallback gracieux).<br>• Persistance idempotente en base **SQLite** avec gestion des retries exponentiels et Dead Letter Queue (DLQ).<br>• Détection de dérive statistique (**Evidently** / tests de distribution KS sur features et concept drift).<br>• Dashboard de surveillance opérationnelle **Streamlit** (*FraudOps Sentinel* en 4 modules).<br>• Tests de charge **Locust** (p95 < 100 ms) et pipeline CI/CD **GitHub Actions**. |

---

## 🏗️ Architecture Globale du Système

```
                             ┌──────────────────────────────────────┐
                             │    Source de Données / Rejeu         │
                             │  (data/test.csv ou Générateur Synth) │
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │       Streaming Producer             │
                             │  (Cadence réglable, injection bursts)│
                             └──────────────────┬───────────────────┘
                                                │  Topic: transactions
                                                ▼
                             ┌──────────────────────────────────────┐
                             │       Redpanda Cluster (Kafka API)   │◄──── transactions.dlq
                             └──────────────────┬───────────────────┘       (Dead Letter Queue)
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐       MLflow Registry
                             │       Streaming Consumer             │◄────── models:/fraud-detector/Production
                             │ (Idempotence, Retries exponentiels)  │       (Chargement dynamique)
                             └──────────┬─────────────────┬─────────┘
                                        │                 │
                           POST /predict│                 │ Inserts SQL
                                        ▼                 ▼
                             ┌───────────────────┐ ┌────────────────┐
                             │   API FastAPI     │ │ Base SQLite    │
                             │ (Scoring + SHAP)  │ │ (data/fraud.db)│
                             └─────────┬─────────┘ └───────┬────────┘
                                       │                   │
                               Prometheus metrics          │ Live feed
                                       ▼                   ▼
                             ┌──────────────────────────────────────┐
                             │      Dashboard Streamlit             │
                             │     (FraudOps Sentinel)              │
                             │ 1. Live Stream & Matrice Risque/Amt  │
                             │ 2. Explicabilité SHAP Forensic       │
                             │ 3. Télémétrie Prometheus & SLA       │
                             │ 4. Surveillance Dérive (Evidently)   │
                             └──────────────────────────────────────┘
```

---

## 📊 Données & Rigueur Méthodologique

### Dataset : Credit Card Fraud Detection (ULB / Kaggle)
- **Volume total :** 284 807 transactions bancaires enregistrées sur 48 heures.
- **Déséquilibre extrême :** 492 transactions frauduleuses (soit **0,1727 %** du volume total, ~ 1 fraude pour 577 transactions saines).
- **Montant des fraudes :** Perte cumulée observée de 60 127,97 € (moyenne : 122,21 €, max : 2 125,87 €).
- **Variables :** `Time`, 28 composantes principales anonymisées ($V_1$ à $V_{28}$) issues d'une ACP, `Amount`, et `Class` (0 = légitime, 1 = fraude).

### Protocole de Découpage Temporel (Zero Data Leakage)
Le dataset a été découpé strictement selon l'ordre chronologique de la variable `Time` (aucun découpage aléatoire) :
1. **Entraînement (70 %, chronologique début) :** 199 364 transactions (384 fraudes, 0,193 %).
2. **Validation (15 %, chronologique milieu) :** 42 721 transactions (56 fraudes, 0,131 %) — étalonnage des hyperparamètres, sélection d'algorithmes et optimisation du seuil de coût.
3. **Test (15 %, chronologique fin) :** 42 722 transactions (52 fraudes, 0,122 %) — **touché une seule fois** à la fin du projet et rejoué en continu par le simulateur streaming.

> **Note méthodologique :** Les transformateurs de features (`Hour`, `log1p(Amount)`) et les techniques de rééquilibrage (SMOTE) ont été calibrés **exclusivement** sur le jeu d'entraînement.

---

## 🔬 Modélisation, Évaluation & Fonction de Coût Métier

### Fonction de Coût Financier Opérationnel
Le seuil de classification optimal $\theta^*$ n'a pas été choisi arbitrairement à 0,5, mais optimisé sur le jeu de validation pour minimiser le coût financier réel d'exploitation :

$$\text{Coût Total} = \sum_{i \in \text{FN}} \text{Amount}_i + C_{\text{verif}} \times (\text{FP} + \text{TP})$$

- **Fraude manquée (Faux Négatif - FN) :** Perte sèche égale au montant intégral de la transaction frauduleuse.
- **Fausse alerte (Faux Positif - FP) :** 5,00 € (coût d'intervention support et blocage temporaire).
- **Bonne détection (Vrai Positif - TP) :** 5,00 € (frais de vérification et dossier administratif).
- **Transaction légitime validée (Vrai Négatif - TN) :** 0,00 €.

### Comparatif des Modèles Expérimentés (Validation Set)

| Modèle | Stratégie Déséquilibre | PR-AUC (Val) | ROC-AUC (Val) | Recall (Val) | Precision (Val) | F1-Score | Coût Total (€) | Seuil Optimal $\theta^*$ | Statut |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | `class_weight='balanced'` | **0.8595** | **0.9851** | **78.57 %** | **100.00 %** | **0.8800** | **623,56 €** | **0.4631** | **🏆 Champion Retenu** |
| LightGBM | Random Under-Sampling (0.1) | 0.8477 | 0.9868 | 82.14 % | 85.19 % | 0.8364 | 435,30 € | 0.9851 | Challenger |
| Logistic Regression | `class_weight='balanced'` | 0.8456 | 0.9823 | 87.50 % | 44.95 % | 0.5939 | 706,55 € | 0.9900 | Baseline |
| LightGBM | SMOTE (0.1) | 0.8311 | 0.9882 | 80.36 % | 83.33 % | 0.8182 | 436,30 € | 0.6945 | Écarté |
| LightGBM (Tuned) | Optuna Search | 0.8070 | 0.9830 | 78.57 % | 84.62 % | 0.8148 | 750,89 € | 0.5468 | Écarté |
| LightGBM (Défaut) | `scale_pos_weight` brute | 0.0294 | 0.8796 | 83.93 % | 2.56 % | 0.0497 | 9 315,91 € | 0.9900 | Rejeté (Sur-alerte) |

### Modèles Non-Supervisés Complémentaires
- **Isolation Forest (`contamination=0.002`) :** Modélisation non-supervisée des anomalies pures sur les transactions saines, produisant un score d'anomalie inversé indépendant des étiquettes historiques.
- **K-Means Clustering ($k=5$) :** Profilage des comportements transactionnels (ex. micro-paiements récurrents, achats de journée standard, transactions nocturnes à montant élevé avec sur-représentation critique des fraudes à > 1,8 %).

### Évaluation Finale sur le Jeu de Test (Un Seul Passage)
Le pipeline **Random Forest** calibré avec son seuil métier optimal $\theta^* = 0{,}4631$ a été évalué sur les 42 722 transactions du jeu de test :

| Métrique Test | Valeur | Interprétation Industrielle |
|---|---|---|
| **PR-AUC** | **0.7664** | Rétention de performance solide sur les données futures |
| **ROC-AUC** | **0.9836** | Capacité de discrimination inter-classes quasi parfaite |
| **Recall (Rappel)** | **75.00 %** | **39 fraudes réelles interceptées sur 52** |
| **Precision** | **86.67 %** | **Seulement 6 fausses alertes** sur 42 722 transactions traitées |
| **F1-Score** | **0.8041** | Équilibre optimal protection bancaire / friction client |
| **Coût Total Test** | **2 597,40 €** | **Réduction de plus de 85 %** des pertes face à l'absence de modèle |

---

## 🧠 Explicabilité Mathématique (SHAP Forensic)

L'interprétabilité du modèle repose sur **SHAP (SHapley Additive exPlanations - TreeExplainer)** :

1. **Variables Discriminantes Globales (Top Facteurs) :**
   - **$V_{12}$** (mean $|SHAP| = 0.0763$) : Principal indicateur de rupture comportementale.
   - **$V_{14}$** (mean $|SHAP| = 0.0675$) : Détecteur structurel de signature frauduleuse.
   - **$V_4$** (mean $|SHAP| = 0.0664$) : Facteur de risque à forte corrélation positive.
   - **$V_{11}$**, **$V_{10}$**, **$V_{17}$** : Composantes factorielles de confirmation.
2. **Explications Locales Dynamiques :**
   Pour chaque transaction analysée, l'API extrait dynamiquement les 3 composantes qui ont le plus pesé dans la décision, permettant aux enquêteurs de justifier immédiatement chaque blocage (conformité RGPD / AI Act).

---

## ⚡ Spécifications de l'API FastAPI

L'API sert le modèle sérialisé depuis le registre MLflow ou ses artefacts locaux avec une latence unitaire inférieure à 5 ms.

### Endpoints Disponibles

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Retourne la santé du service, le statut du modèle chargé, la version MLflow et l'URI. |
| `GET` | `/metrics` | Expose les métriques Prometheus (`fraud_api_requests_total`, `latency`, `fraud_detected`). |
| `POST` | `/predict` | Inférence temps réel unitaire avec score de risque, verdict et explications SHAP top 3. |
| `POST` | `/predict_batch` | Inférence vectorisée pour traitement par lots de transactions. |

### Contrat de Données `POST /predict`

**Requête JSON :**
```json
{
  "transaction_id": "tx_2026_001",
  "Time": 406.0,
  "V1": -1.3598,
  "V2": -0.0728,
  "...": 0.0,
  "V28": -0.0211,
  "Amount": 780.0
}
```

**Réponse JSON :**
```json
{
  "transaction_id": "tx_2026_001",
  "fraud_probability": 0.9412,
  "decision": "FRAUD",
  "threshold": 0.4631,
  "model_version": "Production-v1",
  "top_features": [
    {"feature": "V14", "shap_value": 0.1361},
    {"feature": "V10", "shap_value": 0.0834},
    {"feature": "V12", "shap_value": 0.0729}
  ],
  "latency_ms": 3.82
}
```

*Codes d'erreur : Validation Pydantic stricte (montant négatif, champ manquant ou valeur nulle $\implies$ `HTTP 422 Unprocessable Entity`).*

---

## 📈 Surveillance de Dérive (Evidently Drift Engine)

Le système intègre une veille continue sur deux dimensions statistiques :
1. **Data Drift (Dérive des Features) :** Écart de distribution calculé par test de Kolmogorov-Smirnov / distance Wasserstein entre la distribution de référence (`data/train.csv`) et le flux live (`data/test.csv`).
2. **Concept Drift (Dérive du Risque) :** Surveillance de la proportion moyenne de fraude et du déplacement de la frontière décisionnelle.

*Démonstration dynamique :*
En injectant `DEMO_DRIFT=true`, le Producer applique des distorsions synthétiques sur les variables $V_1$ à $V_5$ et sur les montants, déclenchant instantanément l'indicateur d'alerte rouge sur le dashboard.

---

## 🖥️ Tableau de Bord Opérationnel : FraudOps Sentinel

Accessible sur le port `8501`, le dashboard Streamlit présente un thème dark ergonomique structuré en 4 centres d'opérations :
1. **🛰️ Live Stream Operations :** Flux défilant en temps réel, alertes rougeoyantes pour les fraudes, timeline de risque et matrice Risque vs Montant.
2. **🔍 SHAP Forensic Investigation :** Analyseur d'impact unitaire pour chaque transaction (waterfall des contributions, comparaison profil fraude vs profil sain).
3. **⚡ System Telemetry & SLA :** Jauge de respect de la latence (< 100 ms), compteurs Prometheus de débit et statut d'intégrité de la Dead Letter Queue.
4. **📊 Evidently Drift Governance :** Matrice de dérive par feature, rapport de stabilité des distributions et logs d'alerte.

---

## 🚀 Guide de Démarrage Rapide

### Prérequis
- **Docker** et **Docker Compose** installés
- **Python 3.11+** (pour exécution locale)
- **Make** (facultatif mais recommandé)

### 1. Démarrage Complet en une Commande (Docker Compose)
Pour lancer l'ensemble des 6 services conteneurisés :

```bash
docker compose up -d
```
*ou via le Makefile :*
```bash
make up
```

### 2. Accès aux Services

| Service | URL d'accès | Description |
|---|---|---|
| **Dashboard FraudOps** | [http://localhost:8501](http://localhost:8501) | Interface de surveillance temps réel |
| **Swagger API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Documentation interactive OpenAPI / FastAPI |
| **API Healthcheck** | [http://localhost:8000/health](http://localhost:8000/health) | État du service et du modèle MLflow |
| **Métriques Prometheus** | [http://localhost:8000/metrics](http://localhost:8000/metrics) | Télémétrie brute temps réel |
| **Serveur MLflow** | [http://localhost:5000](http://localhost:5000) | Registre des modèles et tracking des runs |
| **Redpanda Broker** | `localhost:9092` (interne) / `19092` | Broker Kafka haute performance |

---

## 🧪 Tests, Qualité & Benchmarking

### Exécution des Tests Automatisés
La suite comprend 17 tests unitaires et d'intégration validant le feature engineering, les métriques financières, l'API et la liaison MLflow :

```bash
python -m pytest tests/unit -v
```

### Contrôle Qualité & Linting
```bash
ruff check .
black --check .
```
*ou via :*
```bash
make lint
```

### Test de Charge & Résilience (Locust)
Pour simuler 50 utilisateurs concurrents requêtant l'API de scoring :

```bash
locust -f tests/load/locustfile.py --headless -u 50 -r 10 -t 60s
```
*Objectif SLA contractuel : p95 < 100 ms (atteint en conditions réelles : ~ 4-12 ms).*

### Déclenchement de la Démonstration de Drift
Pour générer un rapport de dérive simulé et observer la bascule d'état sur le dashboard :

```bash
DEMO_DRIFT=true DRIFT_STRENGTH=2.0 python -m src.fraud.monitoring.drift \
  --reference data/train.csv --current data/test.csv --output data/drift_report.json
```

---

## 📁 Arborescence Complète du Projet

```text
Fraude_Detection-main/
├── .github/workflows/ci.yml       # Pipeline CI/CD GitHub Actions (Lint, Test, Docker)
├── configs/                       # Configurations YAML versionnées
│   ├── data.yaml                  # Chemins des jeux de données et remote DVC
│   ├── features.yaml              # Définitions des variables et transformations
│   ├── serving.yaml               # Paramètres API, hôte, port et seuil actif
│   └── train.yaml                 # Configuration de l'entraînement et modèle cible
├── dashboard/
│   └── app.py                     # Dashboard Streamlit opérationnel (FraudOps Sentinel)
├── data/                          # Données (gérées via DVC pour les volumineux)
│   ├── raw/creditcard.csv         # Dataset source ULB (versionné DVC)
│   ├── train.csv                  # Split entraînement (70 %, chronologique)
│   ├── val.csv                    # Split validation (15 %, chronologique)
│   ├── test.csv                   # Split test (15 %, chronologique)
│   ├── fraud.db                   # Base de données SQLite des prédictions live
│   ├── threshold.json             # Seuil optimal métier validé (0.4631)
│   ├── model_comparison.csv       # Tableau comparatif complet des modèles
│   └── drift_report.json          # Rapport statistique de dérive généré
├── fraud-detection-plan/          # Documentation contractuelle & suivi de projet
│   ├── docs/                      # Spécifications fonctionnelles, système, architecture
│   └── tasks/                     # Tâches et jalons hebdomadaires (ANAS.md, MEDO.md)
├── notebooks/                     # Notebooks Jupyter exploratoires
│   └── 01_exploratory_data_analysis.ipynb
├── report_assets/                 # Graphiques haute résolution du rapport académique
├── src/fraud/                     # Code source modulaire du package de production
│   ├── data/
│   │   ├── eda.py                 # Analyse exploratoire et statistiques
│   │   ├── features.py            # Transformateur FraudFeatureEngineer Scikit-Learn
│   │   └── loader.py              # Ingestion et découpage chronologique sans fuite
│   ├── models/
│   │   ├── anomaly.py             # Isolation Forest et clustering K-Means
│   │   ├── baseline.py            # Pipeline de référence Régression Logistique
│   │   ├── explain.py             # Analyse d'explicabilité SHAP TreeExplainer
│   │   ├── metrics.py             # PR-AUC, matrice de coût et recherche de seuil
│   │   ├── train.py               # Pipeline d'entraînement et enregistrement MLflow
│   │   └── tune.py                # Optimisation bayésienne des hyperparamètres (Optuna)
│   ├── monitoring/
│   │   └── drift.py               # Moteur de détection de dérive (features & concept)
│   ├── serving/
│   │   └── api.py                 # API FastAPI, Pydantic, Prometheus & SHAP
│   ├── storage/
│   │   └── db.py                  # Schéma SQLite, persistance et requêtes d'agrégation
│   └── streaming/
│       ├── consumer.py            # Consommateur Kafka résilient, retries & DLQ
│       ├── producer.py            # Simulateur de streaming temps réel avec anomalies
│       └── simulator.py           # Gestionnaire du rejeu de transactions
├── tests/
│   ├── load/locustfile.py         # Scénario de test de charge Locust (50 utilisateurs)
│   └── unit/                      # Tests unitaires et d'intégration pytest (17 tests)
├── docker-compose.yml             # Orchestration des 6 conteneurs de production
├── Dockerfile.api                 # Conteneur API FastAPI
├── Dockerfile.dashboard           # Conteneur Dashboard Streamlit
├── Dockerfile.streaming           # Conteneur Streaming Producer / Consumer
├── Makefile                       # Raccourcis de commandes d'exploitation
├── pyproject.toml                 # Métadonnées du projet et dépendances de build
├── requirements.txt               # Dépendances Python de production
├── REPORT_ANAS.md                 # Rapport scientifique et technique complet (Anas)
├── PHASES_1_2_REPORT.md           # Bilan d'exécution des phases 1 & 2
├── PHASES_3_4_REPORT.md           # Bilan d'exécution des phases 3 & 4
├── Rapport_Academique_Detection_Fraude.pdf # Document PDF académique complet
└── Presentation_Detection_Fraude.pptx      # Support de présentation pour soutenance
```

---

## 🎓 Livrables Académiques & Documents de Soutenance

Le projet s'accompagne d'un corpus documentaire complet prêt pour l'évaluation et la soutenance :
- 📄 **Rapport Scientifique Complet (Markdown) :** [`REPORT_ANAS.md`](REPORT_ANAS.md)
- 📑 **Rapport Académique Mis en Page (PDF) :** [`Rapport_Academique_Detection_Fraude.pdf`](Rapport_Academique_Detection_Fraude.pdf) / [`rapport_academique.html`](rapport_academique.html)
- 📊 **Support de Présentation Interactif (HTML) :** [`presentation_soutenance.html`](presentation_soutenance.html)
- 📑 **Support de Soutenance (PDF) :** [`Presentation_Soutenance_MLOps.pdf`](Presentation_Soutenance_MLOps.pdf)
- 📽️ **Support PowerPoint :** [`Presentation_Detection_Fraude.pptx`](Presentation_Detection_Fraude.pptx)

---

## 🔒 Limites Formelles & Perspectives

1. **Variables Anonymisées ($V_1$ à $V_{28}$) :** Les composantes résultant d'une ACP ne permettent pas de nommer en clair les comportements précis (ex. "changement soudain de pays"), même si SHAP quantifie exactement leur impact relatif.
2. **Absence d'identifiant client récurrent (`client_id`) :** Empêche le calcul de features d'historique de compte (ex. vélocité horaire par carte). Le feature engineering s'est donc concentré sur la temporalité (`Hour`) et le profilage monétaire (`log1p(Amount)`).
3. **Horizon temporel restreint (48 heures) :** Absence de visibilité sur les cycles hebdomadaires ou mensuels de paie.
4. **Perspectives :** Intégration d'un Graph Neural Network (GNN) pour la détection de réseaux de fraude organisée, et déploiement d'un feature store distribué (Feast).
