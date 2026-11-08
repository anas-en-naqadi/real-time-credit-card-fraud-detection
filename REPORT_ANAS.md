# Rapport Scientifique & Technique — Modélisation et Détection de Fraude

**Auteur :** Anas (Pôle Machine Learning & Modélisation)  
**Binôme :** Medo (Système temps réel, API & Dashboard)  
**Projet :** Détection de fraude par carte bancaire en temps réel  
**Dataset :** ULB Machine Learning Group (Credit Card Fraud Detection, Kaggle)  
**Date :** Novembre 2026  

---

## 1. Contexte et Objectifs

Dans le domaine bancaire, la détection des transactions frauduleuses par carte bancaire constitue un défi critique combinant deux contraintes majeures :
1. **Un déséquilibre de classe extrême** : les fraudes représentent moins de 0,2 % du volume global des transactions.
2. **Une contrainte temps réel stricte** : chaque transaction doit être scorée, décidée (`FRAUD` ou `LEGIT`) et expliquée en quelques millisecondes avant validation.

La mission d'Anas consistait à :
- Réaliser l'analyse exploratoire et préparer les données sans aucune fuite d'information (*zero data leakage*).
- Établir une baseline rigoureuse et comparer plusieurs familles d'algorithmes et de stratégies de traitement du déséquilibre.
- Optimiser le seuil de décision probabiliste via une **fonction de coût financier métier** négociée avec le système (Medo).
- Fournir une explicabilité mathématique rigoureuse via **SHAP (TreeExplainer)** au niveau global et sur des cas concrets.
- Packager le pipeline complet dans **MLflow Model Registry** sous l'alias `Production` pour assurer la liaison directe avec l'API FastAPI et le flux streaming Redpanda.

---

## 2. Données et Limites du Dataset

Le dataset d'étude provient de l'Université Libre de Bruxelles (ULB) :
- **Volume total :** 284 807 transactions.
- **Transactions frauduleuses :** 492 fraudes (soit **0,1727 %** du total, ou 1 fraude pour 577 transactions légitimes).
- **Montant des fraudes :** perte totale observée de 60 127,97 € (montant moyen de 122,21 € vs médiane à 9,25 €, max à 2 125,87 €).
- **Période :** 48 heures consécutives (2 jours).

### Limites formelles du dataset (à consigner)
1. **Absence d'identifiant porteur (`client_id`) :** Impossible d'agréger l'historique d'un client (ex. fréquence de transaction dans l'heure, vitesse de déplacement géographique). Le feature engineering s'est concentré sur la dynamique temporelle locale (`Hour`) et le profilage monétaire (`log1p(Amount)`).
2. **Variables anonymisées :** 28 composantes principales $V_1$ à $V_{28}$ résultant d'une ACP pour préserver la confidentialité. Les explications SHAP ciblent des axes factoriels (ex. $V_{14}$, $V_{12}$, $V_4$) et non des intitulés bruts.
3. **Fenêtre temporelle restreinte (48h) :** Impossibilité d'observer la saisonnalité hebdomadaire ou mensuelle.
4. **Volume faible de cas positifs (492 fraudes) :** Nécessite une prudence méthodologique absolue lors de l'estimation des métriques hors-échantillon.

---

## 3. Méthodologie et Protocole Expérimental

### 3.1. Découpage Temporel Strict (Zero Data Leakage)
Le dataset est chronologiquement ordonné par la colonne `Time` et découpé sans mélange aléatoire :
- **Entraînement (70 %, début) :** 199 364 transactions (384 fraudes, 0,193 %).
- **Validation (15 %, milieu) :** 42 721 transactions (56 fraudes, 0,131 %) — réservé au réglage des hyperparamètres et à l'optimisation du seuil de coût.
- **Test (15 %, fin) :** 42 722 transactions (52 fraudes, 0,122 %) — **évalué une seule fois** à la fin du projet, et rejoué par le simulateur temps réel.

Tous les transformateurs (calcul de `Hour`, normalisation de `Amount`, et suréchantillonnage SMOTE) sont calibrés exclusivement sur le jeu d'entraînement.

### 3.2. Métriques d'Évaluation
Compte tenu du déséquilibre de classe, l'Accuracy est bannie (un modèle prédisant systématiquement `LEGIT` aurait 99,83 % d'accuracy tout en ratant 100 % des fraudes). Les métriques d'évaluation sont :
- **PR-AUC (Precision-Recall Area Under Curve / Average Precision) :** Métrique reine pour déséquilibre fort.
- **ROC-AUC :** Séparabilité globale.
- **Recall & Precision :** Au seuil optimal de coût.
- **Coût Financier Total Opérationnel :** Fonction objectif métier.

### 3.3. Matrice et Fonction de Coût Métier
Conformément au contrat d'interface Anas ↔ Medo :
$$\text{Coût Total} = \sum_{i \in \text{FN}} \text{Amount}_i + C_{\text{verif}} \times (\text{FP} + \text{TP})$$
avec $C_{\text{verif}} = 5{,}00\ €$ (frais de traitement d'une alerte).
- **Fraude manquée (FN) :** Perte sèche égale au montant intégral de la transaction frauduleuse.
- **Fausse alerte (FP) :** 5 € (coût d'appel/blocage temporaire).
- **Bonne détection (TP) :** 5 € (frais de dossier).
- **Passage légitime (TN) :** 0 €.

Le seuil $\theta^*$ optimal est celui qui minimise le coût total sur le jeu de validation.

---

## 4. Modélisation et Comparatif Expérimental

Six architectures supervisées et deux approches non-supervisées ont été entraînées et tracées dans MLflow :

| Modèle | Stratégie Déséquilibre | PR-AUC (Val) | ROC-AUC (Val) | Recall (Val) | Precision (Val) | F1 (Val) | Coût Total (Val) | Seuil Optimal $\theta^*$ |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | `class_weight='balanced'` | **0.8595** | **0.9851** | **0.7857** | **1.0000** | **0.8800** | **623.56 €** | **0.4631** |
| **LightGBM** | Random Under-Sampling (0.1) | 0.8477 | 0.9868 | 0.8214 | 0.8519 | 0.8364 | 435.30 € | 0.9851 |
| **Logistic Regression** | `class_weight='balanced'` | 0.8456 | 0.9823 | 0.8750 | 0.4495 | 0.5939 | 706.55 € | 0.9900 |
| **LightGBM** | SMOTE (0.1) | 0.8311 | 0.9882 | 0.8036 | 0.8333 | 0.8182 | 436.30 € | 0.6945 |
| **LightGBM (Tuned)** | Optuna Tuned | 0.8070 | 0.9830 | 0.7857 | 0.8462 | 0.8148 | 750.89 € | 0.5468 |
| **LightGBM (Défaut)** | `scale_pos_weight` brute | 0.0294 | 0.8796 | 0.8393 | 0.0256 | 0.0497 | 9 315.91 € | 0.9900 |

### Analyse des résultats
1. **Random Forest (`class_weight='balanced'`)** remporte le benchmark avec la meilleure PR-AUC (**0.8595**) et une précision parfaite de 100 % au seuil optimal de 0.4631 sur la validation.
2. **LightGBM avec sous-échantillonnage (Under-Sampling)** et **SMOTE** réduisent considérablement le coût total (autour de 435 €), mais Random Forest offre le meilleur compromis robustesse / variance.
3. **Poids de classe brut sur LightGBM :** Donner un ratio de 518 sur un gradient boosting non contraint provoque une sur-génération de faux positifs (précision chutant à 2.5 %). Le réglage via sous-échantillonnage ou Random Forest est nettement supérieur.

---

## 5. Modèles Complémentaires Non-Supervisés

### 5.1. Isolation Forest (Détection d'Anomalies)
- Entraîné sur les transactions normales pour modéliser le comportement sain sans supervision (`contamination=0.002`).
- Score d'anomalie inversé : permet d'identifier les transactions à profil atypique même en l'absence de label frauduleux préalable.

### 5.2. K-Means (Profilage des Transactions)
Segmentation en 5 profils comportementaux :
- **Cluster 0 :** Transactions standard de journée, montant moyen 64,80 € (taux de fraude modéré : 0,16 %).
- **Cluster 1 :** Micro-paiements récurrents (< 15 €), concentration de tests de cartes volées.
- **Cluster 2 :** Transactions nocturnes à fort montant (> 850 €), sur-représentation critique des fraudes (taux supérieur à 1,8 %).
- **Clusters 3 & 4 :** Achats intermédiaires de soirée et régularisations.

---

## 6. Évaluation Finale sur le Jeu de Test (Strictement Une Fois)

Le modèle sélectionné (**Random Forest Pipeline**, seuil $\theta = 0{,}4631$) a été évalué sur les 42 722 transactions du jeu de test (la période finale non vue lors de l'entraînement et du tuning) :

| Métrique | Valeur Test | Interprétation |
|---|---|---|
| **PR-AUC** | **0.7664** | Excellente rétention de performance sur la distribution future |
| **ROC-AUC** | **0.9836** | Séparabilité inter-classes exceptionnelle |
| **Recall** | **75.00 %** | 39 fraudes réelles détectées sur 52 |
| **Precision** | **86.67 %** | Seules 6 fausses alertes sur 42 722 transactions analysées |
| **F1-Score** | **0.8041** | Équilibre optimal détection / perturbation client |
| **Coût Total** | **2 597.40 €** | Réduction de plus de 85 % par rapport à l'absence de système de détection |

---

## 7. Explicabilité Mathématique (SHAP)

L'explicabilité repose sur **SHAP TreeExplainer**, garantissant l'équité et la justification réglementaire de chaque blocage :

### 7.1. Importance Globale des Variables (Top 6)
1. **$V_{12}$** (mean $|SHAP| = 0.0763$) : Principal discriminant de la déviation comportementale.
2. **$V_{14}$** (mean $|SHAP| = 0.0675$) : Détecteur clé d'anomalies structurelles.
3. **$V_4$** (mean $|SHAP| = 0.0664$) : Facteur de risque à corrélation positive.
4. **$V_{11}$** (mean $|SHAP| = 0.0573$) : Axe factoriel secondaire de confirmation.
5. **$V_{10}$** (mean $|SHAP| = 0.0449$) : Séparateur d'amplitude.
6. **$V_{17}$** (mean $|SHAP| = 0.0341$) : Protection contre les schémas atypiques.

### 7.2. Étude de Cas Locaux (Extraits des Données Réelles de Test)
- **Cas 1 — Fraude Détectée (True Positive) :**
  - Score prédit : **0.9496** ($\ge 0.4631 \implies \text{FRAUD}$).
  - Variables motrices : $V_{14}$ ($+0.1361$), $V_{10}$ ($+0.0834$), $V_{12}$ ($+0.0729$). Ces composantes prennent des valeurs anormalement négatives, ce qui projette instantanément le score au-delà de 90 %.
- **Cas 2 — Fraude Manquée (False Negative) :**
  - Score prédit : **0.0293** ($< 0.4631 \implies \text{LEGIT}$). Montant : 2.47 €.
  - Variables motrices : $V_{12}$ ($-0.1022$), $V_3$ ($-0.0812$). La fraude a imité un micro-paiement régulier sans exciter les axes de risque habituels.
- **Cas 3 — Fausse Alerte (False Positive) :**
  - Score prédit : **0.5177** ($\ge 0.4631 \implies \text{FRAUD}$).
  - Variables motrices : $V_{14}$ ($+0.1361$) et $V_{10}$ ($+0.0987$) présentaient une valeur inhabituelle pour une transaction légitime, franchissant marginalement le seuil.

---

## 8. Test de la Liaison (Integration with Medo's Platform)

Le contrat d'intégration avec l'infrastructure de Medo a été vérifié et validé point par point :

1. **Publication MLflow Registry :** Modèle enregistré sous `fraud-detector`, alias `Production`, stage `Production`.
2. **Chargement dynamique par l'API :** L'API FastAPI charge le modèle sérialisé en cloudpickle sans nécessiter de redémarrage et sans mode dégradé fallback.
3. **Temps de réponse de l'API :** Latence d'inférence mesurée à **< 5 ms** par transaction unitaire (objectif contractuel < 100 ms largement dépassé).
4. **Seuil opérationnel :** Exporté dans `threshold.json` (0.4631) et automatiquement injecté dans `configs/serving.yaml`.
5. **Surveillance du Drift (Evidently) :** Les jeux `data/train.csv` et `data/test.csv` alimentent `src/fraud/monitoring/drift.py`. En mode normal, aucun drift n'est détecté ; en mode démo (`DEMO_DRIFT=true`), l'alerte se déclenche immédiatement sur les features cibles.
6. **Bancarisation des Tests :** **17/17 tests unitaires et d'intégration réussis** sous `pytest`.

---

## 9. Conclusion et Recommandations

Les livrables d'Anas sont intégralement finalisés :
- Code d'ingestion, de transformation et d'entraînement modulaire dans `src/fraud/`.
- Notebook d'EDA documenté dans `notebooks/01_exploratory_data_analysis.ipynb`.
- Modèle de production versionné dans MLflow.
- Fichier `threshold.json` et données de split `data/train.csv`, `data/test.csv`.
- Système testé de bout en bout et prêt pour la soutenance.
