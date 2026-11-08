# ANAS : le modèle

**Mission :** produire le meilleur modèle possible, bien évalué et expliqué.

## Semaine 1 : données et baseline
- [x] Charger `creditcard.csv`, vérifier les colonnes
- [x] Analyse exploratoire : déséquilibre, distributions par classe, `Amount`, `Time`
- [x] Découpage dans l'ordre du temps (train / validation / test)
- [x] Baseline : régression logistique avec poids de classe
- [x] Fonction d'évaluation commune : PR-AUC, recall, precision, coût

## Semaine 2 : features et modèles
- [x] Features : heure de la journée, log de `Amount`
- [x] Random Forest, XGBoost, LightGBM, CatBoost
- [x] Gestion du déséquilibre : poids de classe vs SMOTE vs undersampling, comparés
- [x] Chaque essai enregistré dans MLflow

## Semaine 3 : modèles complémentaires, réglage, seuil, SHAP
- [x] Isolation Forest (détection d'anomalies)
- [x] K-Means pour profiler les transactions
- [x] Optuna sur les 1 ou 2 meilleurs modèles
- [x] Seuil choisi par fonction de coût
- [x] Évaluation finale sur le test, **une seule fois**
- [x] SHAP global et quelques cas (fraude détectée, fraude manquée, fausse alerte)
- [x] Modèle final et `threshold.json` dans MLflow

## Semaine 4 : rapport
- [x] Méthodologie, résultats, SHAP, limites du dataset

## Livrables
1. Notebook d'analyse exploratoire
2. Code d'entraînement dans `src/` (pas seulement dans des notebooks)
3. Tableau comparatif des modèles (export MLflow)
4. Modèle final + seuil dans MLflow
5. Données de référence pour Evidently
6. Rapport

## Règles
- Pas de fuite de données : tout ce qui apprend se fait sur l'entraînement seulement.
- Seed fixé (42).
- Le test n'est touché qu'une fois.
