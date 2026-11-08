# 02. Données : Credit Card Fraud (ULB, Kaggle)

| Élément | Valeur |
|---|---|
| Transactions | 284 807 |
| Fraudes | 492 (0,17 %) |
| Colonnes | `Time`, `V1` à `V28` (PCA anonymisées), `Amount`, `Class` (0 = légitime, 1 = fraude) |
| Période | 2 jours |

## Découpage (important)
Découpage **dans l'ordre de `Time`**, pas au hasard :
- Début → entraînement
- Milieu → validation (réglages, seuil)
- Fin → test (utilisé une seule fois, et rejoué par le simulateur)

Les transformations et SMOTE s'appliquent **sur l'entraînement uniquement** (SMOTE dans les folds de validation croisée).

## Métrique principale
PR-AUC, recall et precision. Pas l'accuracy : un modèle qui répond toujours `LEGIT` a 99,83 % d'accuracy et ne sert à rien.

## Limites à écrire dans le rapport
1. **Pas d'identifiant client** : impossible de créer des features du type « nombre de transactions de ce client dans l'heure ». Le feature engineering se limite à `Time` (heure de la journée) et `Amount` (log, z-score).
2. **Variables anonymisées** : l'explication dira « V14 », pas « montant inhabituel ».
3. **Seulement 2 jours** de données : pas de saisonnalité.
4. **Très peu de fraudes** (492) : résultats à interpréter avec prudence.

## Organisation des fichiers
```
data/raw/        # creditcard.csv, versionné avec DVC, jamais dans Git
data/processed/  # features prêtes
```
