# 04. Contrat Anas ↔ Medo

> À valider ensemble le jour 1. Il permet à Medo d'avancer avec un faux modèle pendant qu'Anas entraîne le vrai.

## Entrée de l'API (`POST /predict`)
```json
{
  "transaction_id": "tx_43",
  "Time": 406.0,
  "V1": -1.35, "...": "...", "V28": -0.02,
  "Amount": 780.0
}
```

## Sortie
```json
{
  "transaction_id": "tx_43",
  "fraud_score": 0.94,
  "verdict": "FRAUD",
  "threshold": 0.37,
  "top_features": [
    {"feature": "V14", "shap_value": 1.7},
    {"feature": "V4",  "shap_value": 0.8},
    {"feature": "V12", "shap_value": 0.6}
  ]
}
```
`verdict` ∈ {`FRAUD`, `LEGIT`}.

## Ce qu'Anas livre
1. Un **pipeline scikit-learn** complet enregistré dans MLflow sous le nom `fraud-detector`.
2. Un fichier `threshold.json` (le seuil).
3. Les données de référence (le jeu d'entraînement) pour Evidently.

## Endpoints de l'API
`POST /predict`, `GET /health`.

## Fonction de coût (à fixer ensemble)
```
fraude manquée   = montant de la transaction
fausse alerte    = 5 (coût de vérification, à ajuster)
bonne détection  = 5
bon passage      = 0
```
Anas choisit le seuil qui minimise le coût total sur la validation.

## Faux modèle de départ
Une régression logistique simple enregistrée sous `fraud-detector` version 1. Même interface : rien à changer quand le vrai modèle arrive.
