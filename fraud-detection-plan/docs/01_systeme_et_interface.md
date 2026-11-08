# 01. Le système et l'interface

## Ce que fait le système
- **Entrée :** une transaction ULB (`Time`, `V1` à `V28`, `Amount`).
- **Sortie :** `FRAUD` ou `LEGIT`, un score de 0 à 1, et les variables qui ont le plus pesé (SHAP).
- **Quand :** en temps réel, quelques millisecondes après l'arrivée de la transaction.

Règle : score ≥ seuil → `FRAUD`. Le seuil est choisi par Anas pour minimiser le coût (rater une fraude coûte plus cher qu'une fausse alerte).

## L'interface : un seul écran, en lecture seule

Le dashboard n'a **aucun bouton de pilotage**. Il se met à jour tout seul.

### Ce que l'utilisateur voit (de haut en bas)

| Zone | Contenu |
|---|---|
| 1. Chiffres clés | Transactions analysées · Alertes fraude · Latence |
| 2. Efficacité | Fraudes détectées sur fraudes réelles (ex. 18 sur 21) · Fausses alertes |
| 3. Flux en direct | Tableau des dernières transactions : id, montant, score, verdict |
| 4. Détail de l'alerte | Variables qui ont fait monter le score (SHAP) |
| 5. État du drift | `OK` ou `Alerte` |

### Ce que l'utilisateur peut faire
- **Regarder** : le flux défile tout seul, aucune action requise.
- **Cliquer sur une ligne du tableau** pour afficher son détail SHAP en zone 4. Par défaut, la zone 4 montre la dernière alerte.
- C'est tout. Pas de bouton, pas de formulaire, pas de réglage.

### Règles d'interface
- Les lignes `FRAUD` sont en rouge **et** portent le texte `FRAUD` (la couleur seule ne suffit pas).
- Pas plus de 3 chiffres clés en haut.
- Textes courts, aucun jargon technique à l'écran.
- Mise à jour automatique toutes les 1 à 2 secondes.

### Comment la démo est pilotée (sans boutons dans l'interface)
En ligne de commande, avant et pendant la soutenance :
- `make demo` : flux normal
- `make demo-drift` : flux modifié pour déclencher l'alerte de drift

## Exemple
`tx_43` arrive avec `Amount = 780`, `V14 = -9.2`, `V4 = 5.1`. Score : 0,94. Seuil : 0,37. L'API répond `FRAUD`, facteur principal V14. La ligne passe en rouge.

## Hors périmètre (volontairement)
Boutons de validation humaine, réglages, comptes utilisateurs, autoencodeur.
