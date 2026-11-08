# 06. Livrables, démo, checklist

## Livrable final
| Élément | Description |
|---|---|
| Repo GitHub | Code, tests, Docker, CI, README |
| Système | `docker compose up` lance tout |
| Modèle | Pipeline + seuil dans MLflow |
| Rapport PDF | Méthodologie, résultats, SHAP, architecture, limites |
| Slides | Support de soutenance |
| Vidéo de démo | Plan de secours |

## Démo (5 à 7 minutes)
1. `make demo` : le flux défile, la majorité des lignes sont `LEGIT`.
2. Une ligne `FRAUD` apparaît en rouge : cliquer dessus, montrer SHAP.
3. Montrer les chiffres d'efficacité : fraudes détectées sur fraudes réelles.
4. `make demo-drift` : l'état du drift passe à `Alerte`.
5. Montrer MLflow : comparaison des modèles, modèle en production.

Anas explique les modèles et le SHAP. Medo pilote la démo et explique l'architecture.

## Plan du rapport
1. Contexte et objectif
2. Données et limites
3. Méthodologie (split temporel, métriques, coût)
4. Modèles et gestion du déséquilibre
5. Résultats (tableau comparatif)
6. Seuil et impact
7. Explicabilité (SHAP)
8. Architecture temps réel
9. Performance (latence)
10. Limites et perspectives

## Checklist finale
- [ ] Split temporel documenté, aucune fuite
- [ ] Baseline + au moins 3 modèles comparés
- [ ] Stratégies de déséquilibre comparées
- [ ] Isolation Forest et K-Means réalisés
- [ ] Seuil choisi par coût
- [ ] Test utilisé une seule fois
- [ ] SHAP global et cas locaux
- [ ] API testée, latence mesurée
- [ ] Flux temps réel fonctionnel
- [ ] Dashboard affiche les 5 zones
- [ ] Drift démontré
- [ ] `docker compose up` marche depuis un clone propre
- [ ] Rapport relu, slides prêts, démo répétée 2 fois
- [ ] Les deux savent expliquer tout le projet
