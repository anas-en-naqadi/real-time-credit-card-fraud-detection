# PROMPT — Génération de la présentation de soutenance

> Copier tout le bloc ci-dessous dans l'outil de génération (Gamma, ChatGPT, Claude, Canva, Google Slides + Gemini…)
> **et joindre le fichier `CONTEXTE_SLIDES.md`** (ou le coller à la suite).

---

```text
Tu es un expert en présentations académiques et techniques. Crée une présentation de soutenance
de 18 slides, en français, pour un projet de Master IAII (module Data Mining) intitulé :
« Système industriel et MLOps de détection de fraude bancaire en temps réel ».
Présentateurs : Anas EN-NAQADI (Machine Learning) et Mohammed MAHMOUD (MLOps, streaming, dashboard).
Durée : 15 minutes de présentation + 5 à 7 minutes de démo en direct.

SOURCE UNIQUE : le fichier CONTEXTE_SLIDES.md joint. N'invente AUCUN chiffre, AUCUN outil,
AUCUNE fonctionnalité qui n'y figure pas. Si un chiffre manque, laisse « [à compléter] ».
Pour le gain financier sur le test, utilise -57,9 % (3 571,48 € économisés) et non 85 %.

STYLE
- Thème sombre et professionnel (bleu nuit #0F172A, accents bleu #3B82F6, vert #10B981 pour LEGIT,
  rouge #EF4444 pour FRAUD, ambre #F59E0B pour les alertes).
- Une idée par slide. 6 puces maximum, phrases courtes. Pas de pavés de texte.
- Les chiffres clés sont mis en avant en grand (style « big number »).
- Tableaux lisibles (au maximum 6 colonnes ; supprimer les colonnes secondaires si besoin).
- Schémas en blocs et flèches plutôt qu'en texte.
- Chaque slide a des notes orateur (3 à 5 phrases) indiquant qui parle (Anas ou Medo).
- Numéro de slide et pied de page « Master IAII · Data Mining · 2026 ».

STRUCTURE (respecter l'ordre et le contenu)

Slide 1 — Titre
  Titre, sous-titre (coûts asymétriques, SHAP, streaming), noms et pôles des deux auteurs,
  Master IAII · Data Mining · 2026. Phrase d'accroche : « FRAUD ou LEGIT, expliqué, en moins de 5 ms ».

Slide 2 — Problématique métier
  DSP2 : la banque rembourse chaque fraude ratée. Les 3 verrous : déséquilibre extrême,
  piège de l'accuracy, temps réel + droit à l'explication (RGPD).
  Big number : « 99,83 % d'accuracy… et 0 fraude détectée ».

Slide 3 — Objectifs et système en une phrase
  Entrée (Time, V1–V28, Amount) → Sortie (score 0–1, verdict FRAUD/LEGIT, top 3 SHAP).
  Objectifs chiffrés : p95 < 100 ms, seuil choisi par le coût, lancement en une commande Docker.

Slide 4 — Le dataset ULB
  284 807 transactions, 492 fraudes (0,1727 %, 1 sur 577), 48 h, 0 valeur manquante.
  Visuel : barre ou donut montrant le déséquilibre. Montant médian fraude 9,25 € vs légitime 22 €.

Slide 5 — Limites du dataset
  Les 4 limites (pas d'ID client, ACP anonymisée, 2 jours, peu de fraudes) avec leur conséquence.

Slide 6 — Protocole sans fuite de données
  Frise chronologique 70 / 15 / 15 avec volumes, fraudes et usage de chaque bloc.
  Messages : tri par Time, transformations et SMOTE sur l'entraînement uniquement, test utilisé une seule fois, seed 42.

Slide 7 — Feature engineering
  FraudFeatureEngineer (transformateur scikit-learn dans le pipeline) : formules Hour et log_amount,
  pourquoi (cycle jour/nuit, asymétrie des montants), 31 variables en sortie. Métriques : PR-AUC, pas l'accuracy.

Slide 8 — Fonction de coût métier
  Formule C(θ), matrice FN / FP / TP / TN avec les coûts, exemple +495 € vs −3 €,
  θ* = argmin sur 200 seuils (validation). Seuil stocké à part, modifiable sans réentraîner.

Slide 9 — Benchmark des modèles
  Tableau des 6 configurations (Modèle, Stratégie, PR-AUC, Precision, Coût, θ*).
  Mettre en évidence Random Forest (retenu) et l'échec de scale_pos_weight brut.
  3 enseignements en bas de slide.

Slide 10 — Approches non supervisées
  Isolation Forest (contamination 0,002) et K-Means k=5.
  Highlight : le cluster 4 contient 236 transactions dont 196 fraudes (83 %), trouvées sans labels.
  Mentionner aussi le cluster nocturne (heure moyenne 3,4 h).

Slide 11 — Résultats finaux sur le test
  Big numbers : Recall 75 % (39/52), Precision 86,67 % (6 fausses alertes), ROC-AUC 0,984, PR-AUC 0,766.
  Mini matrice de confusion TP 39 · FP 6 · FN 13 · TN 42 664.

Slide 12 — Impact financier
  Cascade : 6 168,88 € sans modèle → 2 372,40 € de fraudes ratées + 225 € d'investigation
  = 2 597,40 € → économie de 3 571,48 € (−57,9 %). Rappeler la prudence (52 fraudes seulement).

Slide 13 — Explicabilité SHAP
  Barres d'importance globale (V14, V12, V4, V10, V11, V17).
  3 cartes de cas réels : vrai positif 94,96 %, faux négatif 2,93 % (2,47 €), faux positif 51,77 %.
  Message : chaque décision est justifiable.

Slide 14 — Architecture temps réel
  Schéma : Producer → Redpanda (transactions / transactions.dlq) → Consumer → API FastAPI
  (MLflow Registry, alias Production) → SQLite → Dashboard Streamlit.
  Bandeau : 6 services Docker Compose avec leurs ports.

Slide 15 — API et résilience
  Gauche : les 4 routes de l'API, validation Pydantic (422), métriques Prometheus, fallback MLflow.
  Droite : idempotence (PRIMARY KEY), réessais exponentiels (5 tentatives), DLQ, commit manuel des offsets.
  Bas : performance — inférence < 5 ms, médiane 3,4 ms, p95 < 8,2 ms sous 50 utilisateurs Locust (objectif < 100 ms).

Slide 16 — Drift et dashboard
  Formule TVD, seuil 0,20. Comparaison : normal (max 0,140, pas d'alerte) vs démo (V4 = 0,529, alerte).
  Les 4 onglets du dashboard FraudOps Sentinel (une ligne chacun). Emplacement pour capture d'écran.

Slide 17 — Qualité, CI/CD et organisation
  17/17 tests (répartition par fichier), GitHub Actions (black, ruff, pytest, build Docker), DVC, MLflow, Makefile.
  Tableau en 2 colonnes : rôles d'Anas et de Medo. Planning en 4 semaines (frise).

Slide 18 — Conclusion, limites et perspectives (puis place à la démo)
  3 résultats clés, écarts honnêtes (Grafana non déployé, drift en TVD maison, cibles make demo absentes),
  perspectives (Feast, réentraînement déclenché par le drift, Kubernetes, Grafana).
  Encadré « Démo en direct » avec les 7 étapes, puis « Merci — Questions ? ».

SORTIE ATTENDUE
Pour chaque slide : titre, contenu à l'écran, suggestion de visuel, notes orateur (avec le présentateur).
```

---

## Répartition orale suggérée

| Slides | Présentateur | Durée |
|---|---|---|
| 1–3 | Anas et Medo (intro à deux) | 2 min |
| 4–13 | Anas (données, modèles, coût, SHAP) | 7 min |
| 14–17 | Medo (architecture, API, drift, CI) | 5 min |
| 18 + démo | Medo pilote la démo, Anas commente SHAP | 1 min + 5–7 min |

## Captures à préparer pour les slides

- Dashboard : onglet Live Stream avec une ligne FRAUD visible (slide 16).
- Onglet SHAP d'une fraude (slide 13 ou 16).
- MLflow : liste des runs et modèle `fraud-detector` en Production (slide 9 ou 17).
- Swagger `/docs` avec une réponse `/predict` (slide 15).
- `report_assets/fig1_benchmark.png` (slide 9) et `report_assets/fig2_shap_drift.png` (slides 13 et 16) existent déjà.
