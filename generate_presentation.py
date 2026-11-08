"""Script de génération automatique d'une présentation PowerPoint professionnelle
sur le projet de Détection de Fraude Bancaire (MLOps & Cost Optimization).
"""

import sys
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path: str = "Presentation_Detection_Fraude.pptx"):
    prs = Presentation()
    # Format 16:9 écran large
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Palette de couleurs Fintech Premium
    DARK_BG = RGBColor(15, 23, 42)       # Slate 900
    CARD_BG = RGBColor(30, 41, 59)       # Slate 800
    CARD_BORDER = RGBColor(51, 65, 85)   # Slate 700
    TEXT_LIGHT = RGBColor(248, 250, 252) # Slate 50
    TEXT_MUTED = RGBColor(148, 163, 184) # Slate 400
    ACCENT_BLUE = RGBColor(59, 130, 246) # Blue 500
    ACCENT_GREEN = RGBColor(16, 185, 129)# Emerald 500
    ACCENT_RED = RGBColor(239, 68, 68)   # Red 500
    ACCENT_YELLOW = RGBColor(245, 158, 11)# Amber 500
    CARD_HIGHLIGHT = RGBColor(30, 58, 138)# Dark Blue Accent

    def set_slide_background(slide, color=DARK_BG):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, title: str, category: str = "PROJET MLOPS - FRAUD DETECTION"):
        # Category Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.8))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_LIGHT

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()
        return card

    # ==========================================
    # SLIDE 1 : TITRE
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Accent decorative bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.15), Inches(3.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_BLUE
    bar.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11), Inches(3.8))
    tf = t_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "SYSTÈME INDUSTRIEL DE DÉTECTION DE FRAUDE BANCAIRE"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_BLUE

    p1 = tf.add_paragraph()
    p1.text = "Pipeline MLOps, Optimisation des Coûts Métier\net Gouvernance en Temps Réel"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT
    p1.space_before = Pt(14)

    p2 = tf.add_paragraph()
    p2.text = "Dataset ULB European Cardholders • Scikit-learn & LightGBM • SHAP Explainability • FastApi & Kafka"
    p2.font.size = Pt(15)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(20)

    # Footer cards on Slide 1
    features_summary = [
        ("0.17 %", "Taux de fraude réel"),
        ("DSP2 & Coûts", "Matrice asymétrique"),
        ("< 5 ms", "Latence d'inférence API"),
        ("100 %", "Traçabilité MLOps (MLflow)")
    ]
    for i, (metric, label) in enumerate(features_summary):
        c_left = Inches(1.2 + i * 2.8)
        card = add_card(s1, c_left, Inches(5.6), Inches(2.5), Inches(1.2))
        c_box = s1.shapes.add_textbox(c_left, Inches(5.65), Inches(2.5), Inches(1.1))
        c_tf = c_box.text_frame
        c_p1 = c_tf.paragraphs[0]
        c_p1.text = metric
        c_p1.font.size = Pt(20)
        c_p1.font.bold = True
        c_p1.font.color.rgb = ACCENT_GREEN if i in (0, 2) else ACCENT_BLUE
        c_p1.alignment = PP_ALIGN.CENTER
        c_p2 = c_tf.add_paragraph()
        c_p2.text = label
        c_p2.font.size = Pt(11)
        c_p2.font.color.rgb = TEXT_MUTED
        c_p2.alignment = PP_ALIGN.CENTER

    # ==========================================
    # SLIDE 2 : PROBLÉMATIQUE MÉTIER & COÛTS
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Problématique Métier : Pourquoi l'Accuracy est un Piège Mortel")

    # 3 Cards
    # Card 1: Paradoxe de l'accuracy
    add_card(s2, Inches(0.8), Inches(1.8), Inches(3.7), Inches(4.9))
    b1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(3.3), Inches(4.5))
    t1 = b1.text_frame
    t1.word_wrap = True
    p = t1.paragraphs[0]
    p.text = "1. Le Piège de l'Accuracy"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_YELLOW
    
    bullets1 = [
        ("Déséquilibre extrême :", "492 fraudes sur 284 807 transactions (0.17 %)."),
        ("Modèle 'zéro effort' :", "Un modèle prédisant TOUJOURS 'légitime' atteint 99.83 % d'accuracy !"),
        ("Conséquence bancaire :", "100 % des fraudes passent inaperçues, entraînant des faillites et sanctions."),
        ("Verdict :", "L'accuracy et le ROC-AUC standard sont inopérants. Seuls le Recall et la PR-AUC importent.")
    ]
    for title, desc in bullets1:
        p_t = t1.add_paragraph(); p_t.space_before = Pt(8)
        p_t.text = f"• {title} " ; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = desc; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # Card 2: Asymétrie des coûts
    add_card(s2, Inches(4.8), Inches(1.8), Inches(3.7), Inches(4.9))
    b2 = s2.shapes.add_textbox(Inches(5.0), Inches(2.0), Inches(3.3), Inches(4.5))
    t2 = b2.text_frame
    t2.word_wrap = True
    p = t2.paragraphs[0]
    p.text = "2. La Matrice des Coûts Métier"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_RED

    bullets2 = [
        ("Faux Négatif (FN) :", "Perte sèche = Montant total (DSP2 oblige le remboursement intégral de la victime)."),
        ("Faux Positif (FP) :", "Coût fixe de 5 € (support client, friction, appel de déblocage)."),
        ("Vrai Positif (TP) :", "Coût fixe de 5 € d'investigation (traitement humain, clôture dossier)."),
        ("Vrai Négatif (TN) :", "0 € (transaction fluide).")
    ]
    for title, desc in bullets2:
        p_t = t2.add_paragraph(); p_t.space_before = Pt(8)
        p_t.text = f"• {title} " ; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = desc; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # Card 3: Formule globale
    add_card(s2, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.9))
    b3 = s2.shapes.add_textbox(Inches(9.0), Inches(2.0), Inches(3.3), Inches(4.5))
    t3 = b3.text_frame
    t3.word_wrap = True
    p = t3.paragraphs[0]
    p.text = "3. Objectif d'Optimisation"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_GREEN

    bullets3 = [
        ("Fonction de perte financière :", "Coût Total = ∑ (Montant FN) + 5€ × (|FP| + |TP|)"),
        ("Arbitrage économique :", "Bloquer 500 € coûte 5 € (Gain net: +495 €). Mais bloquer 2 € coûte 5 € (Perte: -3 €)."),
        ("Objectif algorithmique :", "Minimiser le coût en euros réels en trouvant le seuil de décision mathématiquement optimal θ*.")
    ]
    for title, desc in bullets3:
        p_t = t3.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {title} " ; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = desc; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # ==========================================
    # SLIDE 3 : LE DATASET & TEMPORAL SPLIT
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Architecture des Données : Dataset ULB & Split Temporel")

    # Card Left : Le Dataset et PCA V1-V28
    add_card(s3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.9))
    bl = s3.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    tl = bl.text_frame
    tl.word_wrap = True
    p = tl.paragraphs[0]
    p.text = "Origine & Anonymisation (V1 à V28)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE

    items_l = [
        ("Origine académique & bancaire :", "ULB (Université Libre de Bruxelles) & Worldline. 284 807 transactions en Europe (2 jours de septembre 2013)."),
        ("Secret bancaire & RGPD :", "Impossible de divulguer numéros de carte (PAN), noms, marchands, GPS ou soldes."),
        ("Solution PCA (ACP) :", "Les variables V1 à V28 sont les composantes principales orthogonales issues de l'ACP appliquée sur les données sensibles."),
        ("3 Variables brutes conservées :", "Time (secondes écoulées), Amount (montant de l'achat) et Class (0 = légitime, 1 = fraude).")
    ]
    for t, d in items_l:
        p_t = tl.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # Card Right : Temporal Split
    add_card(s3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    br = s3.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tr = br.text_frame
    tr.word_wrap = True
    p = tr.paragraphs[0]
    p.text = "Découpage Temporel Strict (Zero Data Leakage)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_GREEN

    items_r = [
        ("Le piège du split aléatoire :", "Un train_test_split classique mélange passé et futur, provoquant une fuite d'information (lookahead bias)."),
        ("Règle temporelle stricte :", "Tri rigoureux par Time avant toute opération :"),
        ("  - Train (70 %) :", "199 364 transactions (~34 premières heures). 344 fraudes (0.173 %)."),
        ("  - Validation (15 %) :", "42 721 transactions (~7 heures suivantes). Optimisation des seuils."),
        ("  - Test (15 %) :", "42 722 transactions (dernières heures). Évaluation finale intègre.")
    ]
    for t, d in items_r:
        p_t = tr.add_paragraph(); p_t.space_before = Pt(8)
        p_t.text = f"{t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # ==========================================
    # SLIDE 4 : FEATURE ENGINEERING
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Feature Engineering : FraudFeatureEngineer (Scikit-Learn)")

    # 2 Cards for Hour and Log_Amount
    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.9))
    bh = s4.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    th = bh.text_frame; th.word_wrap = True
    p = th.paragraphs[0]; p.text = "Cycle Circadien : Hour = (Time // 3600) % 24"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_YELLOW

    bullets_h = [
        ("Why (Pourquoi ?) :", "Le temps brut 'Time' croît de 0 à 172 800 s sans cyclicité."),
        ("Psychologie de la fraude :", "Les attaques de bots et cartes volées surviennent la nuit (1h-5h du matin) quand les victimes dorment et le support bancaire est réduit."),
        ("Invariance jour 1 / jour 2 :", "Une transaction à 3h du matin le jour 1 et le jour 2 partagent le même risque nocturne."),
        ("How (Comment ?) :", "Conversion des secondes en heures (/ 3600), puis modulo 24 pour replier sur la journée [0h à 23h].")
    ]
    for t, d in bullets_h:
        p_t = th.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    add_card(s4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    ba = s4.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    ta = ba.text_frame; ta.word_wrap = True
    p = ta.paragraphs[0]; p.text = "Stabilisation Variance : log_amount = log1p(Amount)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE

    bullets_a = [
        ("Why (Pourquoi ?) :", "Distribution des montants hautement asymétrique (loi de puissance / Pareto). 99 % des transactions < 100 €, mais quelques-unes > 5 000 €."),
        ("Danger des valeurs extrêmes :", "Distorsion des gradients, explosion des poids de régression, écrasement des distances (K-Means/PCA)."),
        ("Pourquoi log1p et non log ? :", "Certaines autorisations de carte sont à 0.00 €. log(0) = -∞ (génère NaN et crash). log1p(x) = ln(1+x), donc log1p(0) = 0.0 sans discontinuité."),
        ("Intégration Scikit-Learn :", "Implémenté comme BaseEstimator & TransformerMixin pour prévenir tout data leakage.")
    ]
    for t, d in bullets_a:
        p_t = ta.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # ==========================================
    # SLIDE 5 : STRATÉGIES DE DÉSÉQUILIBRE
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Stratégies Face au Déséquilibre : 4 Approches Comparées")

    strat_cards = [
        ("1. class_weight='balanced'", "Pondération Algorithmique", 
         "Ne modifie pas les données. Modifie la loss function en pénalisant ~290x plus les erreurs sur les fraudes. Idéal pour Random Forest."),
        ("2. Random Under-Sampling", "Sous-échantillonnage Majoritaire", 
         "Réduit aléatoirement les transactions légitimes sur le train set pour amener la fraude à 10 %. Entraînement ultra-rapide et net abaissement des coûts."),
        ("3. SMOTE (ratio 0.1)", "Sur-échantillonnage Synthétique", 
         "Génère de nouvelles fraudes plausibles par interpolation k-NN entre fraudes existantes. Densifie la frontière de décision."),
        ("4. scale_pos_weight brut", "Le Piège du Gradient Boosting", 
         "Multiplier par 577 les gradients sur LightGBM crée une panique algorithmique : explosion des faux positifs et perte financière de 9 315 € !")
    ]

    for i, (title, subtitle, desc) in enumerate(strat_cards):
        col = i % 2
        row = i // 2
        left = Inches(0.8 + col * 6.0)
        top = Inches(1.8 + row * 2.5)
        border = ACCENT_RED if i == 3 else (ACCENT_GREEN if i in (0, 1) else CARD_BORDER)
        add_card(s5, left, top, Inches(5.7), Inches(2.3), border_color=border)
        
        tb = s5.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), Inches(5.3), Inches(2.0))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = title
        p1.font.size = Pt(16); p1.font.bold = True
        p1.font.color.rgb = ACCENT_RED if i == 3 else (ACCENT_GREEN if i in (0, 1) else ACCENT_BLUE)
        
        p2 = tf.add_paragraph(); p2.text = subtitle.upper(); p2.font.size = Pt(10); p2.font.color.rgb = TEXT_MUTED
        
        p3 = tf.add_paragraph(); p3.text = desc; p3.space_before = Pt(8)
        p3.font.size = Pt(12); p3.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 6 : BENCHMARK COMPARATIF (TABLEAU)
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Benchmark Modèles & Validation des Performances")

    # Table creation
    rows = 7
    cols = 8
    t_left = Inches(0.8)
    t_top = Inches(1.8)
    t_width = Inches(11.7)
    t_height = Inches(4.8)

    table_shape = s6.shapes.add_table(rows, cols, t_left, t_top, t_width, t_height)
    table = table_shape.table

    headers = ["Modèle", "Stratégie", "PR-AUC", "ROC-AUC", "Recall", "Precision", "Coût Total", "Seuil θ*"]
    col_widths = [Inches(1.8), Inches(2.3), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.2), Inches(1.6), Inches(1.1)]
    for idx, width in enumerate(col_widths):
        table.columns[idx].width = width

    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid(); cell.fill.fore_color.rgb = CARD_HIGHLIGHT
        p = cell.text_frame.paragraphs[0]
        p.text = h; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = TEXT_LIGHT
        p.alignment = PP_ALIGN.CENTER

    data = [
        ["Random Forest", "class_weight='balanced'", "0.8595", "0.9851", "78.57 %", "100.00 %", "623.56 €", "0.4631"],
        ["LightGBM", "Random Under-Sampling", "0.8477", "0.9868", "82.14 %", "85.19 %", "435.30 €", "0.9851"],
        ["Logistic Regr.", "class_weight='balanced'", "0.8456", "0.9823", "87.50 %", "44.95 %", "706.55 €", "0.9900"],
        ["LightGBM", "SMOTE (ratio 0.1)", "0.8311", "0.9882", "80.36 %", "83.33 %", "436.30 €", "0.6945"],
        ["LightGBM", "Optuna Tuned", "0.8070", "0.9830", "78.57 %", "84.62 %", "750.89 €", "0.5468"],
        ["LightGBM", "scale_pos_weight brut", "0.0294", "0.8796", "83.93 %", "2.56 %", "9 315.91 €", "0.9900"],
    ]

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            # Highlight best and worst
            if i == 0 and j == 0:
                cell.fill.fore_color.rgb = CARD_BG
            elif i == 1 and j == 6: # Best cost
                cell.fill.fore_color.rgb = RGBColor(6, 78, 59)
            elif i == 5: # Worst row
                cell.fill.fore_color.rgb = RGBColor(69, 10, 10) if j in (2, 5, 6) else CARD_BG
            else:
                cell.fill.fore_color.rgb = CARD_BG
            
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11)
            p.font.color.rgb = ACCENT_GREEN if (i == 1 and j == 6) or (i == 0 and j in (2, 5)) else (ACCENT_RED if i == 5 and j in (2, 5, 6) else TEXT_LIGHT)
            if j >= 2:
                p.alignment = PP_ALIGN.RIGHT
            else:
                p.alignment = PP_ALIGN.LEFT

    # ==========================================
    # SLIDE 7 : OPTIMISATION DU SEUIL θ*
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Optimisation du Seuil Décisionnel : find_optimal_cost_threshold")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.9))
    bl = s7.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    tl = bl.text_frame; tl.word_wrap = True
    p = tl.paragraphs[0]; p.text = "Pourquoi Abandonner le Seuil Par Défaut 0.5 ?"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_YELLOW

    b_thresh = [
        ("Le seuil par défaut p = 0.5 :", "Est conçu pour une classification symétrique avec coûts d'erreurs équivalents."),
        ("Seuil trop haut (p = 0.8) :", "Presque zéro faux positif (faible coût support), mais explosion des fraudes manquées à plusieurs milliers d'euros."),
        ("Seuil trop bas (p = 0.05) :", "Toutes les fraudes sont interceptées, mais l'équipe fraude est submergée par 5 € par alerte inutile."),
        ("Scanning numérique :", "Le module teste 200 seuils sur le set de validation et extrait le minimum global de coût.")
    ]
    for t, d in b_thresh:
        p_t = tl.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # Card Right: Production Deployment of Threshold
    add_card(s7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    br = s7.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tr = br.text_frame; tr.word_wrap = True
    p = tr.paragraphs[0]; p.text = "Industrialisation du Seuil Opérationnel"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_GREEN

    b_prod = [
        ("Sauvegarde découplée :", "Exporté dans 'threshold.json' avec métriques d'audit associées."),
        ("Injection automatique :", "Rechargé dynamiquement par l'API FastAPI au démarrage sans modifier le modèle."),
        ("Seuil retenu pour le déploiement :", "θ* = 0.4631 (Random Forest), garantissant 100 % de précision en validation et 0 fausse alerte."),
        ("Option économique agressive :", "θ* = 0.9851 (LightGBM Under-sampling) permet de réduire la facture totale à 435 €.")
    ]
    for t, d in b_prod:
        p_t = tr.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # ==========================================
    # SLIDE 8 : EXPLICABILITÉ SHAP
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Explicabilité SHAP : Audit Réglementaire & Analyse Locale")

    # 3 Cards
    c1 = add_card(s8, Inches(0.8), Inches(1.8), Inches(3.7), Inches(4.9))
    tb1 = s8.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(3.3), Inches(4.5))
    t1 = tb1.text_frame; t1.word_wrap = True
    p = t1.paragraphs[0]; p.text = "1. Théorie des Jeux"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE

    sh1 = [
        ("Valeurs de Shapley :", "Attribution juste de la contribution de chaque variable à la probabilité finale."),
        ("Formule additive :", "Prédiction = Base Value (0.17%) + ∑ (SHAP value_i)."),
        ("TreeExplainer :", "Algorithme optimisé en temps polynomial O(T L D²) pour parcourir les arbres de décision instantanément.")
    ]
    for t, d in sh1:
        p_t = t1.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    c2 = add_card(s8, Inches(4.8), Inches(1.8), Inches(3.7), Inches(4.9))
    tb2 = s8.shapes.add_textbox(Inches(5.0), Inches(2.0), Inches(3.3), Inches(4.5))
    t2 = tb2.text_frame; t2.word_wrap = True
    p = t2.paragraphs[0]; p.text = "2. Top Variables Globales"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_GREEN

    sh2 = [
        ("1. V14 (Impact 0.086) :", "Signature majeure d'anomalie de transaction bancaire."),
        ("2. V12 (Impact 0.078) :", "Forte déviation du comportement du compte."),
        ("3. V4 (Impact 0.062) :", "Indicateur critique de risque transactionnel."),
        ("4. V10 & V11 :", "Confirmation de l'état de compromission de la carte.")
    ]
    for t, d in sh2:
        p_t = t2.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    c3 = add_card(s8, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.9))
    tb3 = s8.shapes.add_textbox(Inches(9.0), Inches(2.0), Inches(3.3), Inches(4.5))
    t3 = tb3.text_frame; t3.word_wrap = True
    p = t3.paragraphs[0]; p.text = "3. Audit Local (RGPD)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_YELLOW

    sh3 = [
        ("Droit à l'explication :", "Le RGPD impose d'expliquer pourquoi une transaction a été bloquée."),
        ("Fiche d'incident unitaire :", "Pour chaque refus, extraction des 5 variables ayant fait basculer le score."),
        ("Audit des cas critiques :", "Analyse automatique des TP (fraudes interceptées), FN (fraudes manquées) et FP (fausses alertes).")
    ]
    for t, d in sh3:
        p_t = t3.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # ==========================================
    # SLIDE 9 : MONITORING DU DRIFT
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Gouvernance MLOps : Détection de Dérive (Data & Concept Drift)")

    add_card(s9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.9))
    b_dl = s9.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    tdl = b_dl.text_frame; tdl.word_wrap = True
    p = tdl.paragraphs[0]; p.text = "Surveillance Statistique des Distributions"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE

    b_drift1 = [
        ("Pourquoi surveiller les inputs X ? :", "Les réclamations de fraude mettent 30 à 60 jours à arriver. Le drift d'entrée est l'unique indicateur d'alerte immédiat."),
        ("Test à deux échantillons :", "Compare data/train.csv (référence historique) avec le flux récent de production (data/test.csv)."),
        ("Métrique de divergence :", "Distance de Variation Totale (TVD) sur histogrammes normalisés pour chacune des 29 variables numériques."),
        ("Seuil critique de tolérance :", "Seuil fixé à 0.20 : en deçà = bruit normal ; au-delà = alerte de dérive.")
    ]
    for t, d in b_drift1:
        p_t = tdl.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    add_card(s9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    b_dr = s9.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tdr = b_dr.text_frame; tdr.word_wrap = True
    p = tdr.paragraphs[0]; p.text = "Boucle de Réaction Industrielle (CI/CD)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_RED

    b_drift2 = [
        ("Preuve par le test (DEMO_DRIFT) :", "L'injection d'un décalage artificiel fait bondir le score de V4 à 0.529 (> 0.20) et déclenche l'alarme."),
        ("Génération du rapport :", "data/drift_report.json expose le statut exact et les scores variable par variable."),
        ("Plan de remédiation automatique :", "1. Notification sur Dashboard et alerte MLOps.\n2. Fallback de sécurité (règles conservatrices).\n3. Déclenchement automatique du pipeline de réentraînement sur fenêtre glissante.")
    ]
    for t, d in b_drift2:
        p_t = tdr.add_paragraph(); p_t.space_before = Pt(10)
        p_t.text = f"• {t} "; p_t.font.bold = True; p_t.font.size = Pt(12); p_t.font.color.rgb = TEXT_LIGHT
        run = p_t.add_run(); run.text = d; run.font.bold = False; run.font.color.rgb = TEXT_MUTED

    # ==========================================
    # SLIDE 10 : ARCHITECTURE TECHNIQUE & DEPLOYMENT
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Architecture Technique & Déploiement Temps Réel")

    arch_cards = [
        ("FastAPI Serving", "Latence Inférence < 5 ms", 
         "API REST haute performance. Endpoints /predict et /health. Injection de dépendance du modèle Scikit-learn sérialisé et du seuil optimal."),
        ("Streaming Kafka & Simulateur", "Ingestion Événementielle", 
         "Producer de transactions temps réel, simulation d'attaques et de drift. Consumer asynchrone pour scoring en continu."),
        ("Registre MLflow & DVC", "Reproductibilité Totale", 
         "Versioning des datasets massifs via DVC. Tracking de chaque run (PR-AUC, ROC-AUC, matrice des coûts, paramètres, artefacts)."),
        ("Dashboard de Supervision", "Observabilité 360°", 
         "Interface d'analyse temps réel : flux en direct, métriques système, explications SHAP interactives et statut de dérive.")
    ]

    for i, (title, subtitle, desc) in enumerate(arch_cards):
        col = i % 2
        row = i // 2
        left = Inches(0.8 + col * 6.0)
        top = Inches(1.8 + row * 2.5)
        add_card(s10, left, top, Inches(5.7), Inches(2.3))
        
        tb = s10.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), Inches(5.3), Inches(2.0))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = title
        p1.font.size = Pt(16); p1.font.bold = True; p1.font.color.rgb = ACCENT_BLUE
        
        p2 = tf.add_paragraph(); p2.text = subtitle.upper(); p2.font.size = Pt(10); p2.font.color.rgb = ACCENT_GREEN
        
        p3 = tf.add_paragraph(); p3.text = desc; p3.space_before = Pt(8)
        p3.font.size = Pt(12); p3.font.color.rgb = TEXT_LIGHT

    prs.save(output_path)
    print(f"Presentation generated successfully at: {output_path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Presentation_Detection_Fraude.pptx"
    create_deck(out)
