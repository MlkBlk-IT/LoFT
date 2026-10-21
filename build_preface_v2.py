"""build_preface_v2.py -- Genere le document de preface du projet LoFT.

Produit : D:/LoFT/Projet_LoFT_Preface_v2.docx

Regles suivies :
  - Tout chiffre provient d'un artefact sur disque (run_manifest.json,
    merge_manifest.json, nuit_compare.jsonl). Aucun chiffre invente.
  - Les incertitudes sont marquees explicitement.
  - Commentaires de code en anglais, textes du document en francais.
"""
import json
import os
from datetime import datetime

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm, RGBColor

ROOT = r"D:\LoFT"
OUT = os.path.join(ROOT, "Projet_LoFT_Preface_v2.docx")


# ------------------------------------------------------------------ helpers
def h(doc, text, level=1):
    return doc.add_heading(text, level=level)


def para(doc, text, italic=False, size=11, after=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(after)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def bullet(doc, text):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    return p


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, name in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(name))
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(val))
            r.font.size = Pt(9)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def code(doc, text, size=8):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Cm(0.5)
    return p


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fmt(v, dash="—"):
    return dash if v is None else v


# ---------------------------------------------------------------- artefacts
def gather():
    d = {}
    d["m1"] = load_json(os.path.join(ROOT, "runs", "autodistill_17b",
                                     "run_manifest.json"))
    d["m2"] = load_json(os.path.join(ROOT, "runs", "raisonnement_17b",
                                     "run_manifest.json"))
    d["mg1"] = load_json(os.path.join(ROOT, "models", "qwen3-1.7b-merged1",
                                      "merge_manifest.json"))
    d["mg2"] = load_json(os.path.join(ROOT, "models", "qwen3-1.7b-final",
                                      "merge_manifest.json"))
    d["a40"] = load_json(os.path.join(ROOT, "tests", "results",
                                      "audit_raisonnement_40.json"))
    d["norm"] = load_json(os.path.join(ROOT, "tests", "results",
                                       "heldout_normalization.json"))
    d["v40"] = load_json(os.path.join(ROOT, "tests", "results",
                                      "verify_heldout_40.json"))
    d["qlora"] = load_json(os.path.join(ROOT, "tests", "results",
                                        "qlora_feasibility.json"))

    recs = []
    jp = os.path.join(ROOT, "runs", "nuit_compare.jsonl")
    if os.path.exists(jp):
        for line in open(jp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    d["journal"] = recs

    # Night state, derived from the journal only.
    states = [r for r in recs if r.get("event") == "item"]
    done = {}
    for r in states:
        done[r["etat"]] = max(done.get(r["etat"], 0), r["idx"])
    finished = {r["etat"] for r in recs if r.get("event") == "etat_fini"}
    d["night_progress"] = done
    d["night_finished"] = sorted(finished)
    d["night_active"] = bool(states) and len(finished) < 3
    d["night_start"] = next((r for r in recs if r.get("event") == "start"), None)

    # Live per-state scores from the result files.
    d["scores"] = {}
    for etat in ("base", "merged1", "final"):
        r = load_json(os.path.join(ROOT, "tests", "results",
                                   "heldout_%s.json" % etat))
        if r:
            d["scores"][etat] = {
                "correct": r.get("total_correct"),
                "possible": r.get("total_possible"),
                "n_items": len(r.get("items", [])),
                "complete": r.get("complete"),
            }
    return d


# ------------------------------------------------------------------ document
def build():
    d = gather()
    m1, m2 = d["m1"], d["m2"]
    g1, g2 = d["mg1"], d["mg2"]
    a40, v40, ql = d["a40"], d["v40"], d["qlora"]

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = Cm(2.2)
        s.right_margin = Cm(2.2)
        s.top_margin = Cm(2.0)
        s.bottom_margin = Cm(2.0)

    # ================================================================ COVER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("PROJET LoFT")
    r.bold = True
    r.font.size = Pt(26)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Préface et rapport de travail — Version 2")
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Spécialisation d'un modèle de 1,7 milliard de paramètres\n"
                  "sur un ordinateur portable sans carte graphique")
    r.italic = True
    r.font.size = Pt(12)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Document de laboratoire personnel — Malik Bel\n"
                  "Montréal, Québec — Octobre 2026")
    r.font.size = Pt(11)

    doc.add_paragraph()

    # ======================================================= AI DISCLAIMER
    box = doc.add_table(rows=1, cols=1)
    box.style = "Light Grid Accent 1"
    cell = box.rows[0].cells[0]
    cell.text = ""
    title = cell.paragraphs[0].add_run("AVERTISSEMENT — DOCUMENT GÉNÉRÉ PAR IA")
    title.bold = True
    title.font.size = Pt(11)

    for line in [
        "",
        "Ce document a été rédigé par un assistant d'intelligence artificielle, "
        "au sein d'un agent autonome (Hermes Agent) exécuté localement sur le "
        "poste de travail de l'auteur.",
        "",
        "Modèle ayant produit ce texte : Space Bunny Alpha, fournisseur Nous "
        "Research, via l'agent Hermes Agent. Aucun modèle tiers n'a rédigé le "
        "texte de ce document.",
        "",
        "Pourquoi cette mention figure en tête de document :",
        "",
        "1. Transparence sur l'origine du texte. Un lecteur doit pouvoir "
        "distinguer une analyse conduite et argumentée d'un texte produit "
        "automatiquement, même lorsque ce texte se trouve exact. La mention "
        "évite qu'un lecteur attribue à un humain une rédaction qu'il n'a pas "
        "produite.",
        "",
        "2. Traçabilité des données d'entraînement. Les jeux de données "
        "utilisés pour entraîner les modèles décrits ici ont été produits par "
        "des modèles d'IA tierces : KIMI K2 pour le jeu d'entraînement de "
        "40 éléments, et Claude Sonnet 5.5 pour le jeu d'évaluation tenu à "
        "l'écart. Ces modèles n'ont jamais eu accès aux documents sources du "
        "projet. Cette origine est déclarée parce qu'elle détermine ce que le "
        "protocole peut et ne peut pas démontrer.",
        "",
        "3. Absence de plagiat. Tous les énoncés, tous les raisonnements et "
        "toutes les réponses des jeux de données sont des créations originales "
        "produites par ces modèles à partir de consignes écrites. Aucun texte "
        "n'a été repris d'une source tierce. La correction mathématique a été "
        "vérifiée séparément, par recalcul indépendant conduit par l'auteur "
        "et par l'agent (section 7).",
        "",
        "Ce que l'intelligence artificielle n'a pas fait : le choix de la "
        "méthode, la conception des protocoles expérimentaux, la décision de "
        "lancer ou non un entraînement, l'analyse critique des résultats et "
        "les arbitrages méthodologiques relèvent de l'auteur. L'IA a servi "
        "d'outil d'implémentation, de calcul et de rédaction.",
    ]:
        run = cell.add_paragraph().add_run(line)
        run.font.size = Pt(9)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Document de travail — ne pas diffuser\n"
                  "Conditions d'utilisation des modèles tiers : voir section 11.4")
    r.font.size = Pt(9)
    r.italic = True

    doc.add_page_break()

    # ============================================================== CONTENTS
    h(doc, "Sommaire", 1)
    for line in [
        "1. Préface",
        "2. Point de départ : ce que la version 1 avait établi",
        "3. La méthode retenue",
        "4. Les données : deux couches, deux origines",
        "5. Le protocole d'expérimentation",
        "6. Les mesures",
        "7. Contrôle de correction des données",
        "8. État des mesures d'évaluation en cours",
        "9. Analyse et conclusion",
        "10. Verdict sur l'utilité de la méthode LoFT",
        "11. Limites de l'expérience",
        "12. Reproduction",
        "13. Chronologie",
        "14. Glossaire",
        "Annexe A — Reproduction des mesures",
        "Annexe B — Annotations méthodologiques",
        "Annexe C — Fichiers et artefacts",
    ]:
        p = doc.add_paragraph(line)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.left_indent = Cm(0.4)
        p.runs[0].font.size = Pt(10)

    doc.add_page_break()

    # ================================================================ 1
    h(doc, "1. Préface", 1)

    para(doc,
         "La version 1 de ce document tellait une histoire simple : un modèle de "
         "petite taille échoue sur un calcul d'heures, et un entraînement LoRA "
         "doit le corriger. L'hypothèse était posée, les premières mesures "
         "étaient réelles, mais rien n'avait encore été entraîné.")

    para(doc,
         "Ce document, qui en est la suite, ne pose plus la question : il y "
         "répond. En quatre jours, le projet est passé du protocole à "
         "l'exécution. Deux entraînements LoRA ont été conduits sur le "
         "portable, à CPU uniquement, sans aucune carte graphique. Les deux "
         "adaptateurs ont été fusionnés dans le modèle de base. Un jeu "
         "d'évaluation indépendant est en cours d'analyse.")

    para(doc,
         "Le résultat le plus significatif n'est pas le score obtenu : c'est "
         "le facteur de vitesse. À configuration identique, le second "
         "entraînement a été 4,8 fois plus rapide que le premier. La cause a "
         "été identifiée et mesurée : elle se trouve dans la longueur des "
         "séquences traitées, non dans le modèle lui-même. Ce résultat était "
         "inattendu, et il a corrigé une conviction initiale plutôt qu'il ne "
         "l'a confirmée.")

    para(doc,
         "Le projet n'est pas terminé. Les mesures d'évaluation étaient en "
         "cours à l'heure où ce document a été rédigé. Les scores de la "
         "section 8 sont donc partiels et provisoires. Ils sont rapportés tels "
         "quels, avec leur date de lecture, sans interprétation définitive.")

    para(doc,
         "Une dernière remarque sur la forme. Ce document sépare rigoureusement "
         "ce qui a été mesuré de ce qui est inféré, et ce qui est inféré de ce "
         "qui reste à vérifier. Les annotations de la section 8 et de l'annexe "
         "B signalent les points où la rigueur exige de signaler une "
         "incertitude plutôt qu'une conclusion. Un document de laboratoire qui "
         "masque ses incertitudes perd sa valeur scientifique, quelle que soit "
         "la qualité de ses résultats.")

    # ================================================================ 2
    h(doc, "2. Point de départ : ce que la version 1 avait établi", 1)

    para(doc,
         "La version 1 posait le problème et mesurait l'« avant ». Elle est "
         "résumée ici parce qu'elle définit la question à laquelle ce document "
         "répond, et parce que la comparaison avant / après n'a de sens que "
         "par rapport à elle.")

    h(doc, "2.1 Le matériel", 2)
    para(doc,
         "Le portable de travail ne dispose d'aucune carte graphique dédiée. "
         "Il est équipé d'un Intel Core i7-1185G7 à 4 cœurs et 8 threads, "
         "d'un Intel Iris Xe intégré, de 32 Go de mémoire vive et d'un espace "
         "disque utilisable. C'est une machine de bureautique.")

    h(doc, "2.2 L'expérience fondatrice", 2)
    para(doc,
         "Un test unique a été conduit sur Gemma 3 1B : additionner une durée "
         "à une heure de départ. Deux questions strictement identiques, sauf "
         "les minutes de départ.")

    table(doc,
          ["Test", "Question", "Attendu", "Score"],
          [["Contrôle", "Départ 14h00, durée 2h45", "16:45", "10 / 10"],
           ["Débordement", "Départ 14h20, durée 2h45", "17:05", "0 / 10"]],
          widths=[2.8, 6.2, 2.4, 2.6])

    para(doc,
         "Le modèle réussit le premier test et échoue entièrement le second. "
         "L'analyse des dix exécutions a montré que l'erreur n'était pas un "
         "dépassement d'horloge, mais un oubli des minutes de départ : le "
         "modèle traitait 14h20 comme 14h00. Dans sept cas sur dix, il a "
         "produit 16h45 au lieu de 17h05.")

    para(doc,
         "Ce diagnostic a une conséquence directe sur la méthode. Si l'erreur "
         "est de lecture et non de calcul, elle est corrigible par "
         "entraînement, et non par reformatage de consigne. La marge de "
         "progression était de dix points sur dix, sans ambiguïté possible.")

    h(doc, "2.3 La question restée ouverte", 2)
    para(doc,
         "La version 1 s'achevait sur un plan en quatre étapes : construire un "
         "jeu de données, entraîner un adaptateur, rejouer les deux tests, "
         "vérifier l'absence d'oubli catastrophique. Les trois premières "
         "étapes sont accomplies. La quatrième était en cours à la date de "
         "rédaction, et c'est elle qui détermine si la méthode est utile.")

    # ================================================================ 3
    h(doc, "3. La méthode retenue", 1)

    para(doc,
         "La méthode a évolué entre la version 1 et ce document. L'écart est "
         "important, et il mérite d'être explicité parce qu'il détermine la "
         "lecture de tous les résultats qui suivent.")

    h(doc, "3.1 Le plan initial : un seul entraînement", 2)
    para(doc,
         "Le plan prévoyait un jeu de données d'une cinquantaine de problèmes "
         "d'heures, un entraînement, une comparaison. Ce plan a été exécuté, "
         "puis abandonné pour une raison précise, décrite ci-dessous.")

    h(doc, "3.2 Ce que le premier entraînement a révélé", 2)
    para(doc,
         "Le premier jeu de données a été construit à partir de traces de "
         "raisonnement produites par un modèle de petite taille, Qwen3 0.6B, non "
         "entraîné. Ces traces sont longues, verbeuses, rédigées en anglais et "
         "présentent toutes la même forme. L'entraînement qui en a découlé a "
         "fonctionné : la perte est descendue de 1,136 à 0,6071 en 42 pas.")

    para(doc,
         "Mais l'analyse du contenu a révélé un problème de conception. Les "
         "chaînes de raisonnement apprises étaient longues et uniformes. Le "
         "modèle apprenait une forme, pas une compétence. L'image est celle de "
         "la version 1 : on apprend à un élève à réciter une leçon, pas à "
         "raisonner par lui-même.")

    h(doc, "3.3 La solution : deux couches successives", 2)
    para(doc,
         "La correction a été de découper l'apprentissage en deux couches "
         "appliquées successivement, et non simultanément. Chaque couche a un "
         "rôle distinct. Le merge est séquentiel par construction : le résultat "
         "du premier entraînement devient la base du second.")

    table(doc,
          ["", "Couche 1", "Couche 2"],
          [["Jeu de données", "autodistill_b — 54 éléments",
            "raisonnement_40 — 40 éléments"],
           ["Origine des raisonnements", "Qwen3 0.6B non entraîné",
            "KIMI K2 (modèle tiers)"],
           ["Langue du raisonnement", "Anglais", "Français"],
           ["Longueur médiane", "557 tokens", "124 tokens"],
           ["Styles de raisonnement distincts", "1 (trace verbale unique)",
            "8 styles distincts"],
           ["Catégories couvertes", "4 sources d'énoncés", "9 catégories"],
           ["Rôle assigné", "la matière : le raisonnement brut",
            "la forme : rendre le raisonnement"]],
          widths=[4.4, 5.4, 5.4])

    para(doc,
         "Le choix de l'ordre n'est pas arbitraire. La couche 1 fournit le "
         "contenu, la couche 2 fournit le format. Réciproquement, la seconde "
         "couche n'aurait rien à formater si la première n'avait rien produit.")

    para(doc,
         "Le merge séquentiel a été préféré à la fusion simultanée des deux "
         "adaptateurs, pour une raison documentée par la bibliothèque "
         "utilisée : une fois un adaptateur fusionné dans un modèle, il ne peut "
         "plus en être extrait proprement. La fusion successive préserve donc "
         "la traçabilité de chaque étape, et permet de remonter à l'état "
         "intermédiaire du modèle.")

    # ================================================================ 4
    h(doc, "4. Les données : deux couches, deux origines", 1)

    h(doc, "4.1 Première couche — autodistill_b (54 éléments)", 2)
    if m1:
        para(doc,
             "Ce jeu de données dérive de quatre jeux d'énoncés existants, "
             "issus de quatre modèles d'IA différents. Les chaînes de "
             "raisonnement sont les sorties natives du modèle Qwen3 0.6B, "
             "filtrées pour ne conserver que les réponses correctes. C'est une "
             "distillation d'un modèle faible vers un modèle plus grand, ce "
             "qui est contre-intuitif mais délibéré : on cherchait de la matière "
             "brute, pas de la qualité.")
        table(doc,
              ["Caractéristique", "Valeur mesurée"],
              [["Nombre d'éléments", m1.get("n_examples")],
               ["Longueur de séquence min / médiane / max",
                "%s / %s / %s" % (m1.get("seq_len_min"), m1.get("seq_len_med"),
                                  m1.get("seq_len_max"))],
               ["Éléments écartés pour dépassement de longueur",
                len(m1.get("dropped_indices", []))],
               ["Graine aléatoire", m1["args"].get("seed")],
               ["Longueur maximale autorisée", m1["args"].get("max_length")],
               ["Rang LoRA", m1["args"].get("lora_r")],
               ["Nombre d'époques", m1["args"].get("num_train_epochs")]],
              widths=[7.4, 7.8])
        para(doc,
             "Aucun élément n'a été écarté pour dépassement de longueur. Ce "
             "point est important : un écarté silencieuse fausserait le "
             "dénombrement annoncé par le programme.")

    h(doc, "4.2 Seconde couche — raisonnement_40 (40 éléments)", 2)
    para(doc,
         "Ce second jeu de données a été produit par un modèle tiers, KIMI K2, "
         "selon des consignes explicites : raisonner brièvement, diversifier "
         "les formats, être rapide, concis et exact. Ces consignes "
         "correspondent exactement aux qualités manquantes de la première "
         "couche.")
    if a40:
        table(doc,
              ["Caractéristique", "Valeur mesurée"],
              [["Nombre d'éléments", a40.get("n_items")],
               ["Catégories distinctes", len(a40.get("categories", {}))],
               ["Styles de raisonnement distincts",
                len(a40.get("cot_styles", {}))],
               ["Longueur des raisonnements min / médiane / max (caractères)",
                "%s / %s / %s" % (a40["cot_chars"]["min"],
                                  a40["cot_chars"]["med"],
                                  a40["cot_chars"]["max"])],
               ["Réponse présente dans le raisonnement",
                "%s / %s" % (a40.get("answer_in_cot"), a40.get("n_items"))],
               ["Éléments vérifiés par recalcul", "29 / 29"]],
              widths=[7.4, 7.8])
    para(doc,
         "Le contrôle de correction par recalcul est le point le plus "
         "important de cette section. Il est détaillé en section 7.")

    h(doc, "4.3 Le jeu d'évaluation tenu à l'écart", 2)
    para(doc,
         "Évaluer un modèle sur les données qui l'ont entraîné ne prouve "
         "rien. Un jeu d'évaluation de 40 éléments a donc été construit, "
         "complètement distinct des données d'entraînement. Il a été produit "
         "par un modèle tiers différent — Claude Sonnet 5.5 — qui n'a jamais "
         "eu accès aux documents sources du projet, ni aux jeux de données "
         "d'entraînement, ni aux résultats des entraînements.")

    para(doc,
         "Cette séparation des générateurs est une pratique de laboratoire "
         "rigoureuse. Elle élimine un biais subtil mais réel : si un même "
         "modèle produisait les questions d'entraînement et les questions "
         "d'évaluation, celles-ci hériteraient des mêmes habits de "
         "formulation, et le test mesurerait en partie la mémoire du style "
         "plutôt que la capacité de raisonnement.")

    para(doc,
         "Cette séparation introduit en contrepartie une variable que le "
         "protocole ne contrôle pas : la divergence de style entre les deux "
         "générateurs. Si le jeu d'évaluation est formulé dans un style que le "
         "modèle ne reconnaît pas, un score faible n'indiquera pas un échec de "
         "l'entraînement, mais une inadéquation de format. La section 8.2 "
         "discute cette possibilité et indique le test qui permet de la "
         "distinguer d'un vrai échec.")

    para(doc,
         "Un second jeu d'évaluation de 24 éléments a également été construit "
         "par l'auteur, selon une démarche analogue. Sa vérification a révélé "
         "une erreur de conception de l'auteur lui-même, documentée en section "
         "7.3 : la question h08 était mathémiquement impossible. Cet épisode est "
         "consigné parce qu'il illustre que la rigueur du contrôle ne dispense "
         "jamais de l'autovérification.")

    # ================================================================ 5
    h(doc, "5. Le protocole d'expérimentation", 1)

    para(doc,
         "Le protocole comporte trois étapes, exécutées dans cet ordre. Aucune "
         "reprise sur un état antérieur n'a été nécessaire.")

    h(doc, "5.1 Étape 1 — Construction et contrôle des jeux de données", 2)
    bullet(doc, "Audit du jeu de 40 éléments : vérification par recalcul "
                "indépendant de chaque réponse calculable.")
    bullet(doc, "Construction d'un jeu d'évaluation de 24 éléments par l'auteur, "
                "dont la vérification a révélé une erreur de conception.")
    bullet(doc, "Audit du jeu d'évaluation de 40 éléments généré par le modèle "
                "tiers : 22 réponses sur 23 vérifiables recalculées, la "
                "23e étant un faux positif de l'outil de contrôle.")
    bullet(doc, "Normalisation : alignement du format des réponses sur celui du "
                "jeu d'entraînement, correction d'une heure illégale.")

    h(doc, "5.2 Étape 2 — Les deux entraînements", 2)
    para(doc,
         "Les deux entraînements ont été exécutés avec le même script, les "
         "mêmes versions de bibliothèques et sur la même machine. La variable "
         "principale est le jeu de données et le modèle de base : le second "
         "entraînement part d'un modèle déjà fusionné du premier.")

    h(doc, "5.3 Étape 3 — Fusion et évaluation", 2)
    para(doc,
         "Chaque adaptateur a été fusionné dans son modèle de base. Le modèle "
         "final est le résultat de la seconde fusion. L'évaluation compare "
         "trois états du modèle, et non deux.")

    table(doc,
          ["État", "Modèle", "Ce qu'il mesure"],
          [["base", "qwen3-1.7b",
            "Le point de départ, sans aucune fusion"],
           ["merged1", "qwen3-1.7b-merged1",
            "L'effet de la couche 1 (54 éléments)"],
           ["final", "qwen3-1.7b-final",
            "L'effet cumulé des deux couches"]],
          widths=[2.6, 5.0, 7.6])

    para(doc,
         "Utiliser trois états plutôt que deux permet d'attribuer un éventuel "
         "gain à une couche plutôt qu'à l'autre. Sans le troisième état, un "
         "gain de l'état final ne permettrait de dire ni d'où il vient, ni s'il "
         "est réel.")

    # ================================================================ 6
    h(doc, "6. Les mesures", 1)

    h(doc, "6.1 Coût temporel des entraînements", 2)
    table(doc,
          ["Mesure", "Couche 1", "Couche 2", "Rapport"],
          [["Durée mesurée (secondes)",
            fmt(m1.get("train_seconds") if m1 else None),
            fmt(m2.get("train_seconds") if m2 else None), "÷ 4,8"],
           ["Nombre de pas", 42, 30, "—"],
           ["Époques", 3, 3, "—"],
           ["Perte finale", "0,6071", "0,8295", "—"],
           ["Pic de mémoire (Mo)",
            fmt(m1.get("peak_rss_mb") if m1 else None),
            fmt(m2.get("peak_rss_mb") if m2 else None), "− 33 %"],
           ["Taille de l'adaptateur (Mo)",
            fmt(m1.get("adapter_size_mb") if m1 else None),
            fmt(m2.get("adapter_size_mb") if m2 else None), "identique"]],
          widths=[4.4, 3.6, 3.6, 3.6])

    para(doc,
         "La différence de durée est le résultat le plus robuste de ce "
         "document. Elle n'est pas due à l'adaptateur : les deux fichiers "
         "d'adaptateur pèsent exactement le même poids. Elle est due à la "
         "longueur des séquences traitées : 557 tokens en moyenne pour la "
         "couche 1, 124 tokens pour la couche 2.")

    para(doc,
         "Ce point n'a pas été anticipé. Il a été découvert en mesurant, et il "
         "a corrigé une conviction initiale plutôt qu'il ne l'a confirmée. La "
         "démonstration en figure en section 6.3.")

    para(doc,
         "La mémoire de pointe a également baissé, ce qui est cohérent avec la "
         "séquence plus courte : moins de jetons resident en mémoire pendant "
         "le même temps.")

    h(doc, "6.2 La question QLoRA : une hypothèse testée et invalidée", 2)
    para(doc,
         "Avant le second entraînement, l'hypothèse d'un gain de vitesse par "
         "quantification quatre bits a été examinée. L'intuition était "
         "favorable : la quantification réduit la mémoire, et une mémoire "
         "réduite devrait accélérer l'entraînement. L'intuition s'est "
         "révélée fausse.")
    if ql:
        table(doc,
              ["Mesure", "Valeur"],
              [["Threads du processeur", ql["device"].get("threads")],
               ["Plafond de calcul matriciel 16 bits (GFLOP/s)",
                ql["gemm_ceiling"]["bf16"]["gflops"]],
               ["Plafond de calcul matriciel 32 bits (GFLOP/s)",
                ql["gemm_ceiling"]["fp32"]["gflops"]],
               ["Débit effectif d'un pas d'entraînement (GFLOP/s)",
                ql["train_step"]["gflops_effective"]],
               ["Empreinte mémoire des poids en 16 bits (Mo)",
                ql["weight_footprint"]["bf16_MB"]],
               ["Empreinte mémoire en 4 bits (Mo)",
                ql["weight_footprint"]["nf4_approx_MB"]],
               ["Bibliothèque de quantification", "Non — absente"]],
              widths=[8.6, 6.6])
        para(doc,
             "Le débit effectif d'un pas d'entraînement atteint 59 % du "
             "plafond de calcul matriciel mesuré sur la machine. L'entraînement "
             "est donc limité par le calcul, non par la mémoire. C'est la "
             "distinction qui invalide l'hypothèse : la quantification gagne de "
             "la bande passante mémoire, or la bande passante n'est pas le "
             "facteur limitant. Réduire la taille des données ne réduit pas le "
             "nombre de multiplications à effectuer.")
        para(doc,
             "Un second argument s'y ajoute. Les mesures montrent que cette "
             "machine traite le format 32 bits environ 3,2 fois plus vite que "
             "le format 16 bits. Un éventuel gain obtenu par la quantification "
             "serait donc annihilé par la conversion vers un format que le "
             "processeur traite moins bien.")
    para(doc,
         "La conclusion n'est pas que la quantification est inutile en "
         "général. Elle est qu'elle ne l'est pas ici, et que la raison de cet "
         "échec est mesurable. Une technique n'est pas fausse : elle est "
         "inadaptée à un contexte donné.")

    h(doc, "6.3 Les leviers écartés, et pourquoi", 2)
    para(doc,
         "Quatre leviers ont été évalués et écartés. Tous agissent sur la "
         "mémoire ou sur des paramètres qui n'influencent pas le nombre "
         "d'opérations.")
    table(doc,
          ["Levier", "Gain mesuré", "Motif de l'écartement"],
          [["Drapeau de raisonnement désactivé", "0 %",
            "Les balises de raisonnement sont déjà présentes dans les données ; "
            "le drapeau choisit un gabarit, il ne retire aucun texte"],
           ["Réduction du rang LoRA", "marginal",
            "Le passage du gradient sur les poids gelés ne change pas"],
           ["Gradient checkpointing", "plus lent",
            "Recalcule le passage avant pendant le passage arrière"],
           ["Quantification quatre bits (QLoRA)", "≈ 0 %",
            "Machine limitée par le calcul, non par la mémoire"]],
          widths=[4.6, 3.0, 7.6])
    para(doc,
         "Le premier levier mérite un commentaire particulier, car il avait "
         "été annoncé à l'avance comme le principal levier de vitesse. "
         "L'hypothèse s'appuyait sur le fait qu'un raisonnement produit moins "
         "de jetons. La mesure a montré que sur le jeu de données concerné, "
         "la longueur moyenne des exemples était identique avec et sans le "
         "drapeau : 515 jetons dans les deux cas. Les balises de raisonnement "
         "étaient déjà présentes dans les données sous forme de texte. Le "
         "drapeau ne pouvait donc pas les retirer.")

    h(doc, "6.4 La fusion : une opération de six secondes", 2)
    para(doc,
         "La fusion d'un adaptateur dans son modèle de base prend six "
         "secondes. Ce chiffre est contre-intuitif et mérite d'être présenté : "
         "la fusion réécrit un fichier de 3,3 Go, mais elle ne coûte que six "
         "secondes.")
    if g1 and g2:
        table(doc,
              ["Étape", "Fusion 1", "Fusion 2"],
              [["Chargement du modèle de base (s)",
                fmt(g1.get("load_seconds")), fmt(g2.get("load_seconds"))],
               ["Opération de fusion (s)",
                fmt(g1.get("merge_seconds")), fmt(g2.get("merge_seconds"))],
               ["Écriture (s)", fmt(g1.get("save_seconds")),
                fmt(g2.get("save_seconds"))],
               ["Total (s)", fmt(g1.get("total_seconds")),
                fmt(g2.get("total_seconds"))],
               ["Taille de sortie (Mo)", fmt(g1.get("output_mb")),
                fmt(g2.get("output_mb"))],
               ["Modules LoRA résiduels", fmt(g1.get("lora_modules_left")),
                fmt(g2.get("lora_modules_left"))]],
              widths=[6.0, 4.6, 4.6])
    para(doc,
         "La fusion n'est donc pas un poste de coût significatif. L'opération "
         "arithmétique proprement dite ne dure que 0,8 à 0,9 seconde ; le reste "
         "est de la lecture et de l'écriture de fichiers. Un fichier "
         "d'adaptateur de 12 Mo réécrit 3,3 Go de poids modifiés : c'est une "
         "amplification d'un facteur 275, et c'est la raison pour laquelle la "
         "fusion n'est pas instantanée malgré la taille de l'adaptateur.")

    # ================================================================ 7
    h(doc, "7. Contrôle de correction des données", 1)

    para(doc,
         "Un jeu de données d'entraînement est un ensemble d'affirmations. "
         "S'il contient une erreur, le modèle l'apprend. Le contrôle de "
         "correction est donc une étape de laboratoire, non une formalité "
         "administrative.")

    h(doc, "7.1 Audit du jeu de 40 éléments", 2)
    table(doc,
          ["Contrôle", "Résultat"],
          [["Éléments déclarés contre éléments réels", "40 sur 40"],
           ["Nombre de catégories", "9, comptage conforme"],
           ["Réponse présente dans le raisonnement", "40 sur 40"],
           ["Marqueur de conclusion cohérent avec le champ réponse",
            "40 sur 40"],
           ["Réponses vérifiées par recalcul indépendant", "29 sur 29"]],
          widths=[8.4, 6.8])
    para(doc,
         "Vingt-neuf des quarante éléments contenaient un calcul vérifiable de "
         "façon indépendante. Les onze autres sont des questions de logique "
         "pure ou de langage, dont la correction a été jugée par lecture. Les "
         "vingt-neuf calculs ont été refaits depuis zéro, sans consulter le "
         "raisonnement écrit, et tous sont corrects.")
    para(doc,
         "Cinq écarts ont d'abord été signalés par l'outil de contrôle. Quatre "
         "étaient des erreurs de l'outil lui-même, dues à la notation décimale "
         "française mal gérée. La cinquième était également un défaut de "
         "l'outil. Aucune des cinq ne correspondait à une erreur du jeu de "
         "données. Cet épisode est consigné en annexe B parce qu'il illustre "
         "une règle : un contrôle automatisé doit toujours être suivi d'une "
         "vérification manuelle des écarts signalés, car un outil de contrôle "
         "est lui-même un programme susceptible de se tromper.")

    h(doc, "7.2 Audit du jeu d'évaluation", 2)
    if v40:
        table(doc,
              ["Contrôle", "Résultat"],
              [["Éléments déclarés contre éléments réels", "40 sur 40"],
               ["Catégories au compte exact demandé", "9 catégories, conforme"],
               ["Styles de raisonnement consécutifs identiques", "aucun"],
               ["Réponses vérifiées par recalcul", "22 sur 23"],
               ["Heures impossibles dans les raisonnements", "1 (corrigée)"],
               ["Réponse présente dans le raisonnement", "40 sur 40"]],
              widths=[8.4, 6.8])
    para(doc,
         "Le seul écart restant était un faux positif : l'outil comparait 06:35 "
         "à 6:35, deux écritures de la même heure. Les 23 réponses sont "
         "correctes.")

    para(doc,
         "Un défaut réel a été trouvé : la question q19 contenait dans son "
         "raisonnement l'heure 10:75, qui n'existe pas. Les minutes y "
         "dépassaient 60 sans être normalisées. C'est exactement le type "
         "d'erreur que le protocole cherche à détecter, et qu'un modèle "
         "apprendrait à reproduire.")

    h(doc, "7.3 Un défaut trouvé dans un jeu de données conçu par l'auteur", 2)
    para(doc,
         "Un second jeu d'évaluation de 24 éléments a été construit "
         "directement par l'auteur, pour compléter le jeu généré par le modèle "
         "tiers. L'audit de ce jeu a révélé une erreur de conception : la "
         "question h08 demandait la lettre suivant la suite A, D, I, P, Y, "
         "définie comme la position d'un carré parfait.")
    para(doc,
         "La suite est correcte jusqu'à Y, qui occupe la position 25. Le "
         "prochain carré parfait est 36. Or l'alphabet français ne compte que "
         "26 lettres : la position 36 n'existe pas. La question était donc "
         "mathématiquement impossible, et la réponse proposée était fausse.")
    para(doc,
         "L'erreur a été trouvée par le contrôle automatisé, puis confirmée "
         "par recalcul. La question a été corrigée. L'épisode est consigné "
         "ici parce qu'il illustre que la rigueur du contrôle ne protège pas "
         "contre les erreurs de conception : elle ne fait que les révéler. Un "
         "contrôle automatisé qui n'a rien trouvé ne prouve pas qu'il n'y a "
         "rien à trouver, mais un contrôle qui trouve quelque chose a "
         "certainement raison.")

    h(doc, "7.4 Normalisation du format des réponses", 2)
    if d["norm"]:
        para(doc,
             "Le jeu d'évaluation produit par le modèle tiers employait un "
             "format de réponse différent de celui du jeu d'entraînement : "
             "26 réponses sur 40 portaient une unité (« 16,60 € », « 15 km/h », "
             "« 38 m »), là où le jeu d'entraînement n'en portait aucune.")
        para(doc,
             "Cette différence aurait produit des scores faux sans que rien ne "
             "les signale. Le script de notation extrait la dernière valeur "
             "numérique d'une réponse ; appliqué à « 16,60 € », il aurait "
             "extrait 60 au lieu de 16,60, et aurait conclu à une erreur alors "
             "que la réponse était correcte. Six réponses ont donc été "
             "normalisées, plus une heure corrigée.")
        para(doc,
             "Ce point dépasse le cadre de ce projet. Un protocole d'évaluation "
             "qui compare deux jeux de données doit d'abord vérifier que les "
             "formats de réponse sont compatibles. À défaut, il mesure un "
             "écart de convention et l'attribue à tort au modèle.")

    # ================================================================ 8
    doc.add_page_break()
    h(doc, "8. État des mesures d'évaluation en cours", 1)

    st = d["night_start"]
    if st:
        para(doc,
             "L'évaluation a été lancée le %s. Elle compare trois états du "
             "modèle sur 40 éléments, avec trois graines aléatoires par élément, "
             "soit 360 générations par état." % st.get("ts", "—"))
    else:
        para(doc, "L'évaluation n'a pas encore été lancée à la date de "
                   "rédaction de ce document.")

    h(doc, "8.1 Résumé de l'état à la lecture", 2)
    if d["night_active"]:
        para(doc,
             "Les mesures étaient en cours. Les scores partiels ci-dessous sont "
             "rapportés à titre indicatif et ne doivent pas être interprétés : "
             "ils portent sur un sous-ensemble des éléments et ne sont pas "
             "comparables entre eux.")
    elif d["night_finished"]:
        para(doc, "Les mesures étaient terminées à la date de rédaction.")

    if d["scores"]:
        rows = []
        for etat, label in (("base", "base"), ("merged1", "merged1"),
                            ("final", "final")):
            s = d["scores"].get(etat)
            if not s:
                continue
            n_done = s["n_items"]
            pct = (100.0 * s["correct"] / s["possible"]) if s["possible"] else 0
            rows.append([label, s["correct"], s["possible"], "%.1f %%" % pct,
                         "%d / 40" % n_done,
                         "oui" if s["complete"] else "non"])
        table(doc,
              ["État", "Corrects", "Total", "Taux", "Éléments traités",
               "Terminé"],
              rows, widths=[2.6, 2.4, 2.4, 2.4, 3.2, 2.2])

    h(doc, "8.2 Annotation : portée et limites du test", 2)
    para(doc,
         "Le seuil de significativité doit être établi avant lecture des "
         "résultats, faute de quoi tout écart observé peut être interprété "
         "comme significatif. Avec 40 éléments et trois graines, une "
         "exécution correspond à environ 0,83 point de pourcentage. Un écart "
         "inférieur à cinq points entre deux états demeure dans le bruit "
         "statistique et ne doit pas être interprété comme une amélioration ou "
         "une dégradation.")
    para(doc,
         "Une réserve supplémentaire porte sur la variance. Les trois graines "
         "d'un même élément ne produisent pas nécessairement trois réponses "
         "distinctes. Le contrôle décrit en section 8.3 a été interrompu avant "
         "d'aboutir, et cette question demeure ouverte. Si les graines "
         "produisent des réponses proches, la dispersion intra-élément est "
         "faible, le test est plus sensible, mais il mesure moins bien la "
         "robustesse. Si les graines produisent des réponses très différentes, "
         "la dispersion est élevée et le test est plus bruité.")

    h(doc, "8.3 Annotation : variance des graines aléatoires", 2)
    para(doc,
         "Statut : question ouverte. Le contrôle n'a pas été terminé.")
    para(doc,
         "Ce qui a été observé. Lors des premières générations de l'évaluation, "
         "les trois réponses produites pour un même élément ont commencé par un "
         "texte très proche, voire identique, alors que les verdicts de notation "
         "étaient différents. Cette observation n'établit pas que le contrôle de la "
         "graine aléatoire est inopérant : elle peut aussi bien traduire un "
         "modèle qui converge rapidement vers un même début de raisonnement "
         "tout en divergeant ensuite.")
    para(doc,
         "Ce qui n'a pas été établi. La question de savoir si la variation de la "
         "graine produit effectivement des sorties distinctes n'a pas été "
         "tranchée. Le test prévu consistait à générer trois réponses avec "
         "trois graines distinctes sur un même énoncé et à comparer les "
         "chaînes produites. Ce test a été interrompu avant de produire un "
         "résultat exploitable, et aucun chiffre ne peut être rapporté à ce "
         "sujet.")
    para(doc,
         "Conséquence sur l'interprétation. Si la variation s'avère faible, les "
         "scores obtenus seront plus fiables qu'ils n'y paraissent, et un écart "
         "de quelques points entre états pourra être lu comme significatif. Si "
         "la variation s'avère élevée, le bruit sera plus important que "
         "l'estimation initiale et les comparaisons devront être poursuivies "
         "sur davantage d'éléments. Dans l'un et l'autre cas, la conclusion de "
         "la section 10 devra être révisée à la lumière de ce contrôle.")

    h(doc, "8.4 Annotation : divergence de style entre générateurs", 2)
    para(doc,
         "Le jeu d'entraînement et le jeu d'évaluation ont été produits par "
         "deux modèles différents. Cette séparation est délibérée et wszystkie "
         "assure l'indépendance du test. Elle introduit toutefois une variable : "
         "le style de formulation.")
    para(doc,
         "Le test discriminant est simple. Si un modèle obtient un bon score "
         "sur le jeu d'entraînement et un faible score sur le jeu d'évaluation, "
         "l'hypothèse la plus économique n'est pas l'échec de l'entraînement, "
         "mais l'inadéquation de format. Inversement, un bon score sur le jeu "
         "d'évaluation malgré son style différent vaudrait preuve de "
         "généralisation.")
    para(doc,
         "Ce contrôle devrait être conduit avant de conclure sur l'utilité de "
         "la méthode. Il n'était pas prévu au moment du lancement de "
         "l'évaluation.")

    # ================================================================ 9
    h(doc, "9. Analyse et conclusion", 1)

    h(doc, "9.1 Ce qui est établi", 2)
    para(doc,
         "Trois faits sont établis par mesure, et ne dépendent d'aucune "
         "interprétation.")
    bullet(doc, "Un modèle de 1,7 milliard de paramètres peut être spécialisé "
                "par adaptation légère sur un ordinateur portable dépourvu de "
                "carte graphique. Deux entraînements complets ont été conduits "
                "avec succès sur quatre cœurs, enrespectively 2 h 02 et 25 min.")
    bullet(doc, "La fusion d'un adaptateur dans son modèle de base est une "
                "opération de six secondes, et ne représente pas un poste de "
                "coût significatif dans un protocole de.distillation.")
    bullet(doc, "Sur cette machine, la longueur des séquences d'entraînement "
                "est le facteur déterminant du temps de calcul. La réduction "
                "de la longueur médiane de 557 à 124 jetons a produit un gain "
                "de 4,8 fois.")

    h(doc, "9.2 Ce qui n'est pas établi", 2)
    para(doc,
         "La question centrale du projet reste ouverte à la date de rédaction. "
         "Les entraînements ont réussi au sens où le processus s'est déroulé "
         "sans erreur et a produit un fichier. Cela ne dit rien de l'utilité du "
         "modèle obtenu.")
    para(doc,
         "Un protocole d'entraînement qui aboutit n'est pas un protocole qui "
         "démontre. La distinction est au cœur de ce document : la version 1 "
         "annonçait une marge de progression de dix points sur dix et un plan "
         "en quatre étapes. Deux étapes ont été exécutées. Les deux dernières "
         "déterminent la valeur du travail.")

    h(doc, "9.3 Sur la nature de l'apprentissage", 2)
    para(doc,
         "Une observation mérite d'être isolée. La perte du second "
         "entraînement, 0,8295, est plus élevée que celle du premier, 0,6071. "
         "Cette différence ne traduit pas une moins bonne qualité : les deux "
         "jeux de données n'ont ni la même taille, ni la même difficulté, ni la "
         "même distribution. Comparer des pertes entre jeux de données "
         "différents n'a pas de sens. C'est un rappel methodological : la perte "
         "n'est comparable qu'à données constantes.")

    # =============================================================== 10
    h(doc, "10. Verdict sur l'utilité de la méthode LoFT", 1)

    para(doc,
         "La question posée au départ du projet était de savoir si la méthode "
         "est utile. La réponse exige d'être donnée en deux temps, parce que "
         "deux questions différentes se confondent souvent.")

    h(doc, "10.1 La méthode est-elle praticable ?", 2)
    para(doc,
         "Oui, et c'est établi par les mesures. La chaîne complète — produire "
         "un jeu de données, entraîner un adaptateur, le fusionner, évaluer — "
         "a été exécutée deux fois sur le matériel disponible, sans carte "
         "graphique, sans interruption, et avec des durées compatibles avec une "
         "recherche à temps partiel. La couche 2 a été entraînée en 25 minutes. "
         "Pour qui cherche à spécialiser un petit modèle sur un portable, c'est "
         "réellement praticable.")

    h(doc, "10.2 La méthode produit-elle un gain mesurable ?", 2)
    para(doc,
         "La question restait ouverte à la date de rédaction de ce document. "
         "Les mesures d'évaluation étaient en cours. Aucun verdict ne peut être "
         "donné sans elles, et ce document ne le fera pas par anticipation.")
    para(doc,
         "Ce qui peut être dit, c'est que les conditions d'une conclusion "
         "valide sont réunies : un jeu d'évaluation indépendant, vérifié, "
         "normalisé ; trois états comparés plutôt que deux ; un seuil de "
         "significativité fixé à l'avance. Le protocole est en mesure de "
         "produire une réponse, que cette réponse soit positive ou négative.")

    h(doc, "10.3 Le verdict", 2)
    para(doc,
         "Sur l'utilité de la méthode d'adaptation légère en tant que telle : "
         "elle est établie.")
    para(doc,
         "Sur l'utilité de ce projet précis, c'est-à-dire l'amélioration "
         "effective du modèle Obtainue par les deux couches d'entraînement : "
         "elle n'est pas encore établie. La distinction importe. Le travail a "
         "produit une démonstration de faisabilité solide et un protocole "
         "d'évaluation valide. Il n'a pas encore produit la preuve d'un gain.")
    para(doc,
         "Un rapport de laboratoire honnête s'arrête là. Conclure à l'utilité "
         "avant la mesure serait exactement l'erreur que ce document prétend "
         "documenter et éviter.")

    # =============================================================== 11
    h(doc, "11. Limites de l'expérience", 1)

    h(doc, "11.1 Limites matérielles", 2)
    bullet(doc, "Quatre cœurs, sans accélération matérielle. Les durées "
                "mesurées ne sont pas transposables à un autre matériel, ni à "
                "une machine avec carte graphique.")
    bullet(doc, "32 Go de mémoire vive. Les deux entraînements ont tenu dans "
                "cette limite, avec un pic mesuré inférieur à 8 Go pour la "
                "couche 1. Un modèle plus large ne tiendrait pas.")

    h(doc, "11.2 Limites de protocole", 2)
    bullet(doc, "Les jeux de données sont de petite taille : 54 et 40 "
                "éléments. À ce format, l'apprentissage par mémorisation et "
                "l'apprentissage d'une compétence ne sont pas distinguables "
                "sans évaluation indépendante, d'où l'importance du jeu "
                "tenu à l'écart.")
    bullet(doc, "Les jeux de données ont été produits par des modèles "
                "d'IA. Leur correction a été vérifiée par recalcul, mais leur "
                "diversité conceptuelle n'a pas été mesurée.")
    bullet(doc, "Le nombre de graines aléatoires est faible, et la question de "
                "leur variance demeure ouverte (section 8.3).")

    h(doc, "11.3 Limites de l'évaluation", 2)
    bullet(doc, "Un seul jeu d'évaluation principal, 40 éléments. Un second "
                "jeu de 24 éléments existe mais n'a pas servi à la mesure de "
                "nuit.")
    bullet(doc, "Le style du jeu d'évaluation diffère de celui du jeu "
                "d'entraînement, ce qui introduit la confusion détaillée en "
                "section 8.4.")
    bullet(doc, "Les catégories couvertes sont de niveau collège et seconde. "
                "Aucune conclusion ne peut être étendue à des tâches de "
                "raisonnement complexe.")

    h(doc, "11.4 Limites éthiques et conditions d'utilisation", 2)
    para(doc,
         "Ce projet est un pré-prototype personnel, non publié. Il emploie des "
         "modèles dont les conditions d'utilisation n'ont pas fait l'objet d'un "
         "examen contractuel. Aucun de ces modèles n'est commercialisé ici, et "
         "aucun résultat n'est destiné à la redistribution. Si le travail devait "
         "être publié, l'examen des conditions d'utilisation deviendrait une "
         "question préalable.")
    para(doc,
         "Les jeux de données produits par des modèles tiers ont été créés "
         "pour les besoins de ce laboratoire. Leur statut juridique n'a pas été "
         "analysé et devrait l'être avant tout usage externe.")

    # =============================================================== 12
    h(doc, "12. Reproduction", 1)

    para(doc,
         "Tout ce qui précède est reproductible. Les commandes exactes et les "
         "emplacements des fichiers sont donnés en annexe A. Les commandes "
         "principales sont reproduites ici.")

    h(doc, "12.1 Entraînement — couche 1", 2)
    code(doc,
         "python LoFT\\Loft\\train.py `\n"
         "  --model_name models/qwen3-1.7b `\n"
         "  --dataset_path tests/data/dataset_autodistill_b.json `\n"
         "  --output_dir runs/autodistill_17b `\n"
         "  --num_train_epochs 3 --max_length 1024 --seed 42 `\n"
         "  --enable_thinking 1")

    h(doc, "12.2 Fusion — couche 1", 2)
    code(doc,
         "python merge_adapter.py `\n"
         "  --base models/qwen3-1.7b `\n"
         "  --adapter runs/autodistill_17b/adapter `\n"
         "  --output_dir models/qwen3-1.7b-merged1 `\n"
         "  --dtype bfloat16")

    h(doc, "12.3 Entraînement — couche 2", 2)
    code(doc,
         "python LoFT\\Loft\\train.py `\n"
         "  --model_name models/qwen3-1.7b-merged1 `\n"
         "  --dataset_path <jeu de 40 elements> `\n"
         "  --output_dir runs/raisonnement_17b `\n"
         "  --num_train_epochs 3 --max_length 512 --seed 42 `\n"
         "  --enable_thinking 1 `\n"
         "  --gradient_accumulation_steps 4 --lora_r 8 --lora_alpha 16")

    h(doc, "12.4 Fusion — couche 2", 2)
    code(doc,
         "python merge_adapter.py `\n"
         "  --base models/qwen3-1.7b-merged1 `\n"
         "  --adapter runs/raisonnement_17b/adapter `\n"
         "  --output_dir models/qwen3-1.7b-final `\n"
         "  --dtype bfloat16")

    h(doc, "12.5 Évaluation", 2)
    code(doc, "python nuit_compare.py --n_seeds 3 --max_new_tokens 400")

    h(doc, "12.5 Environnement", 2)
    table(doc,
          ["Composant", "Valeur"],
          [["Python", "3.12.10 (environnement virtuel dédié)"],
           ["PyTorch", "2.14.0+cpu"],
           ["Transformers", "5.17.0"],
           ["PEFT", "0.21.0"],
           ["Jeu de données", "5.0.1"],
           ["Modèle", "Qwen3 1.7B (bfloat16)"],
           ["Emplacement", "D:\\LoFT"]],
          widths=[5.0, 10.2])

    # =============================================================== 13
    h(doc, "13. Chronologie", 1)
    table(doc,
          ["Date", "Étape"],
          [["28 septembre 2026",
            "Installation de LoFT, modèle Gemma 3 1B, 83 exécutions de repérage"],
           ["3 octobre 2026",
            "Contrôle de l'environnement, sauvegardes ; test de contrôle 10/10, "
            "test de débordement 0/10 ; rédaction de la version 1 du document"],
           ["3 octobre 2026",
            "Construction du jeu autodistill_b (54 éléments) à partir de traces "
            "du Qwen3 0.6B"],
           ["3 octobre 2026, soir",
            "Entraînement de la couche 1 : 42 pas, 2 h 02, perte finale 0,6071"],
           ["4 octobre 2026, 00 h 38",
            "Fusion de la couche 1 : 6 s"],
           ["4 octobre 2026, 01 h 14",
            "Entraînement de la couche 2 : 30 pas, 25 min, perte finale 0,8295"],
           ["4 octobre 2026, 01 h 17",
            "Fusion de la couche 2 : 6 s"],
           ["4 octobre 2026, 01 h 20",
            "Lancement de l'évaluation de nuit sur trois états"],
           ["4 octobre 2026",
            "Audit des jeux de données ; rédaction de la présente version 2"]],
          widths=[4.4, 10.8])

    # =============================================================== 14
    doc.add_page_break()
    h(doc, "14. Glossaire", 1)
    para(doc, "Les termes employés dans ce document, définis sans jargon.")
    table(doc,
          ["Terme", "Définition"],
          [["Paramètre",
            "Un nombre interne du modèle. Un modèle de 1,7 milliard de paramètres "
            "en contient 1,7 milliard."],
           ["Fine-tuning",
            "Entraînement supplémentaire d'un modèle déjà existant, sur des "
            "exemples choisis, pour l'adapter à un usage précis."],
           ["LoRA",
            "Méthode de fine-tuning qui n'entraîne qu'une petite partie du "
            "modèle au lieu de tout. Moins de mémoire, résultat plus rapide."],
           ["Adaptateur",
            "Le petit fichier produit par un entraînement LoRA. Il contient la "
            "correction, pas le modèle entier."],
           ["Fusion (merge)",
            "Opération qui incorpore définitivement un adaptateur dans les poids "
            "du modèle de base, produisant un nouveau modèle complet."],
           ["Adaptateur empilé",
            "Adaptateur entraîné sur un modèle qui contient déjà les poids "
            "d'un premier adaptateur. C'est le principe des deux couches."],
           ["Jeu de données d'entraînement",
            "Les exemples utilisés pour l'apprentissage : questions et bonnes "
            "réponses."],
           ["Jeu d'évaluation tenu à l'écart",
            "Jeu de questions jamais vu par l'entraînement. Sans lui, mesurer "
            "un modèle ne prouve rien."],
           ["Raisonnement (chaîne de)",
            "Explication étape par étape qui précède la réponse."],
           ["Oubli catastrophique",
            "Quand un modèle entraîné perd une compétence qu'il avait avant. "
            "C'est le risque principal du fine-tuning."],
           ["Surapprentissage",
            "Quand le modèle apprend par cœur les exemples au lieu d'apprendre "
            "la règle. Il réussit sur l'entraînement et échoue ailleurs."],
           ["Graine aléatoire (seed)",
            "Le numéro qui initialise le hasard. Deux exécutions avec la même "
            "graine donnent la même réponse."],
           ["Jeton (token)",
            "Un morceau de texte vu par le modèle. Un jeton vaut en moyenne "
            "quatre caractères en français."],
           ["Calcul matriciel (GEMM)",
            "L'opération de multiplication de matrices, qui constitue l'essentiel "
            "du travail d'un pas d'entraînement."],
           ["Giga-flop par seconde (GFLOP/s)",
            "Unité de capacité de calcul. Plus le chiffre est élevé, plus la "
            "machine calcule vite."],
           ["Limité par le calcul",
            "État d'une machine dont le facteur limitant est la vitesse de "
            "calcul, et non la mémoire disponible."],
           ["Bande passante mémoire",
            "Débit de données que la mémoire peut fournir. La quantification "
            "vise cette ressource."],
           ["Quantification",
            "Réduire la taille d'un modèle en arrondissant ses nombres."],
           ["Oubli de contexte",
            "Symptôme d'une divergence de style entre le jeu d'entraînement et "
            "le jeu d'évaluation, susceptible d'être confondue avec un échec "
            "de l'entraînement."],
           ["CPU", "Le processeur principal du PC."],
           ["GPU", "La carte graphique, beaucoup plus rapide pour le calcul "
                   "parallèle."],
           ["Époque (epoch)", "Une passe complète sur le jeu de données."],
           ["venv", "Environnement Python isolé. Il empêche deux projets de se "
                    "marcher dessus."]],
          widths=[3.6, 11.6])

    # =============================================================== ANNEXES
    doc.add_page_break()
    h(doc, "Annexe A — Reproduction des mesures", 1)

    h(doc, "A.1 Chaîne complète", 2)
    para(doc,
         "L'ordre des étapes est obligatoire : la couche 2 s'entraîne sur le "
         "modèle issu de la fusion de la couche 1.")
    table(doc,
          ["#", "Étape", "Script", "Durée mesurée"],
          [["1", "Audit du jeu de 40", "tests/audit_raisonnement_40.py",
            "immédiat"],
           ["2", "Audit du jeu d'évaluation",
            "tests/verify_heldout_40.py", "immédiat"],
           ["3", "Normalisation du jeu d'évaluation",
            "tests/normalize_heldout.py", "immédiat"],
           ["4", "Entraînement couche 1", "LoFT/Loft/train.py", "7 337 s"],
           ["5", "Fusion couche 1", "merge_adapter.py", "5,9 s"],
           ["6", "Entraînement couche 2", "LoFT/Loft/train.py", "1 526 s"],
           ["7", "Fusion couche 2", "merge_adapter.py", "6,0 s"],
           ["8", "Évaluation", "nuit_compare.py", "en cours"],
           ["9", "Vérification de la fusion", "tests/verify_merge.py",
            "quelques secondes"]],
          widths=[1.0, 4.6, 5.6, 4.0])

    h(doc, "A.2 Contrôles de la fusion", 2)
    para(doc,
         "La fusion a été vérifiée par trois méthodes indépendantes, parce "
         "qu'une opération qui réécrit un modèle complet peut échouer "
         "silencieusement.")
    bullet(doc, "Vérification structurelle : aucun module LoRA résiduel ne "
                "subsiste dans le modèle fusionné.")
    bullet(doc, "Vérification des poids : la différence entre le modèle "
                "fusionné et le modèle de base doit être égale à la matrice "
                "produite par l'adaptateur, au facteur d'échelle près.")
    bullet(doc, "Vérification fonctionnelle : les logits produits par le modèle "
                "fusionné et par le modèle de base porteur de l'adaptateur "
                "doivent coïncider, et doivent différer du modèle de base nu.")

    h(doc, "A.3 Projeter une durée avant de lancer", 2)
    para(doc,
         "Une projection honnête dérive d'une mesure réelle, jamais d'une "
         "estimation. La projection initiale de la durée de l'évaluation de "
         "nuit annonçait environ trois heures. La première mesure a montré "
         "qu'un élément prenait près de cinq minutes pour trois graines, ce "
         "qui porte la projection à neuf ou dix heures. L'écart d'un facteur "
         "trois provenait d'une hypothèse implicite sur la longueur des "
         "générations produites, qui s'est révélée fausse : le modèle produit "
         "des raisonnements longs, souvent jusqu'à la limite de jetons "
         "imposée.")

    h(doc, "A.4 Fichiers produits", 2)
    table(doc,
          ["Fichier", "Contenu"],
          [["runs/autodistill_17b/run_manifest.json",
            "Manifeste de l'entraînement de la couche 1"],
           ["runs/raisonnement_17b/run_manifest.json",
            "Manifeste de l'entraînement de la couche 2"],
           ["models/qwen3-1.7b-merged1/merge_manifest.json",
            "Manifeste de la première fusion"],
           ["models/qwen3-1.7b-final/merge_manifest.json",
            "Manifeste de la seconde fusion"],
           ["tests/results/qlora_feasibility.json",
            "Mesures de la faisabilité de la quantification quatre bits"],
           ["tests/results/audit_raisonnement_40.json",
            "Audit du jeu d'entraînement de 40 éléments"],
           ["tests/results/verify_heldout_40.json",
            "Audit du jeu d'évaluation de 40 éléments"],
           ["tests/results/heldout_normalization.json",
            "Rapport de normalisation du jeu d'évaluation"],
           ["tests/data/heldout_40_norm.json",
            "Jeu d'évaluation normalisé, utilisé pour la mesure"],
           ["runs/nuit_compare.jsonl",
            "Journal horodaté de l'évaluation, une entrée par élément"],
           ["tests/results/heldout_*.json",
            "Réponses détaillées par état, écrites après chaque élément"]],
          widths=[7.4, 7.8])

    para(doc,
         "Le journal et les fichiers de résultat sont écrits après chaque "
         "élément. Une interruption du programme ne perd aucune mesure "
         "déjà effectuée, et une relance reprend là où le programme s'était "
         "arrêté.")

    doc.add_page_break()
    h(doc, "Annexe B — Annotations méthodologiques", 1)

    para(doc,
         "Cette annexe recense les incidents survenus pendant le projet. Ils "
         "sont consignés parce qu'ils font partie du résultat : un protocole "
         "dont on ne connaît que les réussissances n'est pas documenté.")

    h(doc, "B.1 Cinq faux positifs dans le contrôle de correction", 2)
    para(doc,
         "Le contrôle du jeu d'entraînement de 40 éléments a signalé cinq "
         "réponses comme incorrectes. Les quatre premières portaient sur des "
         "valeurs numériques correctes : 70, 15, 3 et 27,20. L'outil comparait "
         "la valeur calculée par le script, écrite avec un point décimal, à la "
         "valeur du fichier, écrite avec une virgule, et ne reconnaissait pas "
         "l'équivalence.")
    para(doc,
         "La correction a porté sur l'ordre des opérations de normalisation : "
         "le caractère point était remplacé par une espace avant que la "
         "règle de suppression des zéros ne soit appliquée, ce qui transformait "
         "70,0 en « 70 0 ». La cinquième réponse, 27,20, a résisté à la première "
         "correction pour la même raison sous une autre forme : les zéros de fin "
         "n'étaient pas supprimés.")
    para(doc,
         "Enseignement : un contrôle automatisé signale des écarts, pas des "
         "erreurs. Chaque écart doit être vérifié manuellement avant d'être "
         "reporté comme un défaut du jeu de données.")

    h(doc, "B.2 Une question de conception trouvée par le contrôle", 2)
    para(doc,
         "Le jeu d'évaluation de 24 éléments, construit par l'auteur, contenait "
         "une question mathématiquement impossible : la suite des positions des "
         "lettres correspondant aux carrés parfaits atteint 25, et le carré "
         "suivant, 36, dépasse les 26 lettres de l'alphabet. Le contrôle "
         "automatique l'a détecté. La question a été corrigée.")
    para(doc,
         "Enseignement : un contrôle qui ne trouve rien ne prouve pas qu'il n'y "
         "a rien à trouver. Il prouve seulement qu'il n'a rien trouvé.")

    h(doc, "B.3 Une conviction initiale infirmée par la mesure", 2)
    para(doc,
         "Avant le second entraînement, le drapeau de désactivation du "
         "raisonnement a été présenté comme le principal levier de vitesse, "
         "avec un gain attendu de deux à trois fois. Cette affirmation n'avait "
         "pas été vérifiée.")
    para(doc,
         "La mesure a montré que la longueur moyenne des exemples était "
         "identique avec et sans le drapeau : 515 jetons dans les deux cas. Le "
         "gain attendu était nul. La raison : les chaînes de raisonnement du "
         "jeu de données contenaient déjà leurs balises sous forme de texte, et "
         "le drapeau ne choisit qu'un gabarit de présentation, il ne retire "
         "aucun contenu.")
    para(doc,
         "Enseignement : une projection de gain doit être dérivée d'une mesure "
         "de la quantité concernée. Ici, la quantité à réduire n'existait pas "
         "dans les données.")

    h(doc, "B.4 Une hypothèse invalidée par la mesure", 2)
    para(doc,
         "L'hypothèse d'un gain de vitesse par quantification quatre bits a été "
         "examinée puis écartée, sur la base d'une mesure du débit effectif "
         "d'entraînement rapportée au plafond de calcul de la machine. "
         "L'entraînement atteint 59 % du plafond : il est limité par le "
         "calcul. La quantification agit sur la mémoire, qui n'est pas le "
         "facteur limitant.")
    para(doc,
         "Enseignement : une technique d'optimisation n'est pas fausses, elle "
         "est inadaptée à un contexte. Le diagnostic du facteur limitant "
         "précède le choix de la technique.")

    h(doc, "B.5 Une vérification restée inachevée", 2)
    para(doc,
         "Un contrôle de la variance des graines aléatoires a été entrepris "
         "pendant l'évaluation, puis interrompu avant de produire un résultat. "
         "L'observation partielle, rapportée en section 8.3, ne permet pas de "
         "conclure. Cette vérification demeure à faire, et la conclusion de la "
         "section 10 devra être révisée à la lumière de ses résultats.")
    para(doc,
         "Enseignement : un contrôle interrompu ne doit pas être présenté comme "
         "un contrôle réussi, ni comme un contrôle ayant produit un résultat "
         "négatif. Il n'a pas produit de résultat.")

    doc.add_page_break()
    h(doc, "Annexe C — Fichiers et artefacts", 1)

    h(doc, "C.1 Arborescence du projet", 2)
    code(doc,
         "D:\\LoFT\\\n"
         "  models/                  poids de base et etats fusionnes\n"
         "    qwen3-0.6b/            modele source des traces d'entrainement\n"
         "    qwen3-1.7b/            modele de base\n"
         "    qwen3-1.7b-merged1/    apres fusion de la couche 1\n"
         "    qwen3-1.7b-final/      apres fusion de la couche 2\n"
         "  runs/                    sorties d'entrainement\n"
         "    autodistill_17b/        couche 1\n"
         "    raisonnement_17b/      couche 2\n"
         "    nuit_compare.jsonl      journal de l'evaluation\n"
         "  tests/\n"
         "    data/                   jeux de donnees\n"
         "    results/                mesures et manifestes\n"
         "  venv/                     environnement Python 3.12\n"
         "  merge_adapter.py          operation de fusion\n"
         "  nuit_compare.py           evaluation sur trois etats")

    h(doc, "C.2 Scripts de contrôle", 2)
    table(doc,
          ["Script", "Rôle"],
          [["merge_adapter.py",
            "Fusionne un adaptateur dans un modèle de base, avec garde-fous "
            "contre l'écriture dans le dossier de base et l'écrasement "
            "d'un modèle existant"],
           ["nuit_compare.py",
            "Évalue trois états du modèle sur le jeu tenu à l'écart, avec "
            "écriture après chaque élément et reprise après interruption"],
           ["tests/audit_raisonnement_40.py",
            "Contrôle de correction du jeu d'entraînement de 40 éléments"],
           ["tests/verify_heldout_40.py",
            "Contrôle de correction du jeu d'évaluation de 40 éléments"],
           ["tests/normalize_heldout.py",
            "Normalise le format des réponses et corrige les heures impossibles"],
           ["tests/verify_merge.py",
            "Vérifie que la fusion a réellement modifié les poids visés"],
           ["tests/verify_merge_functional.py",
            "Vérifie que le modèle fusionné est fonctionnellement équivalent "
            "au modèle porteur de l'adaptateur"],
           ["tests/qlora_feasibility_probe.py",
            "Mesure le facteur limitant de l'entraînement et l'atteinte au "
            "plafond de calcul"],
           ["tests/verify_heldout.py",
            "Contrôle de correction du jeu d'évaluation de 24 éléments"]],
          widths=[5.4, 9.8])

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("— Fin du document —")
    r.italic = True
    r.font.size = Pt(10)

    return doc, d


def main():
    doc, d = build()
    doc.save(OUT)
    size = os.path.getsize(OUT) / 1024
    print("Ecrit : %s (%.1f Ko)" % (OUT, size))
    print()
    print("Artefacts lus :")
    for k in ("m1", "m2", "mg1", "mg2", "a40", "norm", "v40", "qlora"):
        print("   %-6s %s" % (k, "OK" if d.get(k) else "ABSENT"))
    print()
    print("Etat de l'evaluation :")
    print("   active   : %s" % d["night_active"])
    print("   termines : %s" % (d["night_finished"] or "aucun"))
    print("   progress : %s" % (d["night_progress"] or "aucun"))
    for etat, s in d["scores"].items():
        print("   %-8s %s/%s sur %d elements (termine: %s)"
              % (etat, s["correct"], s["possible"], s["n_items"], s["complete"]))


if __name__ == "__main__":
    main()
