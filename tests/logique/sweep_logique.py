"""Test 3/5 (version C) — Intersection de contraintes.

Ni calcul, ni conversion, ni heure, ni regles a prioriser. Logique de
filtration : 7 lots, 3 criteres, un seul survit.

Chaque candidat elimine rate EXACTEMENT UN critere. Une mauvaise reponse
identifie donc LE critere non applique -- le score diagnostique, il ne
se contente pas de dire "faux".

verifier() est execute a chaque run: si le puzzle cesse d'avoir une
reponse unique, le script s'arrete au lieu de produire un score faux.

Usage (PowerShell):
  cd D:\\LoFT\\tests\\logique
  ..\\..\\venv\\Scripts\\Activate.ps1
  python sweep_logique.py --tag logique_C --n_seeds 10
"""
import argparse
import json
import os
import re
import time
import unicodedata

import psutil
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("LOFT_MODEL", "D:/LoFT/models/gemma-3-1b-it")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# --- L'inventaire -------------------------------------------------
# REGLE DE CONCEPTION: chaque candidat elimine rate EXACTEMENT UN
# critere. verifier() le controle a chaque run -- sinon une mauvaise
# reponse ne pourrait pas dire QUEL critere a ete rate.
LOTS = [
    # id  categorie   kg   destination   elimine par
    ("A", "standard", 12, "europe"),   # critere 1
    ("B", "standard", 15, "europe"),   # critere 1
    ("C", "fragile", 9, "europe"),     # -> LE survivant
    ("D", "fragile", 16, "europe"),    # critere 2 (poids)
    ("E", "standard", 9, "europe"),    # critere 1
    ("F", "fragile", 9, "asie"),       # critere 3 (destination)
]

TABLEAU = (
    "Lot | Categorie | Poids | Destination\n"
    " A  | standard | 12 kg | Europe\n"
    " B  | standard | 15 kg | Europe\n"
    " C  | fragile  |  9 kg | Europe\n"
    " D  | fragile  | 16 kg | Europe\n"
    " E  | standard |  9 kg | Europe\n"
    " F  | fragile  |  9 kg | Asie"
)

QUESTION = (
    "Une entreprise doit choisir un seul lot a envoyer. Ses criteres sont :\n"
    "1. Le lot doit etre de categorie 'fragile'.\n"
    "2. Le lot doit peser 15 kg ou moins.\n"
    "3. Le lot doit etre destine a l'Europe.\n\n"
    "Inventaire :\n" + TABLEAU + "\n\n"
    "Quel lot faut-il choisir ? Reponds uniquement par la lettre du lot."
)

# Meme criteres, ordre inverse. Le critere discriminant ("fragile") passe
# de la position 1 a la position 3. L'inventaire est IDENTIQUE -- seule
# variable qui change est l'ordre de lecture.
QUESTION_INV = (
    "Une entreprise doit choisir un seul lot a envoyer. Ses criteres sont :\n"
    "1. Le lot doit etre destine a l'Europe.\n"
    "2. Le lot doit peser 15 kg ou moins.\n"
    "3. Le lot doit etre de categorie 'fragile'.\n\n"
    "Inventaire :\n" + TABLEAU + "\n\n"
    "Quel lot faut-il choisir ? Reponds uniquement par la lettre du lot."
)

ATTENDU = "C"

# Reponse attendue -> critere que le modele a rate.
DIAGNOSTIC = {
    "A": "categorie fragile non imposee (critere 1)",
    "B": "categorie fragile non imposee (critere 1)",
    "E": "categorie fragile non imposee (critere 1)",
    "D": "poids <= 15 kg non verifie (critere 2)",
    "F": "destination Europe non verifiee (critere 3)",
}

# Lettres qui ne sont PAS des lots. "lot a envoyer", "a", "c" (conjonction)
# ne doivent pas etre lus comme des reponses.
MOTS_FR = {"a", "au", "aux", "c", "ce", "et", "il", "en", "e"}


def _norm(s):
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", s)


def repond(texte):
    """Lettre du lot citee en reponse, ou None.

    Cherche d'abord une MAJUSCULE isolee (le modele repond "C"),
    puis "lot <lettre>". Les mots francais sont exclus.
    """
    # 1. Lettre isolee en majuscule dans le texte d'origine.
    for m in re.finditer(r"(?<![A-Za-z])([A-G])(?![A-Za-z])", texte):
        return m.group(1)
    # 2. "lot C" / "lot C." explicite.
    a = _norm(texte)
    m = re.search(r"\blot\s+([a-g])\b", a)
    if m and m.group(1) not in MOTS_FR:
        return m.group(1)
    # 3. Reponse courte: une seule lettre isolee, meme minuscule.
    #    "c" seul est une reponse valide (c'est le lot C). Mais dans une
    #    phrase, "c" est la conjonction et ne compte pas.
    mots = a.split()
    if len(mots) == 1 and len(mots[0]) == 1 and mots[0] in "abcdefg":
        return mots[0]
    # 4. Reponse courte avec un seul mot-cle de lot parmi les autres mots.
    if len(mots) <= 8:
        for w in mots:
            if len(w) == 1 and w in "abcdefg" and w not in MOTS_FR:
                return w
    return None


def scorer(ans):
    lettre = repond(ans)
    if lettre is None:
        return "FAIL", "aucune lettre de lot identifiee"
    lettre = lettre.upper()
    if lettre == ATTENDU:
        return "PASS", "lot C, les 3 criteres respectes"
    if lettre in DIAGNOSTIC:
        return "FAIL", "lot %s : %s" % (lettre, DIAGNOSTIC[lettre])
    return "FAIL", "lot %s inattendu" % lettre


def verifier():
    """Le puzzle doit avoir UN survivant, et chaque candidat 1 seul critere."""
    def criteres_manques(lot):
        lid, cat, kg, zone = lot
        m = []
        if cat != "fragile":
            m.append("1")
        if not kg <= 15:
            m.append("2")
        if zone != "europe":
            m.append("3")
        return m

    survivants = [l[0] for l in LOTS if not criteres_manques(l)]
    print("Survivants (3 criteres) :", survivants)
    assert [s.upper() for s in survivants] == ["C"], \
        "BUG: reponse unique non garantie -> %s" % survivants

    print("Diagnostic par candidat elimine :")
    for l in LOTS:
        m = criteres_manques(l)
        if l[0] == "C":
            print("  %s -> SURVIVANT" % l[0])
        else:
            assert len(m) == 1, \
                "BUG: lot %s rate %d criteres (%s), le test diagnostiquerait mal" % (
                    l[0], len(m), m)
            print("  %s -> rate critere %s" % (l[0], m[0]))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--inversed", action="store_true",
                    help="ordre inverse des criteres (test C-bis)")
    a = ap.parse_args()

    print("=" * 62)
    verifier()
    print("=" * 62)

    # Affiche la question AVANT de lancer -- le score n'est interpretable
    # que si on sait exactement ce qui a ete demande.
    question = QUESTION_INV if a.inversed else QUESTION
    print("\nQUESTION (%s):\n%s\n" % ("ordre inverse" if a.inversed else "ordre normal", question))
    print("EXPECTED: %s\nMODELE: %s\n" % (ATTENDU, MODEL))
    print("=" * 62)

    gen = dict(max_new_tokens=a.max_new_tokens, do_sample=True,
               temperature=a.temperature, top_k=a.top_k,
               top_p=a.top_p, repetition_penalty=a.repetition_penalty)

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    m.eval()

    enc = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
        # Qwen3 emet un bloc <think> par defaut. Le desactiver est
        # OBLIGATOIRE pour comparer avec Gemma: sinon on mesure la
        # longueur du raisonnement, pas la qualite de la reponse.
        enable_thinking=False,
    )
    n_in = enc["input_ids"].shape[1]

    runs = []
    for seed in range(a.n_seeds):
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.generate(**enc, **gen)
        dt = time.perf_counter() - t0
        new = out[0, n_in:]
        ans = tok.decode(new, skip_special_tokens=True)
        verdict, raison = scorer(ans)
        r = {"seed": seed, "verdict": verdict, "reason": raison,
             "new_tokens": int(new.shape[0]),
             "generate_s": round(dt, 2),
             "tokens_per_s": round(new.shape[0] / dt, 2) if dt else None,
             "answer": ans}
        runs.append(r)
        print("--- seed %d: %s (%s)" % (seed, verdict, raison))
        print(repr(ans))
        print()

    mi = psutil.Process().memory_info()
    n = len(runs)
    summary = {
        "runs": n,
        "pass_count": sum(r["verdict"] == "PASS" for r in runs),
        "fail_count": sum(r["verdict"] == "FAIL" for r in runs),
        "unique_answers": len({r["answer"] for r in runs}),
        "answer_distribution": json.dumps(
            {k: sum(1 for r in runs if repond(r["answer"]) == k)
             for k in sorted({repond(r["answer"]) for r in runs})},
            ensure_ascii=False),
        "mean_new_tokens": round(sum(r["new_tokens"] for r in runs) / n, 1),
        "mean_tokens_per_s": round(sum(r["tokens_per_s"] or 0 for r in runs) / n, 2),
        "peak_wset_gb": round(getattr(mi, "peak_wset", 0) / 1e9, 2)
        if getattr(mi, "peak_wset", None) else None,
    }
    summary["score"] = "%d/%d" % (summary["pass_count"], n)

    res = {"tag": a.tag, "question": question, "inventory": TABLEAU,
           "criteres_inverses": bool(a.inversed),
           "model_path": MODEL,
           "expected_answer": ATTENDU, "diagnostic": DIAGNOSTIC,
           "gen_kwargs": gen, "model_dtype": str(m.dtype),
           "versions": {"torch": torch.__version__,
                        "transformers": transformers.__version__},
           "summary": summary, "runs": runs}
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "%s.json" % a.tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Ecrit : %s" % out)


if __name__ == "__main__":
    main()