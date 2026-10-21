"""Test E — diagnostic de la lecture de borne stricte.

Le test D a montre que Qwen3-0.6B repond 'C' (9 kg) alors que la
regle 4 dit 'plus de 9 kg'. Deux hypotheses:

  H1 : il ne sait pas lire la borne stricte -> avec '9 kg ou moins'
       la bonne reponse redevient C. S'il repond encore D, il n'a
       pas compris la regle du tout.
  H2 : il a mal lu le sens de la regle 4   -> avec '9 kg ou moins'
       il devrait repondre D (il inverse toujours).

Memoire : le test D inversait la regle 4 (ajoutait 'plus de 9 kg'),
le test C ne l'avait pas. Ici on inverse par rapport a C, donc la
regle 4 redevient l'originale du test B.

Meme inventaire que C et D. Seule la regle 4 change de signe.

Usage (PowerShell):
  cd D:\\LoFT\\tests\\diagnostic
  $env:LOFT_MODEL = "D:/LoFT/models/qwen3-0.6b"
  python sweep_diag.py --tag qwen_diag_E --n_seeds 10 --variant E
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

# Inventaire identique aux tests C et D.
LOTS = [
    ("A", "standard", 12, "europe"),
    ("B", "standard", 15, "europe"),
    ("C", "fragile", 9, "europe"),
    ("D", "fragile", 13, "europe"),
    ("E", "standard", 9, "europe"),
    ("F", "fragile", 11, "asie"),
]

TABLEAU = (
    "Lot | Categorie | Poids | Destination\n"
    " A  | standard | 12 kg | Europe\n"
    " B  | standard | 15 kg | Europe\n"
    " C  | fragile  |  9 kg | Europe\n"
    " D  | fragile  | 13 kg | Europe\n"
    " E  | standard |  9 kg | Europe\n"
    " F  | fragile  | 11 kg | Asie"
)

# Regle 4 : le lot doit peser 9 kg OU MOINS (borne fermee).
# C (9 kg) et E (9 kg) sont valides sur ce critere.
# D (13 kg) est exclu.
# Seule C survit aux 4 criteres.
QUESTION = (
    "Une entreprise doit choisir un seul lot a envoyer. Ses criteres sont :\n"
    "1. Le lot doit etre de categorie 'fragile'.\n"
    "2. Le lot doit peser 15 kg ou moins.\n"
    "3. Le lot doit etre destine a l'Europe.\n"
    "4. Le lot doit peser 9 kg ou moins.\n\n"
    "Inventaire :\n" + TABLEAU + "\n\n"
    "Quel lot faut-il choisir ? Reponds uniquement par la lettre du lot."
)

ATTENDU = "C"

DIAGNOSTIC = {
    "D": "borne mal lue: 9 kg compte comme 'plus de 9' (H1 confirmee?)",
    "C": "corrige, respecte la borne fermee",
    "A": "filtre par poids seul, ignore la categorie",
    "B": "categorie fragile non imposee (critere 1)",
    "E": "filtre par poids seul, ignore la categorie",
    "F": "filtre par categorie seule, ignore le critere 3",
}


def _norm(s):
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", s)


MOTS_FR = {"a", "au", "aux", "c", "ce", "et", "il", "en", "e"}


def repond(texte):
    """Lettre du lot cite EN REPONSE, pas la 1re lettre citee.

    BUG CORRIGE 2026-10-03: la version precedente prenait la premiere
    lettre isolee trouvee. Qwen qui recapitule les 6 lots avant de
    repondre citait 'A' en premier -> score FAUX alors que sa reponse
    finale ('Reponse : **C**') etait correcte.

    Regle: on privilegie la derniere mention explicite (la conclusion),
    et on ignore les enumerations introduites par un mot de liste.
    """
    # 1. Conclusion explicite: "Reponse : X", "Reponse: **X**"
    m = re.search(r"r[ée]ponse\s*[:\-]?\s*\**\s*([A-F])\b", texte, re.I)
    if m:
        return m.group(1)

    # 2. "Le lot X" en fin de texte (conclusion naturalle)
    tous = re.findall(r"\blot\s+(?:[a-z]+\s+)?\**([A-F])\**\b", texte)
    if tous:
        return tous[-1]

    # 3. Lettre isolee -- on prend la DERNIERE (conclusion en fin).
    isolees = re.findall(r"(?<![A-Za-z])([A-F])(?![A-Za-z])", texte)
    if isolees:
        return isolees[-1]

    a = _norm(texte)
    m = re.search(r"\blot\s+([a-f])\b", a)
    if m:
        return m.group(1).upper()
    mots = a.split()
    if len(mots) == 1 and len(mots[0]) == 1 and mots[0] in "abcdef":
        return mots[0].upper()
    if len(mots) <= 8:
        for w in mots:
            if len(w) == 1 and w in "abcdef" and w not in MOTS_FR:
                return w.upper()
    return None


def verifier():
    """Le puzzle doit avoir UN survivant = C."""
    sur = [lid for lid, cat, kg, zone in LOTS
           if cat == "fragile" and kg <= 15 and zone == "europe" and kg <= 9]
    print("Survivants (4 criteres) :", sur)
    assert sur == [ATTENDU], "BUG: attendu %s, survivants %s" % (ATTENDU, sur)
    print("OK : reponse unique = %s" % ATTENDU)
    print()
    print("Elimines :")
    for lid, cat, kg, zone in LOTS:
        m = []
        if cat != "fragile":
            m.append("critere 1 (categorie)")
        if not kg <= 15:
            m.append("critere 2 (poids <= 15)")
        if zone != "europe":
            m.append("critere 3 (destination)")
        if not kg <= 9:
            m.append("critere 4 (poids <= 9)")
        if m:
            print("  %s -> %s" % (lid, " + ".join(m)))
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
    a = ap.parse_args()

    print("=" * 66)
    verifier()
    print("=" * 66)
    print("\nQUESTION (test E):\n%s\n" % QUESTION)
    print("EXPECTED: %s\nMODELE: %s\n" % (ATTENDU, MODEL))
    print("=" * 66)

    gen = dict(max_new_tokens=a.max_new_tokens, do_sample=True,
               temperature=a.temperature, top_k=a.top_k,
               top_p=a.top_p, repetition_penalty=a.repetition_penalty)

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    m.eval()

    enc = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
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
        ans = tok.decode(out[0, n_in:], skip_special_tokens=True)
        lettre = repond(ans)
        verdict = "PASS" if lettre == ATTENDU else "FAIL"
        runs.append({
            "seed": seed, "answer_letter": lettre, "verdict": verdict,
            "reason": DIAGNOSTIC.get(lettre, "sans objet" if verdict == "FAIL"
                                     else "borne fermee respectee"),
            "new_tokens": int(out.shape[1] - n_in),
            "generate_s": round(dt, 2),
            "tokens_per_s": round((out.shape[1] - n_in) / dt, 2) if dt else None,
            "answer": ans,
        })
        print("--- seed %d: %s (repondu %s | %s)"
              % (seed, verdict, lettre, runs[-1]["reason"]))
        print(repr(ans))

    mi = psutil.Process().memory_info()
    n = len(runs)
    summary = {
        "runs": n,
        "pass_count": sum(r["verdict"] == "PASS" for r in runs),
        "answer_distribution": json.dumps(
            {str(k): sum(1 for r in runs if r["answer_letter"] == k)
             for k in sorted({r["answer_letter"] for r in runs},
                             key=lambda x: (x is None, x))},
            ensure_ascii=False),
        "unique_answers": len({r["answer"] for r in runs}),
        "mean_new_tokens": round(sum(r["new_tokens"] for r in runs) / n, 1),
        "peak_wset_gb": round(getattr(mi, "peak_wset", 0) / 1e9, 2)
        if getattr(mi, "peak_wset", None) else None,
    }
    summary["score"] = "%d/%d" % (summary["pass_count"], n)

    res = {
        "tag": a.tag, "test": "E", "question": QUESTION, "inventory": TABLEAU,
        "expected_answer": ATTENDU, "diagnostic": DIAGNOSTIC,
        "model_path": MODEL, "gen_kwargs": gen, "model_dtype": str(m.dtype),
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
        "summary": summary, "runs": runs,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "%s.json" % a.tag)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Ecrit : %s" % out_path)


if __name__ == "__main__":
    main()