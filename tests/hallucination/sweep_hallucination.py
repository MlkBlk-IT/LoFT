"""Test 3/5 (version B) — Anti-hallucination par deduction.

Ni calcul, ni conversion, ni notation. Le modele doit COMBINER deux
regles du texte pour trancher un cas que le texte ne traite pas
explicitement. S'il recite, il se trompe.

Scoring 100% automatique : reponse attendue = un seul libelle exact.
Pas de jugement humain, pas de LLM-juge.

Usage (PowerShell):
  cd D:\\LoFT\\tests\\hallucination
  ..\\..\\venv\\Scripts\\Activate.ps1
  python sweep_hallucination.py --tag halluc_B_regles --n_seeds 10
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

SOURCE = (
    "Voici les regles de classement d'un serveur de fichiers.\n"
    "Regle 1 : tout fichier dont le nom commence par 'A-' est place dans le "
    "dossier Archives.\n"
    "Regle 2 : tout fichier dont le nom commence par 'T-' est place dans le "
    "dossier Temporaire.\n"
    "Regle 3 : tout fichier image (extension .png, .jpg ou .jpeg) est place "
    "dans le dossier Courant, meme si son nom commence par 'A-' ou 'T-'.\n"
    "Regle 4 : tout autre fichier est place dans le dossier Courant."
)

QUESTION = (
    "D'apres les regles ci-dessus, dans quel dossier va le fichier "
    "'T-rapport.png' ?\n\n"
    "Regles :\n" + SOURCE + "\n\n"
    "Reponds uniquement par le nom du dossier, sans explication."
)

ATTENDU = "courant"

# Mots qui signalent que le modele a recite les regles au lieu de trancher.
INTERDITS = ["temporaire", "archive"]

RETOUR = re.compile(r"dossier\s+([a-z]+)", re.I)


def _norm(s):
    """Minuscules, accents retires, ponctuation en espaces."""
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", s)


def scorer(ans):
    """Retourne (verdict, raison)."""
    a = _norm(ans)

    # Le modele doit nommer le dossier. S'il ne le fait pas, on regarde
    # quand meme les mots cles presents pour diagnostiquer.
    m = RETOUR.search(a)
    candidat = m.group(1) if m else None

    if candidat is None:
        # Pas de "dossier X" : chercher le mot isole.
        for mot in ("temporaire", "courant", "archive"):
            if re.search(r"\b%s\b" % mot, a):
                candidat = mot
                break

    if candidat is None:
        return "FAIL", "aucun nom de dossier cite"

    if candidat != ATTENDU:
        return "FAIL", "a choisi '%s' au lieu de '%s'" % (candidat, ATTENDU)

    # Bonne reponse MAIS il a aussi cite les mauvais dossiers (recitation).
    if any(re.search(r"\b%s\b" % w, a) for w in INTERDITS):
        return "FAIL", "bonne reponse mais diction des regles (recite)"

    return "PASS", "deduit '%s' sans recital" % candidat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    a = ap.parse_args()

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
        r = {
            "seed": seed,
            "verdict": verdict,
            "reason": raison,
            "new_tokens": int(new.shape[0]),
            "generate_s": round(dt, 2),
            "tokens_per_s": round(new.shape[0] / dt, 2) if dt else None,
            "answer": ans,
        }
        runs.append(r)
        print("--- seed %d: %s (%s) | %d tok"
              % (seed, verdict, raison, r["new_tokens"]))
        print(repr(ans))
        print()

    mi = psutil.Process().memory_info()
    n = len(runs)
    uniques = len({r["answer"] for r in runs})
    summary = {
        "runs": n,
        "pass_count": sum(r["verdict"] == "PASS" for r in runs),
        "fail_count": sum(r["verdict"] == "FAIL" for r in runs),
        "unique_answers": uniques,
        "mean_new_tokens": round(sum(r["new_tokens"] for r in runs) / n, 1),
        "mean_tokens_per_s": round(sum(r["tokens_per_s"] or 0 for r in runs) / n, 2),
        "peak_wset_gb": round(getattr(mi, "peak_wset", 0) / 1e9, 2)
        if getattr(mi, "peak_wset", None) else None,
    }
    summary["score"] = "%d/%d" % (summary["pass_count"], n)

    res = {
        "tag": a.tag,
        "question": QUESTION,
        "source": SOURCE,
        "expected_answer": ATTENDU,
        "model_path": MODEL,
        "gen_kwargs": gen,
        "model_dtype": str(m.dtype),
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
        "summary": summary,
        "runs": runs,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "%s.json" % a.tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Ecrit : %s" % out)


if __name__ == "__main__":
    main()