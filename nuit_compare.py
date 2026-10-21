"""nuit_compare.py -- Evalue les 3 etats du modele sur le held-out.

Trois etapes de la chaine, memes questions, meme scoring:

  1. base      models/qwen3-1.7b            (rien.merge)
  2. merged1   models/qwen3-1.7b-merged1     (+ adaptateur autodistill 54)
  3. final     models/qwen3-1.7b-final       (+ adaptateur raisonnement 40)

Le held-out (heldout_40_norm.json) n'a JAMAIS ete entrain: c'est ce qui
distingue memorisation de generalisation.

Ecriture INCREMENTALE par item: une interruption garde tout le deja fait.
Journal JSONL: runs/nuit_compare.jsonl

Usage (PowerShell):
  cd D:\\LoFT
  python nuit_compare.py --n_seeds 3 --max_new_tokens 400
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = r"D:\LoFT"
HELDOUT = os.path.join(ROOT, "tests", "data", "heldout_40_norm.json")
RESULTS = os.path.join(ROOT, "tests", "results")
JOURNAL = os.path.join(ROOT, "runs", "nuit_compare.jsonl")

ETATS = [
    ("base", os.path.join(ROOT, "models", "qwen3-1.7b")),
    ("merged1", os.path.join(ROOT, "models", "qwen3-1.7b-merged1")),
    ("final", os.path.join(ROOT, "models", "qwen3-1.7b-final")),
]


# ------------------------------------------------------------------ scoring
def _norm(s):
    """Minuscules, accents retires, ponctuation en espaces.

    Les separateurs decimaux deviennent des espaces: '49,80' et '49.80'
    doivent tous deux produire '49 80', sinon un modele qui repond en
    format anglais est note FAUX a tort.
    """
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s:]", " ", s)


def _variantes(att):
    v = {att}
    for a, b in ((":", "h"), (" ", ""), (".", ","), (",", "."),
                 (" ", ":"), ("", " ")):
        v.add(att.replace(a, b))
    v.add(att.upper())
    return {x for x in v if x}


def reponse_extraite(texte):
    """Derniere valeur citee -- la conclusion, pas l'enumeration."""
    m = re.findall(r"r[ée]ponse\s*[:\-]?\s*\**\s*([^\n.,;]{1,40})", texte, re.I)
    if m:
        return m[-1].strip().strip("*").rstrip(".")
    h = re.findall(r"\b(\d{1,2}[:h]\d{2})\b", texte)
    if h:
        return h[-1].replace("h", ":")
    n = re.findall(r"-?\d+(?:[.,]\d+)?", texte)
    if n:
        return n[-1]
    return texte.strip()[:30]


def scorer(texte, attendu):
    """Juge une reponse generee. Binaire, sans LLM-juge."""
    norm = _norm(texte)
    att = _norm(str(attendu)).strip()
    if not att:
        return False, "attendu vide"

    if norm.strip() == att:
        return True, "exact"

    for v in _variantes(att):
        if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(v), norm):
            return True, "contenu (%s)" % v

    # tolerance numerique <= 1%
    chiffres = re.findall(r"-?\d+(?:[.,]\d+)?", norm)
    try:
        cible = float(att.replace(",", "."))
        for c in chiffres:
            v = float(c.replace(",", "."))
            if abs(cible) > 1e-9 and abs(v - cible) / abs(cible) <= 0.01:
                return True, "numerique ~%s" % v
    except ValueError:
        pass

    # reponse en un mot: le mot doit apparaitre entier
    if " " not in att:
        for w in re.findall(r"(?<![a-z])[a-z]+(?![a-z])", norm):
            if w == att:
                return True, "mot isole %s" % w

    return False, "manque"


# ------------------------------------------------------------------ generation
def generer(model, tok, question, max_new_tokens, temperature, top_p, top_k, seed):
    torch.manual_seed(seed)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=True)
    n_in = enc["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            top_p=top_p, top_k=top_k, repetition_penalty=1.0)
    dt = time.perf_counter() - t0
    txt = tok.decode(out[0, n_in:], skip_special_tokens=True)
    return txt, int(out.shape[1] - n_in), dt


def log(**kw):
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), **kw}
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[journal] %s" % json.dumps(rec, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = json.load(open(HELDOUT, encoding="utf-8"))["items"]
    if args.limit:
        items = items[:args.limit]
    os.makedirs(RESULTS, exist_ok=True)

    print("=" * 78)
    print("COMPARE DE NUIT -- %d items x %d seeds x %d etats"
          % (len(items), args.n_seeds, len(ETATS)))
    print("held-out : %s" % HELDOUT)
    print("=" * 78)
    log(event="start", n_items=len(items), n_seeds=args.n_seeds,
        etats=[n for n, _ in ETATS])

    global_scores = {}

    for etat, path in ETATS:
        if not os.path.exists(os.path.join(path, "config.json")):
            print("\n[SKIP] %s absent : %s" % (etat, path))
            log(event="skip_missing", etat=etat, path=path)
            continue

        out_path = os.path.join(RESULTS, "heldout_%s.json" % etat)
        # Reprise: si le JSON existe et est complet, on ne recommence pas.
        if os.path.exists(out_path):
            try:
                d = json.load(open(out_path, encoding="utf-8"))
                if d.get("complete"):
                    print("\n[skip] %s deja complet (%d items)"
                          % (etat, len(d["items"])))
                    global_scores[etat] = d["total_correct"]
                    global_scores[etat + "_n"] = d["total_possible"]
                    log(event="deja_fait", etat=etat)
                    continue
            except Exception:
                pass

        print("\n" + "=" * 78)
        print("ETAT : %s  ->  %s" % (etat, path))
        print("=" * 78)

        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        model.eval()

        resultats = []
        corrects = 0
        t_etat = time.time()

        for idx, it in enumerate(items, 1):
            qid = it["id"]
            rep, sc, dt = [], 0, 0.0
            for s in range(args.n_seeds):
                txt, ntok, dt = generer(model, tok, it["question"],
                                        args.max_new_tokens, args.temperature,
                                        args.top_p, args.top_k, s)
                ok, why = scorer(txt, it["answer"])
                sc += ok
                dt_tot = dt
                rep.append({"seed": s, "ok": ok, "why": why,
                            "tokens": ntok, "sec": round(dt, 1),
                            "texte": " ".join(txt.split())[:400]})
            corrects += sc

            resultats.append({
                "id": qid, "category": it.get("category", "?"),
                "attendu": it["answer"], "score": "%d/%d" % (sc, args.n_seeds),
                "reponses": rep,
            })

            # Affichage integral: texte + verdict, comme le veut AGENTS.md
            print()
            print("  [%s] %s (%s) — attendu : %s"
                  % (etat, qid, it.get("category", "?"), it["answer"]))
            for r in rep:
                print("    seed %d | %s | %d tok | %.1f s"
                      % (r["seed"], "OK  " if r["ok"] else "FAUX",
                         r["tokens"], r["sec"]))
                print("      %s" % (r["texte"] or "(vide)"))
                print("      motif : %s" % r["why"])
            print("    -> %d/%d   (%.0f%%)"
                  % (sc, args.n_seeds, 100 * sc / args.n_seeds))

            # Ecriture INCREMENTALE a chaque item
            total_possible = len(resultats) * args.n_seeds
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "etat": etat, "model": path, "dataset": HELDOUT,
                    "n_seeds": args.n_seeds,
                    "gen_kwargs": {"max_new_tokens": args.max_new_tokens,
                                   "temperature": args.temperature,
                                   "top_p": args.top_p, "top_k": args.top_k},
                    "total_correct": corrects,
                    "total_possible": total_possible,
                    "complete": False,
                    "items": resultats,
                }, f, ensure_ascii=False, indent=2)
            log(event="item", etat=etat, id=qid, score="%d/%d" % (sc, args.n_seeds),
                corrects=corrects, idx=idx)
            print("  -> %s enregistre (%d/%d, %.1f min ecoulees)"
                  % (qid, idx, len(items), (time.time() - t_etat) / 60),
                  flush=True)

        total_possible = len(resultats) * args.n_seeds
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "etat": etat, "model": path, "dataset": HELDOUT,
                "n_seeds": args.n_seeds,
                "gen_kwargs": {"max_new_tokens": args.max_new_tokens,
                               "temperature": args.temperature,
                               "top_p": args.top_p, "top_k": args.top_k},
                "total_correct": corrects, "total_possible": total_possible,
                "complete": True, "items": resultats,
            }, f, ensure_ascii=False, indent=2)

        pct = 100 * corrects / max(total_possible, 1)
        print("\n" + "-" * 78)
        print("TOTAL %-9s : %d/%d  (%.0f %%)  en %.1f min"
              % (etat, corrects, total_possible, pct, (time.time() - t_etat) / 60))
        print("-" * 78)
        log(event="etat_fini", etat=etat, corrects=corrects,
            possible=total_possible, minutes=round((time.time() - t_etat) / 60, 1))

        global_scores[etat] = corrects
        global_scores[etat + "_n"] = total_possible
        del model

    # ---- tableau final ------------------------------------------------
    print()
    print("=" * 78)
    print("RESULTAT CONSOLIDE")
    print("=" * 78)
    print("%-12s %10s %8s" % ("etat", "score", "taux"))
    for etat, _ in ETATS:
        if etat in global_scores:
            c = global_scores[etat]
            n = global_scores.get(etat + "_n", 1)
            print("%-12s %6d/%-4d %7.0f %%"
                  % (etat, c, n, 100 * c / max(n, 1)))
    log(event="termine", resultats=global_scores)


if __name__ == "__main__":
    sys.exit(main() or 0)
