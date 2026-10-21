"""compare.py -- Evaluation avant / apres fine-tuning.

Charge le modele de base puis le meme modele + adaptateur LoRA, pose les
memes questions, et affiche les reponses cote a cote. C'est la seule
metrique qui compte: est-ce que le Student a appris le raisonnement, ou
a-t-il seulement memorise les reponses ?

Usage (PowerShell):
  cd D:\\LoFT
  .\\venv\\Scripts\\Activate.ps1

  python compare.py --dataset tests/data/dataset_v1.json `
    --base models/qwen3-0.6b --adapter runs/v1_qwen06b/adapter `
    --tag v1 --n_seeds 3

Sortie: D:\\LoFT\\tests\\results\\compare_<tag>.json + tableau console.
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

RESULTATS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "tests", "results")


# ---------------------------------------------------------------- scoring
def _norm(s):
    """Minuscules, accents retires, ponctuation en espaces.

    On remplace TOUS les separateurs de decimales par un espace: '49,80',
    '49.80' et '49 80' doivent tous produire '49 80', sinon le modele qui
    repond en format anglais (point) est note FAUX a tort. Bug observe le
    2026-10-03 sur q01.
    """
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s:]", " ", s)


def _variantes(att):
    """Toutes les facons d'ecrire la meme valeur attendue."""
    v = {att}
    v.add(att.replace(",", "."))
    v.add(att.replace(",", " "))
    v.add(att.replace(".", ","))
    v.add(att.replace(".", " "))
    v.add(att.replace(":", "h"))
    v.add(att.replace(" ", ""))
    v.add(att.upper())
    return {x for x in v if x}


def reponse_extraite(texte):
    """Derniere valeur citee -- la conclusion, pas l'enumeration."""
    # 1. "Reponse : X"
    m = re.findall(r"r[ée]ponse\s*[:\-]?\s*\**\s*([^\n.,;]{1,30})", texte, re.I)
    if m:
        return m[-1].strip().strip("*")
    # 2. dernier nombre (avec heures si present)
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

    # sortie exacte
    if norm.strip() == att:
        return True, "exact"

    # la valeur attendue apparait (formats FR / EN / espace tolérés)
    variants = _variantes(att)
    for v in variants:
        if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(v), norm):
            return True, "contenu (%s)" % v

    # tolerance numerique (<= 1%), virgule ou point
    chiffres = re.findall(r"-?\d+(?:[.,]\d+)?", norm)
    try:
        cible = float(att.replace(",", "."))
        for c in chiffres:
            v = float(c.replace(",", "."))
            if abs(cible) > 1e-9 and abs(v - cible) / abs(cible) <= 0.01:
                return True, "numerique ~%s" % v
    except ValueError:
        pass

    # tolerance "lettre isolee" pour les categories A/B/C
    if len(att) <= 3:
        for m in re.findall(r"(?<![a-z])([a-z])(?![a-z])", norm):
            if m == att:
                return True, "lettre isolee %s" % m

    return False, "manque"


# ---------------------------------------------------------------- generation
def generer(model, tok, question, max_new_tokens, temperature, top_p, top_k,
            seed, enable_thinking):
    torch.manual_seed(seed)
    kwargs = dict(add_generation_prompt=True, tokenize=True,
                  return_dict=True, return_tensors="pt")
    # Qwen3: le template doit toujours recevoir la cle, sinon 5.x leve.
    kwargs["enable_thinking"] = enable_thinking
    enc = tok.apply_chat_template(
        [{"role": "user", "content": question}], **kwargs)
    n_in = enc["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,
        )
    dt = time.perf_counter() - t0
    txt = tok.decode(out[0, n_in:], skip_special_tokens=True)
    return txt, int(out.shape[1] - n_in), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--enable_thinking", type=int, choices=[0, 1], default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.dataset, encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    if args.limit:
        items = items[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    base.eval()

    print("=" * 100)
    print("MODELE      :", args.base)
    print("ADAPTATEUR  :", args.adapter or "(aucun)")
    print("DATASET     :", args.dataset, "-", len(items), "items")
    print("SEEDS       :", args.n_seeds, "| thinking:", bool(args.enable_thinking))
    print("=" * 100)
    print()

    # Ecriture INCREMENTALE: si le run plante (Ctrl+C, OOM, perte de
    # session), tout ce qui est deja genere reste dans le JSON. Avant,
    # le fichier n'etait ecrit qu'a la fin -> une interruption a q18
    # perdait 30 generations.
    os.makedirs(RESULTATS, exist_ok=True)
    out_path = os.path.join(RESULTATS, "compare_%s.json" % args.tag)
    base_reponses, ft_reponses = {}, {}

    def sauver():
        partiels = []
        b_tot = f_tot = 0
        for it in items:
            qid = it["id"]
            if qid not in base_reponses:
                continue
            b_sc = sum(1 for t in base_reponses[qid]
                       if scorer(t, it["answer"])[0])
            b_tot += b_sc
            f_sc = sum(1 for t in ft_reponses.get(qid, [])
                       if scorer(t, it["answer"])[0])
            f_tot += f_sc
            partiels.append({
                "id": qid, "category": it.get("category", "?"),
                "attendu": it["answer"],
                "base_score": "%d/%d" % (b_sc, args.n_seeds),
                "ft_score": ("%d/%d" % (f_sc, args.n_seeds))
                            if qid in ft_reponses else None,
                "base_reponses": base_reponses[qid],
                "ft_reponses": ft_reponses.get(qid, []),
            })
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "base": args.base, "adapter": args.adapter,
                "dataset": args.dataset, "tag": args.tag,
                "gen_kwargs": {
                    "n_seeds": args.n_seeds, "temperature": args.temperature,
                    "top_p": args.top_p, "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                    "enable_thinking": bool(args.enable_thinking),
                },
                "base_total": "%d/%d" % (b_tot, len(partiels) * args.n_seeds),
                "ft_total": ("%d/%d" % (f_tot, len(partiels) * args.n_seeds))
                            if ft_reponses else None,
                "items": partiels,
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)   # atomique: jamais de JSON tronque

    def afficher(it, phase, txt):
        """Montre le texte genere + le verdict, en direct."""
        ok, why = scorer(txt, it["answer"])
        print()
        print("  %s [%s] %s" % (phase, it["id"], it.get("category", "?")))
        print("    attendu : %s" % it["answer"])
        print("    genere  : %s" % (" ".join(txt.split())[:300] or "(vide)"))
        print("    verdict : %s (%s)" % ("OK" if ok else "FAUX", why))

    # --- PASSE 1: modele de base (avant) ---------------------------------
    # PeftModel.from_pretrained modifie le modele EN PLACE (il remplace
    # chaque Linear par un module LoRA), donc on ne peut pas revenir en
    # arriere: deux passes sequentielles, pas de restauration d'etat.
    print("=" * 100)
    print("PASSE 1/2 - modele de BASE (%d generations, ~40 s chacune)"
          % (len(items) * args.n_seeds))
    print("=" * 100)
    for it in items:
        vals = []
        for s in range(args.n_seeds):
            txt, _, _ = generer(base, tok, it["question"], args.max_new_tokens,
                                args.temperature, args.top_p, args.top_k, s,
                                bool(args.enable_thinking))
            vals.append(txt)
            afficher(it, "base s%d" % s, txt)
        base_reponses[it["id"]] = vals
        sauver()
        print("  -> %s enregistre" % it["id"], flush=True)

    # --- PASSE 2: modele + adaptateur (apres) ---------------------------
    if args.adapter:
        print()
        print("=" * 100)
        print("PASSE 2/2 - modele + ADAPTATEUR")
        print("=" * 100)
        base = PeftModel.from_pretrained(base, args.adapter)
        base.eval()
        for it in items:
            vals = []
            for s in range(args.n_seeds):
                txt, _, _ = generer(base, tok, it["question"],
                                    args.max_new_tokens, args.temperature,
                                    args.top_p, args.top_k, s,
                                    bool(args.enable_thinking))
                vals.append(txt)
                afficher(it, "ft s%d" % s, txt)
            ft_reponses[it["id"]] = vals
            sauver()
            print("  -> %s enregistre" % it["id"], flush=True)
        base = base.merge_and_unload()
        base.eval()
        print()
        print("   adaptateur retire (merge_and_unload)")
        sauver()

    print()
    entetes = "%-4s %-14s %-12s %-7s %-7s" % (
        "id", "attendu", "categorie", "base", "ft")
    print(entetes)
    print("-" * 100)

    resultats = []
    base_ok = ft_ok = 0

    for it in items:
        qid = it["id"]
        att = it["answer"]
        cat = it.get("category", "?")

        b_ok = sum(1 for t in base_reponses[qid] if scorer(t, att)[0])
        f_ok = (sum(1 for t in ft_reponses[qid] if scorer(t, att)[0])
                if args.adapter else 0)

        base_ok += b_ok
        ft_ok += f_ok
        print("%-4s %-14s %-12s %-7s %-7s" % (
            qid, str(att)[:13], cat[:11],
            "%d/%d" % (b_ok, args.n_seeds),
            ("%d/%d" % (f_ok, args.n_seeds)) if args.adapter else "-"))

        resultats.append({
            "id": qid, "category": cat, "attendu": att,
            "base_score": "%d/%d" % (b_ok, args.n_seeds),
            "ft_score": ("%d/%d" % (f_ok, args.n_seeds)) if args.adapter else None,
            "base_reponses": base_reponses[qid],
            "ft_reponses": ft_reponses.get(qid, []),
        })

    total = len(items) * args.n_seeds
    print("-" * 100)
    print("TOTAL base : %d/%d (%.0f %%)" % (base_ok, total, 100 * base_ok / total))
    if args.adapter:
        print("TOTAL ft   : %d/%d (%.0f %%)" % (ft_ok, total, 100 * ft_ok / total))
        delta = ft_ok - base_ok
        print("DELTA      : %+d  (%s)" % (delta, "amelioration" if delta > 0
                                          else "identique" if delta == 0 else "regression"))

    # --- cas particulier: les items corrompus ---
    errs = data.get("metadata", {}).get("erreurs_volontaires", []) \
        if isinstance(data, dict) else []
    if errs and args.adapter:
        print()
        print("=" * 100)
        print("ITEMS CORROMPUS VOLONTAIREMENT (apprentissage par coeur attendu)")
        print("=" * 100)
        ids = {e["id"] for e in errs}
        for r in resultats:
            if r["id"] in ids:
                e = [x for x in errs if x["id"] == r["id"]][0]
                faux = str(e["reponse_fausse"])
                mentions = [t for t in r.get("ft_reponses", [])
                            if faux in re.sub(r"\s", "", t)]
                print("  %s  Entraine sur '%s' (faux) | ft le mentionne : %d/%d  -> %s"
                      % (r["id"], faux, len(mentions), args.n_seeds,
                         "MEMORISE" if len(mentions) >= args.n_seeds - 1
                         else "partiellement" if mentions else "non memorise"))

    os.makedirs(RESULTATS, exist_ok=True)
    out = os.path.join(RESULTATS, "compare_%s.json" % args.tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "base": args.base, "adapter": args.adapter,
            "dataset": args.dataset, "tag": args.tag,
            "gen_kwargs": {
                "n_seeds": args.n_seeds, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "enable_thinking": bool(args.enable_thinking),
            },
            "base_total": "%d/%d" % (base_ok, total),
            "ft_total": ("%d/%d" % (ft_ok, total)) if args.adapter else None,
            "items": resultats,
        }, f, ensure_ascii=False, indent=2)
    print()
    print("Ecrit :", out)


if __name__ == "__main__":
    main()