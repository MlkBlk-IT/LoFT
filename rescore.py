"""rescore.py -- Recalcule les scores d'un compare_*.json SANS regénérer.

Pourquoi ça existe : le scorer de compare.py a été corrigé le 2026-10-03
(normalisation virgule / point décimal). Les runs lancés AVANT cette
correction ont des scores faux négatifs — le modèle répondait "49.80 euros"
et était noté FAUX alors que la réponse était correcte.

Ce script relit les réponses déjà stockées dans le JSON et recalcule les
scores avec la version actuelle du scorer. Zéro génération.

Usage (PowerShell):
  cd D:\\LoFT
  python rescore.py --tag v1_s1
  python rescore.py --tag v1_s1 --apply     # réécrit le JSON
"""

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as C
importlib.reload(C)

RESULTATS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "tests", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Réécrit le JSON avec les nouveaux scores.")
    args = ap.parse_args()

    path = os.path.join(RESULTATS, "compare_%s.json" % args.tag)
    if not os.path.exists(path):
        print("INTROUVABLE: %s" % path)
        print("Tags présents: %s" % ", ".join(
            sorted(f[8:-5] for f in os.listdir(RESULTATS)
                   if f.startswith("compare_") and f.endswith(".json"))))
        sys.exit(1)

    d = json.load(open(path, encoding="utf-8"))
    items = d.get("items", [])
    if not items:
        print("Aucun item dans le JSON.")
        sys.exit(1)

    print("=" * 92)
    print("RESCORE  tag=%s  (%d items)" % (args.tag, len(items)))
    print("=" * 92)
    print("%-5s %-14s %-14s %-9s %-9s" % ("id", "attendu", "cat", "base", "ft"))
    print("-" * 92)

    b_tot = f_tot = 0
    n_b = n_f = 0
    changes = []

    for it in items:
        att = it["attendu"]

        b_ok = sum(1 for t in it.get("base_reponses", [])
                   if C.scorer(t, att)[0])
        b_n = len(it.get("base_reponses", []))
        b_tot += b_ok
        n_b += b_n

        reps = it.get("ft_reponses", [])
        f_ok = sum(1 for t in reps if C.scorer(t, att)[0])
        f_n = len(reps)
        f_tot += f_ok
        n_f += f_n

        ancien_b = it.get("base_score")
        ancien_f = it.get("ft_score")
        nouveau_b = "%d/%d" % (b_ok, b_n) if b_n else None
        nouveau_f = ("%d/%d" % (f_ok, f_n)) if f_n else None

        if ancien_b != nouveau_b or ancien_f != nouveau_f:
            changes.append((it["id"], ancien_b, nouveau_b, ancien_f, nouveau_f))

        it["base_score"] = nouveau_b
        it["ft_score"] = nouveau_f
        it["base_reponses_notees"] = [
            {"score": "OK" if C.scorer(t, att)[0] else "FAUX",
             "raison": C.scorer(t, att)[1], "texte": t}
            for t in it.get("base_reponses", [])]
        it["ft_reponses_notees"] = [
            {"score": "OK" if C.scorer(t, att)[0] else "FAUX",
             "reason": C.scorer(t, att)[1], "texte": t}
            for t in reps]

        print("%-5s %-14s %-14s %-9s %-9s" % (
            it["id"], str(att)[:13], it.get("category", "?")[:13],
            nouveau_b or "-", nouveau_f or "-"))

    print("-" * 92)
    print("BASE : %d/%d (%.0f %%)" % (b_tot, n_b, 100 * b_tot / n_b if n_b else 0))
    if n_f:
        print("FT   : %d/%d (%.0f %%)" % (f_tot, n_f, 100 * f_tot / n_f if n_f else 0))
        print("DELTA: %+d" % (f_tot - b_tot))

    if changes:
        print()
        print("SCORES CORRIGES (%d items change(s)):" % len(changes))
        for qid, ab, nb, af, nf in changes:
            print("  %-5s base %s -> %s   |   ft %s -> %s"
                  % (qid, ab, nb, af, nf))
    else:
        print()
        print("Aucun score change (le JSON etait deja correct).")

    # items corrompus volontairement
    errs = d.get("erreurs_volontaires") or []
    if not errs and isinstance(d.get("dataset"), str):
        try:
            src = json.load(open(d["dataset"], encoding="utf-8"))
            errs = src.get("metadata", {}).get("erreurs_volontaires", [])
        except Exception:
            pass
    if errs:
        print()
        print("ITEMS CORROMPUS VOLONTAIREMENT:")
        for e in errs:
            faux = str(e.get("reponse_fausse"))
            for it in items:
                if it["id"] != e["id"]:
                    continue
                mentions = sum(1 for r in it.get("ft_reponses_notees", [])
                               if faux in "".join(r["texte"].split()))
                n = len(it.get("ft_reponses", []))
                etat = ("MEMORISE" if mentions and mentions >= n - 1
                        else "partiellement" if mentions else "non memorise")
                print("  %s  entrainé sur '%s' (vrai: %s) -> ft le sort %d/%d fois  [%s]"
                      % (it["id"], faux, e.get("vraie_reponse"), mentions, n, etat))

    if args.apply:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print()
        print("JSON réécrit :", path)
    else:
        print()
        print("(dry-run — ajoute --apply pour réécrire le JSON)")


if __name__ == "__main__":
    main()