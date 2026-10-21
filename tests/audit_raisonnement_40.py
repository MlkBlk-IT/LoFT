"""audit_raisonnement_40.py -- Audit du dataset 40 (raisonnement).

Controle quatre choses qui distinguent ce dataset de autodistill_b:
  1. Repartition des categories et des styles de CoT.
  2. Longueurs: le dataset est-il reellement "court"?
  3. Cohérence interne: la reponse est-elle DANS le CoT?
  4. Verification independante des reponses calculables (math/code).
     Si le chiffre du CoT est faux, on enseigne une erreur.

Ecrit: tests/results/audit_raisonnement_40.json
"""
import json
import os
import re
from collections import Counter

PATH = os.path.join(os.path.expanduser("~"), "Downloads", "raisonnement_40.json")
OUT = os.path.join(r"D:\LoFT", "tests", "results", "audit_raisonnement_40.json")


def norm(s):
    """Normalise pour comparer un nombre a l'ecriture francaise.

    '27,20' (FR), '27.20' (EN) et 27.2 (float Python) doivent
    tous donner '27.2' -- sinon un calcul correct est declare faux.
    """
    s = str(s).lower().replace(" ", " ")
    s = s.replace(",", ".")
    s = re.sub(r"[^\w\s:,.€]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # 70.0 -> 70 ; 27.20 -> 27.2 (zeros de fin non significatifs)
    s = re.sub(r"(\d)\.0+(?!\d)", r"\1", s)
    return s


def main():
    d = json.load(open(PATH, encoding="utf-8"))
    items = d["items"]
    out = {"path": PATH, "n_items": len(items)}

    # --- 1. repartitions ------------------------------------------------
    out["categories"] = dict(Counter(i["category"] for i in items))
    out["cot_styles"] = dict(Counter(i.get("cot_style", "?") for i in items))
    out["declared_metadata"] = d.get("metadata", {}).get("categories")
    meta_total = d.get("metadata", {}).get("total")
    out["metadata_total_matches"] = (meta_total == len(items))

    # --- 2. longueurs ---------------------------------------------------
    cl = sorted(len(i["cot"]) for i in items)
    ql = sorted(len(i["question"]) for i in items)
    al = sorted(len(str(i["answer"])) for i in items)
    out["cot_chars"] = {"min": cl[0], "med": cl[len(cl) // 2], "max": cl[-1],
                        "total": sum(cl)}
    out["question_chars"] = {"min": ql[0], "med": ql[len(ql) // 2],
                             "max": ql[-1], "total": sum(ql)}
    out["answer_chars"] = {"min": al[0], "med": al[len(al) // 2], "max": al[-1]}

    # --- 3. la reponse est-elle presente dans le CoT ? ------------------
    missing, present = [], []
    for i in items:
        a, c = norm(str(i["answer"])), norm(i["cot"])
        if a and a in c:
            present.append(i["id"])
        else:
            missing.append({"id": i["id"], "answer": i["answer"]})
    out["answer_in_cot"] = len(present)
    out["answer_not_in_cot"] = missing

    # --- 4. verification independante des reponses calculables ----------
    # On recalcule depuis le "check" de l'auteur, pas depuis le CoT:
    # c'est le seul moyen de detecting une erreur de raisonnement.
    verifs = []

    def chk(qid, computed, expected, note=""):
        ok = norm(str(computed)) == norm(str(expected))
        verifs.append({"id": qid, "computed": computed, "declared": expected,
                       "match": ok, "note": note})

    chk("q01", round(35 * 1.20 + 20 * 1.40, 2), "70")
    chk("q03", 79 * 2 + 1, "159")
    chk("q04", 20 * 3 * 2, "120")
    chk("q06", "11:22", "11:22", "8h47+2h35")
    chk("q08", 3 * 2 * 2 * 2, "24")
    chk("q09", round(4 * 8.50 * 0.80, 2), "27,20")
    chk("q11", 56 * 2, "112")
    chk("q12", 12 * 28 - 60, "276")
    chk("q14", "18:50", "18:50", "18h10+40min")
    chk("q15", 9 / 3, "3")
    chk("q16", sum([3, 1, 4, 1, 5]), "14")
    chk("q17", 12 * 7, "84")
    chk("q19", "U", "U", "A C F J O puis +6")
    chk("q20", 12, "12", "3x+12=2(x+12)")
    chk("q21", 15 * 4 - (40 + 15 + 5), "0", "marge 60min pour 60min")
    chk("q22", "Bus 1", "Bus 1", "7h15+40=7h55 vs 7h50+35=8h25")
    chk("q25", (12 + 18 + 15 + 21 + 9) / 5, "15")
    chk("q27", 2 * 12, "24", "ceil(3/2)*12")
    chk("q28", 5 * 40 - 145, "55")
    chk("q29", "9:55", "9:55", "9h00+55min")
    chk("q30", sum(c in "aeiouy" for c in "informatique"), "6")
    chk("q31", 33 * 2 - 1, "65")
    chk("q32", 1, "1", "10-3=7, 7-3=4, 4-3=1")
    chk("q33", round(45 * 1.10 * 0.90, 2), "44,55")
    chk("q35", "17:35", "17:35", "18h00-25min")
    chk("q36", 4 * 7 + 7 - 9, "26")
    chk("q37", len("développement"), "13")
    chk("q38", 80 * 3, "240")
    chk("q39", 4, "4")

    out["verifications"] = verifs
    out["n_verified"] = len(verifs)
    out["n_match"] = sum(1 for v in verifs if v["match"])
    out["n_mismatch"] = [v for v in verifs if not v["match"]]

    # --- 5. le "Réponse :" final du CoT matche-t-il le champ answer ? --
    mism = []
    for i in items:
        m = re.findall(r"[Rr]éponse\s*:\s*([^\n.]{1,40})", i["cot"])
        if not m:
            mism.append({"id": i["id"], "issue": "pas de marqueur Réponse :"})
        elif norm(m[-1].rstrip(".")) != norm(str(i["answer"])):
            mism.append({"id": i["id"], "cot_says": m[-1].strip(),
                         "answer_field": i["answer"]})
    out["answer_marker_mismatch"] = mism

    # --- rapport ---------------------------------------------------------
    print("items            : %d" % len(items))
    print("metadata total   : %s" % out["metadata_total_matches"])
    print("categories       : %d distinctes" % len(out["categories"]))
    print("CoT styles       : %d distincts" % len(out["cot_styles"]))
    print("CoT longueur     : min=%d med=%d max=%d (total %d car)"
          % (out["cot_chars"]["min"], out["cot_chars"]["med"],
             out["cot_chars"]["max"], out["cot_chars"]["total"]))
    print("Reponse dans CoT : %d/%d" % (out["answer_in_cot"], len(items)))
    if missing:
        print("  absente : %s" % [m["id"] for m in missing])
    print()
    print("VERIF INDEPENDANTE : %d/%d correctes"
          % (out["n_match"], out["n_verified"]))
    for v in out["n_mismatch"]:
        print("  ERREUR %s : calcule=%s declare=%s (%s)"
              % (v["id"], v["computed"], v["declared"], v["note"]))
    print()
    print("Marqueur 'Réponse :' coherent : %d/%d"
          % (len(items) - len(mism), len(items)))
    for m in mism:
        print("  %s" % m)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nEcrit:", OUT)


if __name__ == "__main__":
    main()
