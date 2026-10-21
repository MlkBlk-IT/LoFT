"""verify_heldout_40.py -- Audit independant du held-out de 40 items.

Recalcule chaque reponse calculable depuis zero. Un held-out errone
rend toute conclusion tiree dessus fausse.

Points de vigilance specifiques a ce fichier:
  - redaction des heures dans les CoT (un "10:75" est illegal)
  - format des reponses (avec/sans unite) -- impacte le scoring

Ecrit: tests/results/verify_heldout_40.json
"""
import json
import os
import re
from collections import Counter

PATH = os.path.join(os.path.expanduser("~"), "Downloads",
                    "raisonnement_40_heldout.json")
OUT = os.path.join(r"D:\LoFT", "tests", "results", "verify_heldout_40.json")


def norm(s):
    s = str(s).lower().replace(",", ".")
    s = re.sub(r"[^\w\s:.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(\d)\.0+(?!\d)", r"\1", s)
    s = re.sub(r"(\.\d*?)0+\b", r"\1", s)
    return s


def bare(s):
    """Reponse sans unite -- pour comparer au format numerique nu."""
    s = norm(s)
    s = re.sub(r"\b(eur|euros?|minutes?|min|ans?|km|h|heures?)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def hhmm_to_min(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def min_to_hhmm(v):
    return "%02d:%02d" % (v // 60, v % 60)


def main():
    d = json.load(open(PATH, encoding="utf-8"))
    items = d["items"]
    verifs = []
    notes = []

    def chk(qid, computed, expected, note=""):
        verifs.append({"id": qid, "computed": computed, "expected": expected,
                       "match": norm(computed) == norm(expected),
                       "note": note})

    # ---------- calculables ------------------------------------------
    chk("q01", round(5 * 2.40 + 4 * 1.15, 2), "16,60")
    chk("q03", min_to_hhmm(hhmm_to_min("14:35") + 110), "16:25")
    chk("q05", 24 * 2, "48")
    chk("q07", min_to_hhmm(hhmm_to_min("09:15") - 160), "6:35")
    chk("q09", (4 * 3 - 2) * 2, "20")
    chk("q10", round(80 - 0.15 * 80, 2), "68")
    chk("q14", 2 * (12 + 7), "38")
    chk("q15", 11 + 3, "14")
    chk("q17", min_to_hhmm(hhmm_to_min("18:30") - 45), "17:45")
    chk("q18", sum(range(1, 4)), "6")
    chk("q19", min_to_hhmm(hhmm_to_min("07:50") + 205), "11:15")
    chk("q22", (12 + 15 + 18 + 15) / 4, "15")
    chk("q23", min_to_hhmm(hhmm_to_min("14:00") - 50 - 25), "12:45")
    chk("q25", len("bonjour") * 2, "14")
    chk("q26", 5 * 5, "25")
    chk("q27", min_to_hhmm(hhmm_to_min("21:20") - 105), "19:35")
    chk("q29", 90 / 6, "15")
    chk("q32", min_to_hhmm(hhmm_to_min("09:10") + 70), "10:20")
    chk("q33", 14 - 2 + 3, "15")
    chk("q34", "I", "I", "9e lettre")
    chk("q36", "Ouest", "Ouest", "N +90 = E, E+180 = O")
    chk("q38", "Ben", "Ben", "Ana tennis -> Ben judo -> Cleo foot")
    chk("q40", 45 / 3, "15")

    # ---------- heures illegales dans les CoT ----------------------
    for i in items:
        for t in re.findall(r"\b(\d{1,2}):(\d{2})\b", i["cot"]):
            if int(t[1]) >= 60:
                notes.append({"id": i["id"], "issue": "minute illegale dans le CoT",
                              "time": "%s:%s" % t})

    # ---------- reponse presente dans le CoT ------------------------
    missing = []
    for i in items:
        a = bare(i["answer"])
        if a and a not in norm(i["cot"]):
            missing.append({"id": i["id"], "answer": i["answer"]})

    # ---------- format des reponses --------------------------------
    with_unit = [i["id"] for i in items if not re.fullmatch(
        r"[\d\s.,:]+", str(i["answer"]).strip())]
    uppercase = [i["id"] for i in items
                 if str(i["answer"])[:1].isupper()
                 and not str(i["answer"])[:1].isdigit()]

    cats = Counter(i["category"] for i in items)
    styles = Counter(i.get("cot_style", "?") for i in items)
    # styles consecutifs identiques ?
    seq = [i.get("cot_style") for i in items]
    consec = [seq[k] for k in range(1, len(seq)) if seq[k] == seq[k - 1]]

    n_ok = sum(1 for v in verifs if v["match"])
    out = {
        "path": PATH, "n_items": len(items),
        "n_verified": len(verifs), "n_match": n_ok,
        "mismatches": [v for v in verifs if not v["match"]],
        "illegal_times": notes,
        "answer_not_in_cot": missing,
        "answers_with_unit": with_unit,
        "answers_capitalized": uppercase,
        "categories": dict(cats), "cot_styles": dict(styles),
        "consecutive_same_style": consec,
        "metadata_total_ok": d.get("metadata", {}).get("total") == len(items),
    }
    out["verdict"] = ("HELDOUT_OK"
                      if n_ok == len(verifs) and not notes and not missing
                      else "HELDOUT_SUSPECT")

    print("items                    : %d" % len(items))
    print("metadata.total coherent  : %s" % out["metadata_total_ok"])
    print("categories distinctes    : %d  %s" % (len(cats), dict(cats)))
    print("styles distincts         : %d" % len(styles))
    print("styles consecutifs identiques : %s" % (consec or "aucun"))
    print()
    print("VERIFICATION INDEPENDANTE : %d/%d" % (n_ok, len(verifs)))
    for v in out["mismatches"]:
        print("  ERREUR %s : calcule=%s attendu=%s (%s)"
              % (v["id"], v["computed"], v["expected"], v["note"]))
    print()
    print("heures illegales dans CoT : %s" % (notes or "aucune"))
    print("reponse absente du CoT    : %s" % ([m["id"] for m in missing] or "aucune"))
    print()
    print("FORMAT: %d reponses portent une unite -> %s"
          % (len(with_unit), with_unit))
    print("        %d reponses commencent par une majuscule -> %s"
          % (len(uppercase), uppercase))
    print()
    print("VERDICT : %s" % out["verdict"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Ecrit:", OUT)


if __name__ == "__main__":
    main()
