"""normalize_heldout.py -- Normalise le held-out genere au format du training.

Deux problemes mesures sur le fichier d'origine (40 items):

  A. FORMAT: 26/40 reponses portaient une unite ('16,60 EUR', '15 km/h').
     Le training set (raisonnement_40) n'en a AUCUNE. Scorer '70' contre
     '16,60 EUR' fait echouer compare.py: son regex prend le DERNIER nombre
     de la chaine -> 60 au lieu de 16,60.

  B. HEURE ILLEGALE: q19 contenait '10:75' dans son CoT (minutes > 60).
     C'est exactement le bug deja documente dans compare.py. On corrige.

Ecrit: tests/data/heldout_40_norm.json
"""
import json
import os
import re
import unicodedata

SRC = os.path.join(os.path.expanduser("~"), "Downloads",
                   "raisonnement_40_heldout.json")
DST = r"D:\LoFT\tests\data\heldout_40_norm.json"
REPORT = r"D:\LoFT\tests\results\heldout_normalization.json"

UNIT_RE = re.compile(
    r"\s*\b(eur|euros?|minutes?|min|secondes?|sec|heures?|h|km/h|km|"
    r"km|m|cm|mm|kg|g|ans?|an)\b\.?", re.I)


def strip_units(s):
    s = str(s).strip()
    s = s.replace("€", " EUR ")
    s = UNIT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".").strip()
    return s


def fix_illegal_time(t):
    """'10:75' -> '11:15'. Renvoie (corrige, etait_illegal)."""
    h, m = t.split(":")
    h, m = int(h), int(m)
    if m < 60:
        return t, False
    total = h * 60 + m
    return "%d:%02d" % (total // 60, total % 60), True


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    items = d["items"]
    report = {"source": SRC, "dest": DST, "n_items": len(items),
              "units_stripped": [], "times_fixed": []}

    for it in items:
        # --- A. reponse sans unite ----------------------------------
        raw = str(it["answer"])
        clean = strip_units(raw)
        if clean != raw:
            report["units_stripped"].append(
                {"id": it["id"], "before": raw, "after": clean})

        # --- le CoT garde la forme lisible, on aligne juste la fin ----
        it["answer"] = clean
        it["answer_original"] = raw

        # --- B. heures illegales dans le CoT ------------------------
        for t in re.findall(r"\b(\d{1,2}:\d{2})\b", it["cot"]):
            fixed, illegal = fix_illegal_time(t)
            if illegal:
                report["times_fixed"].append(
                    {"id": it["id"], "before": t, "after": fixed})
                it["cot"] = it["cot"].replace(t, fixed)

        # --- le champ check garde son texte, mais on note --------------
        it["cot"] = it["cot"].strip()

    # --- verification finale: plus aucune unite, plus aucune heure illegale
    still_units = [it["id"] for it in items
                   if re.search(r"[a-zA-Z]", str(it["answer"]))]
    bad_times = []
    for it in items:
        for t in re.findall(r"\b(\d{1,2}):(\d{2})\b", it["cot"]):
            if int(t[1]) >= 60:
                bad_times.append({"id": it["id"], "time": "%s:%s" % t})
    rep_in_cot = [it["id"] for it in items
                  if str(it["answer"]).strip().lower()
                  not in unicodedata.normalize("NFC", it["cot"].lower())
                  and str(it["answer"]).strip().lower()
                  not in it["cot"].lower().replace(" ", "")]

    report["answers_still_alphabetic"] = still_units
    report["illegal_times_remaining"] = bad_times
    report["answer_not_in_cot"] = rep_in_cot
    report["verdict"] = ("NORM_OK"
                         if not still_units and not bad_times and not rep_in_cot
                         else "NORM_SUSPECT")

    out = {"metadata": {**d.get("metadata", {}),
                        "normalized_by": "normalize_heldout.py",
                        "source": os.path.basename(SRC)},
           "items": items}
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("items normalises   : %d" % len(items))
    print("unites retirees    : %d" % len(report["units_stripped"]))
    for r in report["units_stripped"][:8]:
        print("   %s : %-16r -> %r" % (r["id"], r["before"], r["after"]))
    if len(report["units_stripped"]) > 8:
        print("   ... %d de plus" % (len(report["units_stripped"]) - 8))
    print()
    print("heures corrigees   : %d" % len(report["times_fixed"]))
    for r in report["times_fixed"]:
        print("   %s : %s -> %s" % (r["id"], r["before"], r["after"]))
    print()
    print("reponses encore alphabetiques : %s" % (still_units or "aucune"))
    print("heures illegales restantes   : %s" % (bad_times or "aucune"))
    print("reponse absente du CoT       : %s" % (rep_in_cot or "aucune"))
    print()
    print("VERDICT : %s" % report["verdict"])
    print("Ecrit   : %s" % DST)

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
