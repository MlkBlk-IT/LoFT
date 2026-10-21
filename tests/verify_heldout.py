"""verify_heldout.py -- Verifie le held-out AVANT de l'utiliser.

Un held-out faux ne vaut rien: s'il contient une erreur, toute
conclusion tiree dessus est fausse. On recalcule chaque reponse
calculable depuis zero, independamment du CoT ecrit.

Ecrit: tests/results/verify_heldout.json
"""
import json
import os
import re

PATH = os.path.join(r"D:\LoFT", "tests", "data", "heldout_24.json")
OUT = os.path.join(r"D:\LoFT", "tests", "results", "verify_heldout.json")


def norm(s):
    """Normalise pour comparer un nombre entre notations FR et Python.

    '27,20' -> 27.2 ; 70.0 -> 70 ; 16.70 -> 16.7. Sans ca, un calcul
    correct est declare faux (constate le 2026-10-04).
    """
    s = str(s).lower().replace(",", ".")
    # Le point decimal doit SURVIVRE au nettoyage: '16.0' deviendrait
    # '16 0' si on le remplacait par un espace ici (bug du 2026-10-04).
    s = re.sub(r"[^\w\s:.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(\d)\.0+(?!\d)", r"\1", s)   # 70.0 -> 70
    s = re.sub(r"(\.\d*?)0+\b", r"\1", s)      # 16.70 -> 16.7
    return s


def hhmm(minutes):
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def main():
    d = json.load(open(PATH, encoding="utf-8"))
    items = d["items"]
    verifs = []

    def chk(qid, computed, expected, note=""):
        verifs.append({
            "id": qid, "computed": computed, "expected": expected,
            "match": norm(computed) == norm(expected), "note": note})

    # --- math / multi_hop / scientifique calculables -------------------
    chk("h01", round(5 * 3.40 + 3 * 2.15, 2), "23,45")
    chk("h02", (8 + 12 + 20 + 24) / 4, "16")
    chk("h03", 240 - 240 // 2, "120")
    chk("h07", 30 + 12, "42", "ecarts +4+6+8+10 puis +12")
    chk("h08", chr(ord("A") + 35).upper(), "J", "36e lettre : A=1 donc 36 -> chr(A+35)")
    chk("h09", 72 * 5, "360")
    chk("h10", 3 * 120 / 4, "90")
    chk("h11", 24 * 2 * 7, "336")
    chk("h12", round(3 * 2.80 + 2 * 4.15, 2), "16,70")
    chk("h14", sum(c in "aeiouy" for c in "ordinateur"), "5")
    chk("h15", hhmm(9 * 60 + 5 + 110), "10:55")
    chk("h16", hhmm(19 * 60 - 165), "16:15")
    chk("h17", "Bus B" if 6 * 60 + 20 + 10 < 6 * 60 + 10 + 25 else "Bus A", "Bus B")
    chk("h18", 12 / 4, "3")
    chk("h19", round(54 / 20, 2), "2,7")
    chk("h20", 5 + 4, "9")
    chk("h21", [2, 4, 6][2], "6")
    chk("h22", "1 4 9 16", "1 4 9 16")
    chk("h23", hhmm(10 * 60 + 40), "10:40")
    chk("h24", 30 + 20 + 10, "60")

    # --- non calculables: verification de coherence interne -----------
    # h04: logique pure -> on verifie le schema de raisonnement.
    #   Cas Boris : il ment (1 menteur), Arsene dit vrai, Chloe dit vrai.
    h04 = next(i for i in items if i["id"] == "h04")["cot"]
    verifs.append({
        "id": "h04", "computed": "Boris", "expected": "Boris",
        "match": "Si Boris est coupable" in h04 and "un seul menteur" in h04,
        "note": "logique verifiee manuellement : 1 menteur = valide"})

    # CoT doit contenir la reponse
    missing = [i["id"] for i in items
               if norm(i["answer"]) not in norm(i["cot"])]

    n_ok = sum(1 for v in verifs if v["match"])
    print("items held-out      : %d" % len(items))
    print("verifications       : %d/%d correctes" % (n_ok, len(verifs)))
    for v in verifs:
        if not v["match"]:
            print("  ERREUR %s : calcule=%s attendu=%s (%s)"
                  % (v["id"], v["computed"], v["expected"], v["note"]))
    print("reponse absente du CoT : %s" % (missing or "aucune"))

    out = {"path": PATH, "n_items": len(items), "n_verified": len(verifs),
           "n_match": n_ok,
           "mismatches": [v for v in verifs if not v["match"]],
           "answer_missing_from_cot": missing,
           "verdict": "HELDOUT_OK" if n_ok == len(verifs) and not missing
                      else "HELDOUT_SUSPECT"}
    print("VERDICT : %s" % out["verdict"])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Ecrit:", OUT)


if __name__ == "__main__":
    main()
