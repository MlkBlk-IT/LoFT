"""Score horaire pour les tests LoFT.

Lit results/<tag>.json et compare chaque reponse a l'heure attendue.
NeJudge PAS par 'correct' du fichier source: ces runs n'ont pas de
champ 'correct' -- le champ 'correct' n'existe que pour les anciens tags.

Usage:
  python score_time.py overflow_fr_d2 "17:05"
  python score_time.py control_fr_d2 "16:45"
"""
import json, re, sys, os

RESULTES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Heures valides uniquement: HH:MM ou HHhMM. 16:65 est REJETE.
HEURE = re.compile(r"\b([01]?\d|2[0-3])[:h](\d{2})\b")


def heures_valides(texte):
    """Toutes les heures bien formees trouvees dans le texte."""
    return ["%02d:%02d" % (int(h), int(m)) for h, m in HEURE.findall(texte)]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    tag, attendu = sys.argv[1], sys.argv[2]

    chemin = os.path.join(RESULTES, tag + ".json")
    if not os.path.exists(chemin):
        print("INTROUVABLE: %s" % chemin)
        print("Tags disponibles: %s" % ", ".join(
            f[:-5] for f in sorted(os.listdir(RESULTES)) if f.endswith(".json")))
        sys.exit(1)

    data = json.load(open(chemin, encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        print("Pas de 'runs' dans %s" % chemin)
        sys.exit(1)

    print("tag=%s  attendu=%s  runs=%d" % (tag, attendu, len(runs)))
    print("-" * 68)
    print("%-5s %-10s %-6s %s" % ("seed", "trouve", "verdict", "toutes les heures"))
    print("-" * 68)

    n_ok = 0
    modes = {}
    for r in runs:
        ans = r.get("answer", "")
        trouvees = heures_valides(ans)
        final = trouvees[-1] if trouvees else None
        ok = final == attendu
        n_ok += ok
        modes[final or "AUCUNE"] = modes.get(final or "AUCUNE", 0) + 1
        print("%-5s %-10s %-6s %s" % (
            r.get("seed"), final or "-", "OK" if ok else "FAUX", ", ".join(trouvees) or "-"))

    print("-" * 68)
    print("SCORE: %d/%d" % (n_ok, len(runs)))
    print("Heures finales produites: %s" % json.dumps(modes, ensure_ascii=False, sort_keys=True))

    # Détection des heures non normalisées (16:65 etc.) -- l'HEURE regex
    # les rejete deja; on compte les occurrences brutes pour le diagnostic.
    brutes = re.findall(r"\b\d{1,2}[:h]\d{2,3}\b", " ".join(r.get("answer", "") for r in runs))
    douteuses = [b for b in brutes if int(re.split(r"[:h]", b)[1]) >= 60]
    print("Heures non normalisees (>=60 min): %d %s"
          % (len(douteuses), sorted(set(douteuses)) if douteuses else ""))


if __name__ == "__main__":
    main()