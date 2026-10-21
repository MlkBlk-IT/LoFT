"""verify_merge_functional.py -- Test fonctionnel du merge.

La verification par norme des tenseurs se heurte au scaling interne de
PEFT (lora_alpha / r) et a l'arrondi bf16. La seule question qui compte
pour la suite du projet est fonctionnelle:

    base + adaptateur  ==  modele mergé ?

On compare les LOGITS des trois modeles sur le meme prompt. Si les
logits coincident, le merge est correct quel que soit le detail
numerique de la representation des poids.

Ecrit: tests/results/verify_merge_functional.json
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = r"D:\LoFT"
BASE = os.path.join(ROOT, "models", "qwen3-1.7b")
MERGED = os.path.join(ROOT, "models", "qwen3-1.7b-merged1")
ADAPTER = os.path.join(ROOT, "runs", "autodistill_17b", "adapter")
OUT = os.path.join(ROOT, "tests", "results", "verify_merge_functional.json")

PROMPTS = [
    "Une cantine achète 4 packs de jus à 6,25 € le pack et 3 plateaux "
    "de fruits à 8,40 €. Elle paie avec deux billets de 50 €. "
    "Combien reçoit-elle de monnaie ?",
    "Calcule 17 × 23 puis explique la méthode.",
]


def logits_for(model, tok, prompt):
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    return out.logits[0, -1].float()


def main():
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(BASE)
    result = {"base": BASE, "merged": MERGED, "adapter": ADAPTER,
              "prompts": PROMPTS, "results": []}

    def cmp(tag_a, la, tag_b, lb, prompt):
        d = (la - lb).abs()
        return {
            "prompt": prompt[:60],
            "a": tag_a, "b": tag_b,
            "max_abs_logit_diff": round(d.max().item(), 6),
            "mean_abs_logit_diff": round(d.mean().item(), 6),
            "argmax_equal": bool(la.argmax().item() == lb.argmax().item()),
            "top5_overlap": len(set(la.topk(5).indices.tolist())
                                 & set(lb.topk(5).indices.tolist())),
        }

    for p in PROMPTS:
        # --- 1. base nu
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).eval()
        lb = logits_for(m, tok, p)
        del m

        # --- 2. base + adaptateur (non merge)
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(m, ADAPTER).eval()
        la = logits_for(m, tok, p)
        del m

        # --- 3. modele mergé
        m = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.bfloat16).eval()
        lm = logits_for(m, tok, p)
        del m

        r1 = cmp("base+adapter", la, "merged", lm, p)
        r2 = cmp("base", lb, "merged", lm, p)
        result["results"].append({"peft_vs_merged": r1, "base_vs_merged": r2})
        print("prompt: %s..." % p[:55])
        print("   base+adapter vs merged : max|dlogit|=%.6f argmax=%s top5=%d/5"
              % (r1["max_abs_logit_diff"], r1["argmax_equal"], r1["top5_overlap"]))
        print("   base         vs merged : max|dlogit|=%.6f argmax=%s top5=%d/5"
              % (r2["max_abs_logit_diff"], r2["argmax_equal"], r2["top5_overlap"]))

    # Verdict sur la seule comparaison qui compte.
    worst = max(r["peft_vs_merged"]["max_abs_logit_diff"]
                for r in result["results"])
    same_top = all(r["peft_vs_merged"]["top5_overlap"] >= 4
                   for r in result["results"])
    # Le merge DOIT differer du base nu: c'est la preuve que le delta a ete
    # applique. S'il etait identique, le merge n'aurait rien fait.
    differs_from_base = max(r["base_vs_merged"]["max_abs_logit_diff"]
                            for r in result["results"])
    result["max_diff_peft_vs_merged"] = round(worst, 6)
    result["max_diff_base_vs_merged"] = round(differs_from_base, 6)
    result["verdict"] = ("MERGE_OK"
                         if worst < 0.05 and same_top and differs_from_base > 1e-3
                         else "MERGE_SUSPECT")

    print()
    print("max |logit diff| base+adapter vs merged : %.6f" % worst)
    print("max |logit diff| base         vs merged : %.6f"
          % differs_from_base)
    print("VERDICT : %s" % result["verdict"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Ecrit:", OUT)


if __name__ == "__main__":
    main()
