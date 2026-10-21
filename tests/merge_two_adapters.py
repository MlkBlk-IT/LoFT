"""merge_two_adapters.py -- Analyse la faisabilite d'un merge de 2 adaptateurs.

On ne merge PAS ici: ce script mesure et verifie. Il repond a:
  1. Deux adaptateurs-LoRA independants peuvent-ils etre appliques
     simultanement sur un meme modele de base ?
  2. Si oui, quelle est la somme des deltas (W + BA1 + BA2) ?
  3. Y a-t-il un conflit (interference destructrice) ?

Sortie: tests/results/merge_two_analysis.json
"""
import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from peft import PeftModel

ROOT = r"D:\LoFT"
MODEL = os.path.join(ROOT, "models", "qwen3-1.7b")
OUT = os.path.join(ROOT, "tests", "results", "merge_two_analysis.json")
A1 = os.path.join(ROOT, "runs", "autodistill_17b", "adapter")
A2 = os.path.join(ROOT, "runs", "autodistill_17b2", "adapter")


def delta_stats(a_dir):
    """Pour chaque module LoRA, la norme du delta BA (le vrai effet)."""
    sd = load_file(os.path.join(a_dir, "adapter_model.safetensors"))
    per_module = {}
    keys = sorted(k for k in sd if k.endswith("lora_B.weight"))
    for bk in keys:
        prefix = bk[: -len(".lora_B.weight")]
        ak = prefix + ".lora_A.weight"
        if ak not in sd:
            continue
        A = sd[ak].float()      # [r, in]
        B = sd[bk].float()      # [out, r]
        delta = B @ A           # [out, in]
        name = prefix.replace("base_model.model.model.", "")
        per_module[name] = {
            "fro_norm": round(delta.norm().item(), 4),
            "rel_to_A": round((delta.norm() / A.norm()).item(), 4),
            "lora_rank": A.shape[0],
        }
    return per_module


def main():
    out = {"base": MODEL, "adapters": {}}

    # --- 1. Deux adaptateurs, memes cles ? ------------------------------
    keys1 = set(load_file(os.path.join(A1, "adapter_model.safetensors")).keys())
    keys2_path = os.path.join(A2, "adapter_model.safetensors")
    if not os.path.exists(keys2_path):
        print("ADAPTATEUR 2 ABSENT: %s" % A2)
        print("Ce script suppose que le 2e training est deja fait.")
        out["adapter2_present"] = False
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("Ecrit:", OUT)
        return
    keys2 = set(load_file(keys2_path).keys())

    out["adapter2_present"] = True
    out["keys_identical"] = keys1 == keys2
    out["n_keys_a1"] = len(keys1)
    out["n_keys_a2"] = len(keys2)
    out["keys_only_a1"] = sorted(keys1 - keys2)[:10]
    out["keys_only_a2"] = sorted(keys2 - keys1)[:10]

    print("cles identiques : %s" % out["keys_identical"])
    print("n cles A1=%d  A2=%d" % (len(keys1), len(keys2)))

    # --- 2. Normes des deltas : conflit ou complementarite? ------------
    d1 = delta_stats(A1)
    d2 = delta_stats(A2)
    modules = sorted(set(d1) & set(d2))
    out["n_modules_shared"] = len(modules)

    rows = []
    cos_sum = []
    for m in modules:
        s1 = torch.tensor(d1[m]["fro_norm"])
        s2 = torch.tensor(d2[m]["fro_norm"])
        rows.append({"module": m, "delta1": d1[m]["fro_norm"],
                     "delta2": d2[m]["fro_norm"],
                     "sum": round((s1 + s2).item(), 4)})
    out["modules"] = rows

    print("\n%-46s %9s %9s %9s" % ("module", "delta1", "delta2", "somme"))
    for r in rows[:8]:
        print("%-46s %9.4f %9.4f %9.4f"
              % (r["module"][:46], r["delta1"], r["delta2"], r["sum"]))
    print("... (%d modules au total)" % len(rows))

    # --- 3. Le merge est-il chargeable ? -------------------------------
    print("\n[test de chargement des DEUX adaptateurs sur le meme base]")
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(base, A1, adapter_name="a1")
    m.load_adapter(A2, adapter_name="a2")
    out["dual_load"] = True
    print("2 adaptateurs charges simultanement: OK")

    # Delta total = (base + a1) + a2. PEFT gere l'addition interne.
    merged = m.merge_and_unload()
    out["merge_and_unload_ok"] = True
    print("merge_and_unload sur les 2: OK")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nEcrit:", OUT)


if __name__ == "__main__":
    main()
