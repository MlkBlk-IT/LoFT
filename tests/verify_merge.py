"""verify_merge.py -- Prouve que le merge a REELLEMENT change les poids.

Un merge_and_unload qui reussit peut produire un modele identique au base
si le delta B@A etait negligeable. Controle: pour chaque module LoRA,
comparer (merged - base) au delta B@A attendu. Puis verifier que les
tenseurs NON cibles sont bit-identiques.

Ecrit: tests/results/verify_merge.json
"""
import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = r"D:\LoFT"
BASE = os.path.join(ROOT, "models", "qwen3-1.7b")
MERGED = os.path.join(ROOT, "models", "qwen3-1.7b-merged1")
ADAPTER = os.path.join(ROOT, "runs", "autodistill_17b", "adapter")
OUT = os.path.join(ROOT, "tests", "results", "verify_merge.json")

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def main():
    sd_a = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
    base_sd = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16).state_dict()
    merged_sd = AutoModelForCausalLM.from_pretrained(
        MERGED, dtype=torch.bfloat16).state_dict()

    rows = []
    skipped = []
    for bk in sorted(k for k in sd_a if k.endswith(".lora_B.weight")):
        prefix = bk[: -len(".lora_B.weight")]
        name = prefix.replace("base_model.model.model.", "")
        if not name.endswith(TARGETS):
            continue
        ak = prefix + ".lora_A.weight"
        key = "model." + name + ".weight"
        if ak not in sd_a or key not in base_sd or key not in merged_sd:
            skipped.append({"module": name,
                            "has_A": ak in sd_a,
                            "in_base": key in base_sd,
                            "in_merged": key in merged_sd})
            continue

        # PEFT applique un scaling a la fusion:
        #   W_merged = W_base + (lora_alpha / r) * B @ A
        # alpha=16, r=8 -> scaling=2. Sans ce facteur, le delta observe
        # parait ~2x trop grand (c'etait le bug de ce script).
        cfg = json.load(open(os.path.join(ADAPTER, "adapter_config.json"),
                             encoding="utf-8"))
        scaling = cfg["lora_alpha"] / cfg["r"]
        delta = (sd_a[bk].float() @ sd_a[ak].float()) * scaling
        obs = merged_sd[key].float() - base_sd[key].float()
        rows.append({
            "module": name,
            "scaling": scaling,
            "expected_fro": round(delta.norm().item(), 5),
            "observed_fro": round(obs.norm().item(), 5),
            "abs_err": round((obs - delta).norm().item(), 6),
        })

    out = {"base": BASE, "merged": MERGED,
           "modules_checked": len(rows), "modules_skipped": skipped}

    # Tenseurs hors cible: doivent etre strictement identiques.
    tgt_suffix = tuple(".%s.weight" % t for t in TARGETS)
    n_untouched = n_changed = 0
    for k, v in base_sd.items():
        if k.endswith(tgt_suffix) or k not in merged_sd:
            continue
        n_untouched += 1
        if not torch.equal(v, merged_sd[k]):
            n_changed += 1
    out["untouched_tensors"] = n_untouched
    out["untouched_modified"] = n_changed

    if rows:
        errs = [r["abs_err"] for r in rows]
        out["mean_abs_err"] = round(sum(errs) / len(errs), 6)
        out["max_abs_err"] = round(max(errs), 6)
        n_nonzero = sum(1 for r in rows if r["observed_fro"] > 0)
        out["modules_with_nonzero_delta"] = n_nonzero
        # Tolerance: bf16 a 7 chiffres significatifs -> W est grand (~30),
        # le stockage fait levier sur les faibles bits du delta.
        tol = 0.5 * max(r["expected_fro"] for r in rows) * 2 ** -8
        out["tolerance_rationale"] = round(tol, 5)
    else:
        out["mean_abs_err"] = None

    ok = (len(rows) > 0
          and out.get("mean_abs_err", 1e9) < (out.get("tolerance_rationale") or 1e9)
          and n_changed == 0
          and out.get("modules_with_nonzero_delta") == len(rows))
    out["verdict"] = "MERGE_OK" if ok else "MERGE_SUSPECT"

    print("modules LoRA verifies   : %d" % len(rows))
    print("modules avec delta != 0 : %d" % out.get("modules_with_nonzero_delta", 0))
    if rows:
        print("erreur abs moyenne       : %.6f" % out["mean_abs_err"])
        print("erreur abs max          : %.6f" % out["max_abs_err"])
        print("tolerance (bf16)        : %.6f" % out["tolerance_rationale"])
        print()
        print("%-40s %10s %10s %10s" % ("module", "attendu", "observe", "erreur"))
        for r in sorted(rows, key=lambda x: x["module"])[:8]:
            print("%-40s %10.4f %10.4f %10.6f"
                  % (r["module"][:40], r["expected_fro"],
                     r["observed_fro"], r["abs_err"]))
    if skipped:
        print()
        print("IGNORES (%d):" % len(skipped))
        for s in skipped[:5]:
            print("  ", s)
    print()
    print("tenseurs hors cible : %d verifies, %d modifies (doit etre 0)"
          % (n_untouched, n_changed))
    print("VERDICT : %s" % out["verdict"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Ecrit:", OUT)


if __name__ == "__main__":
    main()
