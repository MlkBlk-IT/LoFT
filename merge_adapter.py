"""merge_adapter.py -- Merge UN adaptateur LoRA dans le modele de base.

Sequentiel par design: le resultat devient le BASE du training suivant.
On ne peut pas defaire un merge proprement (docs PEFT), donc le modele
d'origine n'est jamais modifie: ecriture dans un NOUVEAU dossier.

dfloat16 explicite: sans lui, from_pretrained peut materialiser en fp32
et le training suivant devient ~2x plus lent.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Modele de base (lecture seule)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output_dir", required=True, help="NOUVEAU dossier")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    # Garde-fou: jamais d'ecriture dans le dossier de base.
    base_abs = os.path.abspath(args.base)
    out_abs = os.path.abspath(args.output_dir)
    if out_abs == base_abs:
        raise SystemExit("ABORT: output_dir identique au dossier de base.")
    if out_abs.startswith(base_abs + os.sep):
        raise SystemExit("ABORT: output_dir est DANS le dossier de base.")

    # Refus d'ecraser un dossier deja rempli.
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        raise SystemExit(
            "ABORT: %s contient deja un modele. Choisis un autre "
            "output_dir (l'existant n'est pas touche)." % args.output_dir)

    print("=" * 66)
    print("MERGE LoRA")
    print("=" * 66)
    print("  base     :", args.base)
    print("  adapter  :", args.adapter)
    print("  output   :", args.output_dir)
    print("  dtype    :", args.dtype)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    t_load = time.time() - t0
    n_base = sum(p.numel() for p in model.parameters())
    print("  params base : %s | chargement %.1f s" % (f"{n_base:,}", t_load))

    t1 = time.time()
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    t_merge = time.time() - t1
    print("  merge_and_unload : %.1f s" % t_merge)

    t2 = time.time()
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tok.save_pretrained(args.output_dir)
    t_save = time.time() - t2

    # Verification: le delta doit avoir disparu (plus de lora_A/B).
    leftovers = [n for n, _ in merged.named_parameters()
                 if "lora_" in n.lower()]
    state = merged.state_dict()
    n_saved = sum(v.numel() for v in state.values() if v.dtype.is_floating_point)

    manifest = {
        "base": args.base,
        "adapter": args.adapter,
        "output_dir": args.output_dir,
        "dtype": args.dtype,
        "n_params": n_base,
        "n_params_after_merge": n_saved,
        "lora_modules_left": len(leftovers),
        "load_seconds": round(t_load, 1),
        "merge_seconds": round(t_merge, 1),
        "save_seconds": round(t_save, 1),
        "total_seconds": round(time.time() - t0, 1),
        "output_mb": round(sum(
            os.path.getsize(os.path.join(args.output_dir, f))
            for f in os.listdir(args.output_dir)
            if os.path.isfile(os.path.join(args.output_dir, f))) / 1024**2, 1),
        "versions": {"torch": torch.__version__},
    }
    with open(os.path.join(args.output_dir, "merge_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 66)
    print("RESULTATS")
    print("=" * 66)
    print("  chargement        : %.1f s" % t_load)
    print("  merge_and_unload  : %.1f s" % t_merge)
    print("  ecriture          : %.1f s" % t_save)
    print("  TOTAL             : %.1f s" % manifest["total_seconds"])
    print("  taille sortie     : %.1f MB" % manifest["output_mb"])
    print("  modules LoRA restants : %d  (doit etre 0)" % len(leftovers))
    print("  manifeste         : %s"
          % os.path.join(args.output_dir, "merge_manifest.json"))


if __name__ == "__main__":
    main()
