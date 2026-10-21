"""speed_probe.py -- Trouve la config la plus rapide pour un dataset court.

Ecriture incrementale dans tests/results/speed_probe.json.

Ce que ce script mesure, et pourquoi:
  1. Deux points de sequence (court vs long) -> pente + intercept.
     On veut savoir si le temps est PROPORTIONNEL a la longueur ou
     s'il y a un plancher fixe. Ça change le levier: si c'est
     proportionnel, raccourcir la sequence aide; si c'est un plancher,
     seul le nombre de params entraines compte.
  2. Effet r (rank) et nombre de cibles LoRA.
  3. Verification que gradient_checkpointing COUTE du temps
     (il recalcule le forward au backward).
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

ROOT = r"D:\LoFT"
OUT = os.path.join(ROOT, "tests", "results", "speed_probe.json")
MODEL = os.path.join(ROOT, "models", "qwen3-1.7b")


def save(d):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT)


def measure(model, seq, runs=2, warmup=1):
    ids = torch.randint(100, 140000, (1, seq))
    labels = ids.clone()
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    torch.manual_seed(42)
    result = {"model": MODEL, "torch": torch.__version__,
              "threads": torch.get_num_threads(), "runs": []}

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in base.parameters())

    # ---- A. pente vs longueur de sequence, config de reference --------
    cfg_ref = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                         r=8, lora_alpha=16, lora_dropout=0.1,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    for seq in (160, 300, 557):
        m = get_peft_model(base, cfg_ref)
        s = measure(m, seq)
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        rec = {"exp": "seq_scaling", "seq": seq, "step_s": round(s, 2),
               "trainable": tr, "ms_per_token": round(1000 * s / seq, 2)}
        result["runs"].append(rec)
        print("seq=%3d  step=%6.2f s  (%.2f ms/token)  trainable=%s"
              % (seq, s, rec["ms_per_token"], f"{tr:,}"))
        save(result)
        # detacher: on recharge pour chaque mesure
        m = m.unload()
        base = m

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    SEQ = 300  # representatif d'un CoT court

    # ---- B. effet rank et nombre de cibles -----------------------------
    variants = [
        ("r8_qkvo",      8, ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ("r4_qkvo",      4, ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ("r8_qv",        8, ["q_proj", "v_proj"]),
        ("r16_qkvo",    16, ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ]
    for name, r, targets in variants:
        c = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                       r=r, lora_alpha=2 * r, lora_dropout=0.1,
                       target_modules=targets)
        m = get_peft_model(base, c)
        s = measure(m, SEQ)
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        result["runs"].append({"exp": "config", "name": name, "r": r,
                               "targets": targets, "step_s": round(s, 2),
                               "trainable": tr})
        print("%-12s r=%2d  step=%6.2f s  trainable=%s"
              % (name, r, s, f"{tr:,}"))
        save(result)
        m = m.unload()

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)

    # ---- C. gradient_checkpointing: coute du temps ? ------------------
    c = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                   r=8, lora_alpha=16, lora_dropout=0.1,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    for gc in (False, True):
        m = get_peft_model(base, c)
        if gc:
            m.gradient_checkpointing_enable()
            m.enable_input_require_grads()
        s = measure(m, SEQ)
        result["runs"].append({"exp": "grad_ckpt", "enabled": gc,
                               "step_s": round(s, 2)})
        print("gradient_checkpointing=%-5s  step=%6.2f s" % (gc, s))
        save(result)
        m = m.unload()

    print("\nEcrit:", OUT)


if __name__ == "__main__":
    main()
