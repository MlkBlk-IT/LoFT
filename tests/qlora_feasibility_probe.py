"""qlora_feasibility_probe.py -- Mesure OU passe le temps d'un step LoRA CPU.

But: repondre par la mesure a "QLoRA 4-bit serait-il plus rapide ici ?"
Sans reponse, tout le reste est speculation.

Methode: on reproduit un step d'entrainement reel (forward + backward
LoRA) et on mesure le FLOP/s effectif, puis on compare a la capacite
GEMM reelle de la machine. Si le CPU est deja proche de son plafond
GEMM, une 4-bit via bitsandbytes (chemin dequant + matmul non-fusionne)
ne peut pas etre plus rapide : elle ajoute de la dequantisation.

Ecrit un JSON de mesures. Ne modifie rien.
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

ROOT = r"D:\LoFT"
OUT = os.path.join(ROOT, "tests", "results", "qlora_feasibility.json")


def timed(fn, warmup=1, runs=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), sum(ts) / len(ts)


def main():
    torch.manual_seed(42)
    device_info = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    print("threads:", device_info["threads"])

    # --- 1. Plafond GEMM machine (bf16 et fp32) ------------------------
    # Matrices a la forme reelle du modele: hidden 2048.
    bench = {}
    for dtype, label in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        a = torch.randn(2048, 2048, dtype=dtype)
        b = torch.randn(2048, 6144, dtype=dtype)
        with torch.no_grad():
            mn, avg = timed(lambda: a @ b, warmup=2, runs=5)
        # 2*M*N*K
        flops = 2 * 2048 * 2048 * 6144
        bench[label] = {
            "best_s": round(mn, 4),
            "gflops": round(flops / mn / 1e9, 1),
        }
        print("GEMM %s : %.1f GFLOP/s (%.4f s)" % (label, bench[label]["gflops"], mn))
        del a, b

    # --- 2. Step d'entrainement LoRA reel --------------------------------
    model_name = os.path.join(ROOT, "models", "qwen3-1.7b")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())

    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                     r=8, lora_alpha=16, lora_dropout=0.1,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Sequence representative du run: mediane 557 tokens, batch 1
    # (per_device_train_batch_size=1, grad_accum=4 cote pipeline).
    seq = 557
    ids = torch.randint(100, 140000, (1, seq))
    labels = ids.clone()

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        optimizer.step()

    mn, avg = timed(step, warmup=1, runs=2)
    # LoRA: fwd 2N + backward grads-activations 2N = 4N par token.
    # (pas de weight-grad sur les poids geles)
    flops = 4 * n_params * seq
    step_info = {
        "best_s": round(mn, 3),
        "mean_s": round(avg, 3),
        "seq_len": seq,
        "gflops_effective": round(flops / mn / 1e9, 1),
    }
    print("step LoRA : %.1f s -> %.1f GFLOP/s effectifs"
          % (mn, step_info["gflops_effective"]))

    # --- 3. Ce que QLoRA devrait changer ---------------------------------
    # Gain 4-bit = bande passante memoire, PAS nombre d'operations.
    # Cout 4-bit = dequant a chaque matmul.
    w_bytes_bf16 = n_params * 2
    w_bytes_nf4 = n_params / 2          # 4.5 bits/param avec les scales
    result = {
        "n_params": n_params,
        "trainable_params": trainable,
        "device": device_info,
        "gemm_ceiling": bench,
        "train_step": step_info,
        "weight_footprint": {
            "bf16_MB": round(w_bytes_bf16 / 1024**2, 1),
            "nf4_approx_MB": round(w_bytes_nf4 / 1024**2, 1),
            "saved_MB": round((w_bytes_bf16 - w_bytes_nf4) / 1024**2, 1),
        },
        "verdict_inputs": {
            "bitsandbytes_installed": False,
            "note": "QLoRA = NF4 + double quant + bf16 compute. "
                    "Sur CPU, bitsandbytes dequantise puis appelle un GEMM "
                    "non-fuse. Le gain vient de la bande passante, pas des FLOPs.",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nEcrit:", OUT)


if __name__ == "__main__":
    main()
