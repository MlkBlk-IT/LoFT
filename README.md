<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:e10600,50:1a0505,100:00ff9c&height=140&section=header&text=LoFT&fontSize=42&fontColor=e10600&fontAlignY=32&desc=Local%20Fine-Tuning%20on%20CPU%20%E2%80%94%20LoRA%20on%20Qwen3%20with%20dataset%20distillation&descAlignY=53&descSize=15&descColor=00ff9c" width="100%" alt="LoFT" />

<img src="https://img.shields.io/badge/PyTorch-e10600?style=for-the-badge&logo=pytorch&logoColor=e10600&labelColor=0d0d0d" /> <img src="https://img.shields.io/badge/PEFT%20%C2%B7%20LoRA-00ff9c?style=for-the-badge&labelColor=0d0d0d" /> <img src="https://img.shields.io/badge/Qwen3-3776ab?style=for-the-badge&labelColor=0d0d0d" /> <img src="https://img.shields.io/badge/100%25%20CPU%20%C2%B7%200%20GPU-00ff9c?style=for-the-badge&labelColor=0d0d0d" />

</div>

> LoRA fine-tuning of Qwen3-0.6B/1.7B on CPU (i7-1185G7, 32GB RAM, 0 VRAM)
> Self-taught AI project — dataset distillation, hallucination tests, compared runs

## Context

Training LoRA adapters on CPU only, no GPU. The project explores the feasibility of local fine-tuning on consumer hardware, with a full dataset-distillation pipeline built from teacher models (Kimi K3, Grok, Mistral, GPT).

## Architecture

```
LoFT/
├── loft/                 # Main package
│   ├── train.py          # LoRA training loop (PEFT + accelerate)
│   ├── merge.py          # Adapter → base model merge
│   ├── export.py         # Quantized GGUF export
│   ├── chat.py           # Test interface
│   └── cli.py            # Unified CLI
├── tests/                # Test suite
│   ├── hallucination/    # Hallucination detection
│   ├── logique/          # Reasoning tests
│   ├── diagnostic/       # Hyperparameter sweeps
│   ├── data/             # Datasets (Kimi, Grok, Mistral, GPT, mixed)
│   └── results/          # Compared JSON results
├── runs/                 # Training runs (manifests + configs)
│   ├── cfgA/             # Configuration A
│   ├── v1_qwen06b/       # Qwen3-0.6B baseline
│   ├── grok_v1_qwen06b/  # Grok distillation
│   ├── gpt_v1_qwen06b/   # GPT distillation
│   └── ...
├── models/               # Base models (gitignored)
├── train_config.yaml     # Training configuration
└── requirements.txt      # Dependencies
```

## Pipeline

1. **Distillation**: dataset generation from teacher models (Kimi K3, Grok, GPT, Mistral)
2. **Filtering**: reject questions with no single answer, fix teacher mistakes
3. **Fine-tuning**: LoRA rank 4-16, CPU, Qwen3-0.6B (CPU-only, ~2h/run)
4. **Evaluation**: hallucination tests, logical reasoning, 40-item heldout
5. **Merge**: adapter → base model → quantized export

## Hardware

- Dell Latitude 5420 — i7-1185G7, 32GB RAM, Iris Xe (0 VRAM)
- 100% CPU — no GPU used
- PEFT/LoRA + accelerate (CPU offloading)

## Proven feasibility

- ✅ LoRA fine-tuning on CPU in ~2h/run (Qwen3-0.6B)
- ✅ Multi-teacher distillation with quality control
- ✅ Hallucination tests with configuration sweeps
- ✅ Adapter merge and export
- ⚠️ Qwen3-1.7B: feasible but ~8h/run (QLoRA probe)

## Author

Malik — IT Support student (DEP), self-taught
Project started: 2026-09-28

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00ff9c,50:03150c,100:e10600&height=100&section=footer" width="100%" alt="" />

**⚡ Built on CPU. Proven on disk. ⚡**

</div>
