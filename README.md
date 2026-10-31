# LoFT

CPU-only LoRA fine-tuning experiments for Qwen3 models, including dataset distillation, comparison runs and hallucination tests.

## Context

LoFT explores what is practical on consumer hardware without a dedicated GPU. The reference machine is an i7-1185G7 laptop with 32 GB of RAM and no VRAM.

Training is therefore slow and resource-sensitive. Results should be read as documented experiments, not as a promise that every machine will behave the same way.

## Pipeline

1. Prepare and inspect the dataset.
2. Distil examples from teacher models.
3. Train a LoRA adapter with PEFT and Accelerate.
4. Compare the result with the base model.
5. Run reasoning and hallucination checks.
6. Merge or export the adapter only after reviewing the run manifest.

## Repository map

- `loft/` — training, merge, export, chat and CLI code
- `tests/` — reasoning, hallucination and diagnostic tests
- `runs/` — configurations, manifests and comparison results
- `docs/` — experiment notes

## Hardware note

CPU-only fine-tuning needs patience, free disk space and enough RAM. Check the selected model size and sequence length before starting a long run.

## Status

Research and experimentation. Results are tied to the recorded configuration and dataset.

## Author

Built by [Malik Belkacemi](https://github.com/MlkBlk-IT).
