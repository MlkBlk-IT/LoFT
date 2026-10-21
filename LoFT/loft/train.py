"""Entraînement LoRA pour Qwen3 -- réécriture adaptée au projet.

Remplace loft/train.py (sauvegardé dans loft/train_original_backup.py).

Corrections par rapport au code d'origine:

1. max_length=128 codé en dur -> configurable via --max_length.
   Une CoT de raisonnement Qwen3 dépasse largement 128 tokens.

2. labels = input_ids.copy() -> masque de perte sur la réponse seulement.
   L'original apprenait aussi sur la question, ce qui contamine toute
   mesure de progression.

3. Pas de chat_template -> apply_chat_template du modèle.
   Qwen3 exige 'user\\n...\\nassistant\\n<think>\\n...\\n</think>\\n\\n...\\n'.
   Entraîner sur '\\n### Response:\\n' apprendrait un format que le modèle
   ne produira jamais à l'inférence.

4. Pic RAM mesuré au début/fin -> suivi réel via un thread d'échantillonnage.

5. r/alpha/dropout codés en dur -> --lora_r/--lora_alpha/--lora_dropout.

6. Impossible de borner un test -> --max_steps et --seed.

Format dataset attendu (JSON, liste d'objets):
    [{"question": "...", "cot": "<think>...</think>...réponse", "answer": "..."}]

Le champ 'cot' peut contenir ou non les balises <think> -- elles sont
ajoutées automatiquement si absentes.
"""

import argparse
import json
import os
import platform
import threading
import time

import psutil
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ORIGINAL = "loft/train_original_backup.py"


class PeakMemoryMonitor:
    """Échantillonne le RSS dans un thread pour capturer le pic réel.

    psutil.Process().memory_info() lu avant/après manque le pic: le
    cœur de l'entraînement (forward + backward) se termine avant la
    lecture finale.
    """

    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_rss_mb = 0.0
        self.peak_vms_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                mi = proc.memory_info()
                self.peak_rss_mb = max(self.peak_rss_mb, mi.rss / 1024**2)
                self.peak_vms_mb = max(self.peak_vms_mb, mi.vms / 1024**2)
            except psutil.Error:
                break
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return False


def build_example(row, tokenizer, max_length, enable_thinking):
    """Construit un exemple avec masque de perte sur la réponse seule.

    Retourne None si la réponse ne tient pas dans max_length: on préfère
    un exemple propre et tronqué par le dataset qu'un exemple coupé au
    milieu du raisonnement.
    """
    question = row.get("question") or row.get("instruction") or ""
    if row.get("input"):
        question = question + "\n" + row["input"]
    answer = row.get("answer") or ""
    cot = row.get("cot") or row.get("output") or ""

    # La CoT peut être fournie sans les balises. On les normalise pour que
    # l'adaptateur apprenne exactement ce que l'inférence produira.
    if cot and "<think>" not in cot:
        body = cot if "</think>" in cot else cot + "</think>"
        cot = "<think>\n" + body.strip()
    elif cot:
        cot = "<think>\n" + cot.split("<think>", 1)[1].strip()
    content = (cot + "\n\n" + answer).strip() if cot else answer

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_length:
        return None
    if len(prompt_ids) >= len(full_ids):
        return None

    # -100 = token ignoré par la loss (convention HuggingFace).
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "n_prompt": len(prompt_ids),
        "n_target": len(full_ids) - len(prompt_ids),
    }


def load_and_tokenize(path, tokenizer, max_length, enable_thinking, limit):
    raw = json.load(open(path, encoding="utf-8"))

    # Deux formats acceptés:
    #   [ {...}, {...} ]                      liste nue (format LoFT d'origine)
    #   {"metadata": {...}, "items": [...]}   objet enveloppé (notre format)
    if isinstance(raw, dict):
        raw = raw.get("items") or raw.get("data") or raw.get("examples")
        if raw is None:
            raise SystemExit(
                "Objet JSON sans 'items'/'data'/'examples'. Clés vues: %s"
                % list(json.load(open(path, encoding='utf-8')).keys()))

    if limit:
        raw = raw[:limit]

    feats, dropped = [], []
    for i, row in enumerate(raw):
        ex = build_example(row, tokenizer, max_length, enable_thinking)
        if ex is None:
            dropped.append(i)
            continue
        feats.append(ex)

    if not feats:
        raise SystemExit(
            "Aucun exemple ne tient dans max_length=%d. Augmente --max_length."
            % max_length
        )
    return Dataset.from_list(feats), dropped


def lora_targets(model):
    """Noms des modules ciblés par LoRA.

    On cible le groupe attention complet (q,k,v,o) : c'est le minimum
    stable. q_proj seul converge vite mais expresse peu; ajouter le MLP
    (gate/up/down) augmente la capacité mais coûte ~2x le temps CPU.
    """
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    ordre = [("q_proj", "k_proj", "v_proj", "o_proj"),
             ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")]
    for groupe in ordre:
        if all(c in names for c in groupe):
            return list(groupe)
    raise SystemExit("Aucune cible LoRA reconnue. Modules vus: %s" % sorted(names)[:20])


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tuning LoRA CPU, adapté au projet Qwen3/CoT.")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="Borne le test. -1 = epochs seulement.")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="Ne garder que les N premiers exemples (test rapide).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--enable_thinking", type=int, choices=[0, 1], default=1,
                    help="1 = Qwen3<think> actif (défaut, adapté au projet).")
    ap.add_argument("--no_layernorm", action="store_true",
                    help="Ne pas déverrouiller les LayerNorm (défaut: débloqués).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print("=" * 66)
    print("ENVIRONNEMENT")
    print("=" * 66)
    print("  python      :", platform.python_version())
    print("  torch       :", torch.__version__)
    print("  transformers:", transformers.__version__)
    print("  peft        :", __import__("peft").__version__)
    print("  threads     :", torch.get_num_threads())
    print("  modele      :", args.model_name)
    print("  seed        :", args.seed)
    print("  thinking    :", bool(args.enable_thinking))
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id

    if not args.no_layernorm:
        # Conservé du code LoFT d'origine: certains modèles gelent leurs
        # LayerNorm et l'adaptateur n'arrive pas à s'adapter.
        for name, param in model.named_parameters():
            if "layernorm" in name.lower():
                param.requires_grad = True

    targets = lora_targets(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("LoRA: cibles=%s r=%d alpha=%d -> %s params entraînables / %s (%.4f%%)"
          % (targets, args.lora_r, args.lora_alpha,
             f"{trainable:,}", f"{total:,}", 100 * trainable / total))
    print()

    ds, dropped = load_and_tokenize(
        args.dataset_path, tokenizer, args.max_length,
        bool(args.enable_thinking), args.limit)
    if dropped:
        print("Exemples écartés (dépassent max_length=%d): %s"
              % (args.max_length, dropped))
    lens = [len(f["input_ids"]) for f in ds]
    tgts = [f["n_target"] for f in ds]
    print("Dataset: %d exemples | longueur min/med/max = %d/%d/%d"
          % (len(ds), min(lens), sorted(lens)[len(lens) // 2], max(lens)))
    print("         tokens de réponse (avec loss): min/med/max = %d/%d/%d"
          % (min(tgts), sorted(tgts)[len(tgts) // 2], max(tgts)))
    print()

    steps_per_epoch = max(1, len(ds) // (args.per_device_train_batch_size
                                         * args.gradient_accumulation_steps))
    print("Entraînement: epochs=%s max_steps=%s -> ~%d pas par epoch"
          % (args.num_train_epochs, args.max_steps, steps_per_epoch))
    print()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds)

    print("=" * 66)
    print("ENTRAINEMENT")
    print("=" * 66)
    t0 = time.time()
    with PeakMemoryMonitor() as mem:
        trainer.train()
    train_s = time.time() - t0

    adapter_dir = os.path.join(args.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print()

    # --- vérification: l'adaptateur doit-il vraiment/learn only the answer ? ---
    print("=" * 66)
    print("VERIFICATION DU MASQUE DE PERTE")
    print("=" * 66)
    feat = ds[0]
    n_ignores = sum(1 for x in feat["labels"] if x == -100)
    n_total = len(feat["labels"])
    print("  tokens total          : %d" % n_total)
    print("  tokens ignorés (-100) : %d  <- la question" % n_ignores)
    print("  tokens avec loss      : %d  <- la réponse" % (n_total - n_ignores))
    print("  ratio                 : %.1f%% de la séquence apprise"
          % (100 * (n_total - n_ignores) / n_total))
    print()

    adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    adapter_mb = (os.path.getsize(adapter_file) / 1024**2
                  if os.path.exists(adapter_file) else 0.0)

    manifest = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "args": vars(args),
        "lora_target_modules": targets,
        "n_examples": len(ds),
        "dropped_indices": dropped,
        "seq_len_min": min(lens), "seq_len_med": sorted(lens)[len(lens) // 2],
        "seq_len_max": max(lens),
        "train_seconds": round(train_s, 1),
        "peak_rss_mb": round(mem.peak_rss_mb, 1),
        "peak_vms_mb": round(mem.peak_vms_mb, 1),
        "adapter_size_mb": round(adapter_mb, 2),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": __import__("peft").__version__,
        },
        "platform": platform.platform(),
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 66)
    print("RESULTATS")
    print("=" * 66)
    print("  durée entraînement : %.1f s" % train_s)
    print("  pic RSS            : %.0f MB (%.2f GB)"
          % (mem.peak_rss_mb, mem.peak_rss_mb / 1024))
    print("  taille adaptateur  : %.2f MB" % adapter_mb)
    print("  adaptateur         : %s" % adapter_dir)
    print("  manifeste          : %s" % os.path.join(args.output_dir, "run_manifest.json"))
    print("  (l'original est sauvegarde dans %s)" % ORIGINAL)


if __name__ == "__main__":
    main()