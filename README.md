# LoFT — Local Fine-Tuning on CPU

> Fine-tuning LoRA de Qwen3-0.6B/1.7B sur CPU (i7-1185G7, 32GB RAM, 0 VRAM)
> Projet d'IA autodidacte — distillation de datasets, tests d'hallucination, runs comparés

## Contexte

Entraînement d'adaptateurs LoRA sur processeur CPU uniquement, sans GPU. Le projet explore la faisabilité du fine-tuning local sur du matériel grand public, avec un pipeline complet de distillation de datasets à partir de modèles enseignants (Kimi K3, Grok, Mistral, GPT).

## Architecture

```
LoFT/
├── loft/                 # Package principal
│   ├── train.py          # Boucle d'entraînement LoRA (PEFT + accelerate)
│   ├── merge.py          # Fusion adapter → modèle de base
│   ├── export.py         # Export quantisé GGUF
│   ├── chat.py           # Interface de test
│   └── cli.py            # CLI unifié
├── tests/                # Suite de tests
│   ├── hallucination/    # Détection d'hallucinations
│   ├── logique/          # Tests de raisonnement
│   ├── diagnostic/       # Sweeps de hyperparamètres
│   ├── data/             # Datasets (Kimi, Grok, Mistral, GPT, mixte)
│   └── results/          # Résultats JSON comparés
├── runs/                 # Runs d'entraînement (manifests + configs)
│   ├── cfgA/             # Configuration A
│   ├── v1_qwen06b/       # Baseline Qwen3-0.6B
│   ├── grok_v1_qwen06b/  # Distillation Grok
│   ├── gpt_v1_qwen06b/   # Distillation GPT
│   └── ...
├── models/               # Modèles de base (gitignored)
├── train_config.yaml     # Configuration d'entraînement
└── requirements.txt      # Dépendances
```

## Pipeline

1. **Distillation** : Génération de datasets à partir de modèles enseignants (Kimi K3, Grok, GPT, Mistral)
2. **Filtrage** : Rejet des questions sans réponse unique, correction des fautes des enseignants
3. **Fine-tuning** : LoRA rank 4-16, CPU, Qwen3-0.6B (CPU-only, ~2h/run)
4. **Évaluation** : Tests d'hallucination, raisonnement logique, heldout 40 items
5. **Fusion** : Merge adapter → modèle de base → export quantisé

## Matériel

- Dell Latitude 5420 — i7-1185G7, 32GB RAM, Iris Xe (0 VRAM)
- 100% CPU — aucun GPU utilisé
- PEFT/LoRA + accelerate (CPU offloading)

## Faisabilité prouvée

- ✅ Fine-tuning LoRA sur CPU en ~2h/run (Qwen3-0.6B)
- ✅ Distillation multi-enseignants avec contrôle qualité
- ✅ Tests d'hallucination avec sweep de configurations
- ✅ Fusion et export d'adaptateurs
- ⚠️ Qwen3-1.7B : faisable mais ~8h/run (QLoRA probe)

## Auteur

Malik — étudiant DEP Soutien Informatique, autodidacte
Projet démarré : 2026-09-28
