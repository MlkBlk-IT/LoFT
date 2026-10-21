# Pré-prototype LoFT — journal de mesures

Toutes les valeurs sont **mesurées**, pas estimées.
Machine : i7-1185G7 4C/8T AVX2 · 31,4 Go RAM · Intel Iris Xe (CPU-only)
Python 3.12.10 (venv dédié D:\LoFT\venv) · torch 2.14.0+cpu · transformers 5.17.0 · peft 0.21.0

## Datasets

| | Teacher | Items | Justes | Faux balisés | Retirés | Étapes/CoT |
|---|---|---|---|---|---|---|
| `dataset_v1.json` | **Kimi K3** | 20 | 19 | 1 (q13, Ohm) | 0 | 5 |
| `dataset_grok_v1.json` | **Grok (xAI)** | 18 | 17 | 1 (g14, Newton) | 2 | 4 |

**Kimi** — 1 erreur involontaire trouvée et corrigée : q09 comptait 5 voyelles,
la réponse est 6 (l'élision `d'une` est un seul mot).

**Grok** — 2 items retirés car **sans réponse unique** :
- q19 : 4 placements satisfont les contraintes, milieu = A, B, C ou D
- q20 : polygones 3, 5, 6 — aucune progression, « heptagone » arbitraire
- q10 : 2 intrus possibles (Flûte et Trompette hors cordes), conservé avec réserve

**Faux balisés** — mécanisme identique dans les deux datasets (confusion
concept/formule), ce qui permet une comparaison directe :
- Kimi q13 : `I = U/R` lu comme `U − R` → faux `12`, vrai `2`
- Grok g14 : `a = F/m` lu comme `F − m` → faux `2`, vrai `5`

## Entraînement (LoRA CPU)

Configuration identique pour les deux runs.

| Paramètre | Valeur |
|---|---|
| Student | Qwen3-0.6B (596 M paramètres réels) |
| Cibles LoRA | `q_proj, k_proj, v_proj, o_proj` |
| Params entraînables | 2 293 760 / 598 343 680 (**0,3834 %**) |
| r / alpha / dropout | 8 / 16 / 0.1 |
| Batch / accumulation | 1 / 4 |
| Epochs | 3 |
| max_length | 512 |
| Learning rate | 2e-4 |
| Seed | 42 |
| enable_thinking | 1 |
| Masque de perte | question ignorée (`-100`), réponse + CoT apprise |

### Run Kimi — `runs/v1_qwen06b`

```
durée        : 673 s (11 min 13, 44,9 s/step, 15 steps)
pic RSS      : 3 496 MB (3,41 Go)
adaptateur   : 8,78 Mo
séquence     : 225 / 259 / 380 tokens (min/méd/max)
tokens loss  : 178 / 204 / 274
loss         : 1,208 → 0,928 (−23 %), oscillante (variance sur 20 items)
```

### Run Grok — `runs/grok_v1_qwen06b`

```
durée        : (en cours)
```

## Modifications apportées à `loft/train.py`

Backup : `loft/train_original_backup.py` (sha256 identique à l'original).

| # | Défaut LoFT d'origine | Correction |
|---|---|---|
| 1 | `max_length=128` codé en dur | `--max_length` CLI — une CoT de raisonnement dépasse 128 tokens |
| 2 | `labels = input_ids.copy()` | Masque `-100` sur la question ; **70 ignorés / 230 appris** vérifié |
| 3 | `\n### Response:\n` plat | `apply_chat_template` du modèle — format Qwen3 exact |
| 4 | Pic RAM lu avant/après | Thread `PeakMemoryMonitor`, échantillonnage 250 ms |
| 5 | `r=8` codé en dur | `--lora_r / --lora_alpha / --lora_dropout` |
| 6 | Impossible de borner un test | `--max_steps`, `--seed`, `--limit` |
| 7 | Format liste nue seulement | Accepte `{metadata, items}` et `[...]` |
| 8 | LayerNorm verrouillés | Débloqués (conservé de l'original, `--no_layernorm` pour l'inverse) |

## Chat template Qwen3 — source et comportement

Source : `tokenizer_config.json` de `Qwen/Qwen3-0.6B` (HF) + doc Transformers
`chat_templating`. eos `151645`, pad `151643`, `add_bos_token=False`,
`tokenizer_class=Qwen2Tokenizer`, `model_max_length=131072`.

```
entraînement  (add_generation_prompt=False, enable_thinking=True)
  user\nQUESTION\nassistant\n<think>\nCoT\n</think>\n\nRÉPONSE\n

inférence      (add_generation_prompt=True, enable_thinking=True)
  user\nQUESTION\nassistant\n

sans thinking  (enable_thinking=False)
  user\nQUESTION\nassistant\n<think>\n\n</think>\n\n
```

**`enable_thinking=False` est obligatoire pour comparer avec Gemma** — sinon
on mesure la longueur du raisonnement, pas la qualité de la réponse.

## Comparaison avant / après — Kimi

`compare.py`, 1 seed, `max_new_tokens=500`, ~35 min.

```
BASE : 14/20 (70 %)
FT   : 16/20 (80 %)
DELTA: +2
```

| Item | Catégorie | Base | FT |
|---|---|---|---|
| q16 | code | 0/1 | **1/1** |
| q17 | planification | 0/1 | **1/1** |

Échecs persistants dans les deux versions : q07 (multi_hop), q11 et q12
(spatio-temporel), q18 (planification). **Spatio-temporel : 0/4 avant comme après.**

**Faux balisé q13 : `12` sorti 1/1 fois → MÉMORISÉ.**
Taux de contamination : 1/20 = **5 %**.

### Différences de méthode observées (lecture des CoT)

| Item | Base | Après fine-tuning |
|---|---|---|
| q01 | « some juice packs » (récite sans chiffres) | « 4 packs at 6.25, 3 plates at 8.40 » (énonce les données) |
| q02 | langage naturel (« Calculating 3/5 of 35 ») | symbolique puis décomposition (`3/5 * 350`, `0.6`) |
| q03 | 2 exigences sur 3 énoncées | **3 exigences sur 3** énoncées |

Motif transmis : **énoncer les données et les critères avant de raisonner** —
structure commune aux CoT de Kimi (« Données : ... » en étape 1).

## Limites connues

- **1 seed.** Une génération unique peut être un accident. Les 3 seeds sont prévus.
- **20 puis 18 items.** Très petit. Un delta de +2 sur 18-20 items n'est pas
  significatif — il indique une direction, pas un effet.
- **Scorer corrigé en cours de route.** Le run initialamsung notait FAUX les
  réponses en point décimal (`49.80` vs `49,80`). Corrigé puis rescope via
  `rescore.py`. Le seul itemaffected est q01.
- **Entraînement et test sur le même dataset.** Pas de held-out split. Le
  `+2` peut inclure de la mémorisation.
- **Qwen3-0.6B n'est pas le Student du brief** (Qwen3-8B). C'est un banc
  d'essai de pipeline.