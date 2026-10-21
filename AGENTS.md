# LoFT — pré-prototype personnel (non publié)

## But du dépôt

Ce dépôt est un **pré-prototype PERSONNEL, non publié**. Il ne teste pas les
modèles : il teste la **chaîne de distillation**.

Chaîne validée en un seul point d'entrée :
`Teacher → JSON → vérification → LoRA CPU → compare`

Le sujet final visé est **Qwen3-8B**. Le CPU-only Qwen3-0.6B n'est qu'un
substitut bon marché pour valider la mécanique.

Les ToS des vendors sont assumés : ce code n'est pas destiné à la publication.

## Arborescence (lue au démarrage de session)

| Chemin | Rôle |
|---|---|
| `LoFT/` | Package Python (repo git imbriqué) — train, merge, quantize, chat |
| `models/` | poids de base : `gemma-3-1b-it`, `qwen3-0.6b`, `qwen3-1.7b` |
| `runs/` | sorties d'entraînement : `v1_qwen06b`, `grok_v1_qwen06b`, `mistral_v1_qwen06b`, `gpt_v1_qwen06b`, `cfgA`, `dryrun` |
| `tests/data/` | datasets JSON (kimi, grok, mistral, gpt, mixte) |
| `tests/results/` | JSON de résultats comparés |
| `tests/` | scripts de sweep + dossiers `diagnostic/`, `hallucination/`, `logique/` |
| `venv/` | environnement Python 3.12 — **ne jamais réinstaller dedans** |

## Environnement

- **Toujours** utiliser `D:\LoFT\venv\Scripts\Activate.ps1` (Python 3.12).
- **Ne jamais** installer de paquet dans le venv sans demande explicite.
- CPU-only. `max_new_tokens` >= 200 sinon les CoT sont tronqués.
- Qwen3 : `enable_thinking=False` pour comparer les runs de façon homogène.

## Qui lance quoi

**L'agent ne lance JAMAIS un run de sa propre initiative.** Le propriétaire
lance lui-même en PowerShell. L'agent fournit :

1. la commande PowerShell complète, affichée **avant** exécution ;
2. la projection de durée, calculée depuis une mesure réelle (pas une
   estimation) ;
3. l'explication de ce que le run doit prouver.

Quand le propriétaire colle la sortie du terminal, l'agent :

- affiche **le texte intégral** de chaque question + le verdict ;
- n'affiche **jamais** seulement `q01 ok` ;
- écrit le JSON de résultat **à chaque item**, pas à la fin.

## Livrables

- Langue : code et commentaires en **anglais**, explications et livrables en **français**.
- Aucun livrable professionnel ne doit contenir d'information personnelle.
