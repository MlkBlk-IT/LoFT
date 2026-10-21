# Tests v2 — overflow de minutes (2026-10-03)

## Hypothese

Gemma-3-1b-it **ne sait pas** normaliser un overflow de minutes (`20 + 45 = 65`).
Le prompting ne peut pas le reparer (constate dans `annotations.md` : `16:60`, `16:65`,
minutes de depart perdues). Un LoRA **si** — c'est la seule voie.

## Fichiers

| Fichier | Role |
|---|---|
| `prompt_overflow_fr.txt` | Test 1 — overflow minutes, FR, prompt "etapes courtes" |
| `prompt_control_fr.txt` | Test 2 — controle negatif, meme prompt, overflow = 0 |

## Expected (calcule, pas ecrit a la main)

| Test | Prompt | Expected | Overflow | Baseline attendue |
|---|---|---|---|---|
| 1 | 14h20 + 2h45 | **17:05** | 1h05 | **0/10** |
| 2 | 14h00 + 2h45 | **16:45** | 45 min | **10/10** |

## Commandes PowerShell

```powershell
cd D:\LoFT
.\venv\Scripts\Activate.ps1
cd D:\LoFT\tests

python constraint_sweep.py --tag overflow_fr_d2 `
  --prompt_file prompt_overflow_fr.txt `
  --n_seeds 10 --temperature 0.5 --top_k 64 --top_p 0.95 `
  --repetition_penalty 1.05
```

```powershell
python constraint_sweep.py --tag control_fr_d2 `
  --prompt_file prompt_control_fr.txt `
  --n_seeds 10 --temperature 0.5 --top_k 64 --top_p 0.95 `
  --repetition_penalty 1.05
```

## Lecture des resultats

`constraint_sweep.py` ne verifie PAS l'heure. Il verifie `digits_ok` et `sentences_ok`
(prompt_control.txt). Il faut une verification manuelle, ou un nouveau script.

**A noter avant de lancer :** pour ces 2 prompts, la reponse doit contenir
`17:05` ou `16:45` — donc `digits_ok` sera `False` par construction.
Les compteurs `digits_ok`/`both_ok` de ces runs n'ont pas le meme sens
que ceux de `prompt_control.txt`. Ne pas les comparer a ceux de l'ancien test.

## Hypothese de sortie

Test 1 : si `n_digits` contient `17:05` -> le modele a surmonte l'overflow (ou a devine).
Test 1 : si on voit `16:65`, `17:65`, `16:60` -> il n'a pas surmonte.

Test 2 : doit reproduire `10/10` (117 tok moyens, 6.79 tok/s, 2.62 GB).

## Resultats (2026-10-03 14:57, config D2 identique aux deux)

| Tag | Prompt | Attendu | **SCORE** |
|---|---|---|---|
| `control_fr_d2` | 14h00 + 2h45 | 16:45 | **10/10** |
| `overflow_fr_d2` | 14h20 + 2h45 | 17:05 | **0/10** |

Seule difference entre les deux: `14h20` vs `14h00`. Meme phrase,
meme config, memes seeds. Le bug est **localise aux minutes de depart
non nulles**.

Metriques communes (identiques): peak_wset 2.62 GB, stopped_on_eos 10/10.
`mean_new_tokens` 117.2 (controle) vs 123.2 (overflow).
`mean_tokens_per_s` 5.55 vs 5.56 -- pas de cout de performance.

## Modes d'echec du test 1

| Mode | Seeds | Mecanique |
|---|---|---|
| Minutes effacees | 7 | `14h20 + 2h45 = 16h45` -- traite 14h20 comme 14h00 |
| Minutes volees | 1 | `16:05` -- a soustrait 20 min |
| Duree doublee | 2 | `16:45 + 2h45 = 19:45` |

**Heures non normalisees (>=60 min): 0 dans les deux tests.**
Le modele ne deborde JAMAIS. Il n'extrait pas l'argument.

Preuve seed 2: ecrit `2h45 (une heure et 45 minutes)` -- calcule
correctement 1h45, puis repond 16:05. La conversion est acquise,
elle n'est pas appliquee au depart.

## Interpretation

La regle manquante est: **normaliser avant d'additionner**
(20 + 45 = 65 -> +1h). Regle arithmetique, pas un probleme de langage
ni de format. Cible de LoRA, pas de prompt.

Baseline = 0/10 -> 10 points de headroom disponibles. C'est le point
de depart du projet.

## Reproductibilite

```powershell
python score_time.py overflow_fr_d2 "17:05"
python score_time.py control_fr_d2 "16:45"
```

## Avertissement de lecture

`constraint_sweep.py` ne verifie pas l'heure. `digits_ok` / `both_ok`
sont FAUX par construction sur ces deux tests (la reponse DOIT contenir
une heure). Ne pas les lire comme un score. Seul `score_time.py` compte.