# Annotations — tests baseline Gemma 3 1B it (2026-09-28)
Modèle : google/gemma-3-1b-it, CPU, bf16, transformers 5.17.0, torch 2.14.0+cpu, 4 threads. RAM pic 2.62 GB. Vitesse 4.5 à 10.3 tok/s (dérive, cause non vérifiée).

## Tous les runs (83)
| Tag | Langue | Config | Défi | Correct |
|---|---|---|---|---|
| test1_defaults | FR | Google, temp non fixée | P0 (17h05) | 0/1 |
| test2_config | FR | 0.2 / k2 / rp1.3 | P0 | 0/1 |
| sweep_v1 | FR | v1 | P0 | 0/10 |
| sweep_d1 | FR | v1 | 14h20+2h30 (16h50) | 1/10 |
| sweep_d2 | FR | v1 | 14h00+2h45 (16h45) | 10/10 |
| sweep_d3 | FR | v1 | 14h20+2h00 (16h20) | 1/10 |
| en_google_p0 | EN | Google | 14:20+2h45 (17:05) | 0/10 |
| en_google_d2 | EN | Google | 14:00+2h45 (16:45) | 6/10 |
| en_v1_d2 | EN | v1 | D2 | 5/10 |
| en_cfgD_d2 | EN | 0.2 / k2 / p0.95 / rp1.3 | D2 | 8/10 |
| en_greedy_d2 | EN | greedy | D2 | 1/1 |

## Modes d'échec (lecture manuelle)
- FR : recopie infidèle de la durée (2h30->3h30, 2h00->2h40), minutes non normalisées (16h60, 17h65), durée ajoutée deux fois, minutes de départ perdues (16h45 pour 14h20+2h45).
- EN P0 : 16:65 (x4), 16:45 (minutes de départ perdues), 16:20.75 (mélange décimal/horloge).
- EN D2 Google : 16:46 x2, 21:30, 14:45. v1 : 16:46 x4, 14:45. D : 5:45 PM ("3 PM"+2h), 14:05 (2.75/60).
- Détour en secondes (45 x 60 = 2700 s) : Google 3 runs, v1 2 runs.
- Config D : "14:00 (3 PM)" faux dans 8 runs sur 10; 7 textes distincts sur 10.
- Greedy : détour en secondes puis saut à 16:45, juste sans raisonnement valide.

## Limites
- n=10 par config; un run par graine; un seul modèle.
- Config D : runs quasi identiques, IC optimiste.
- Langue confondue avec notation d'heure et traduction de la consigne.
- Check auto : dernière heure du texte, modulo 12 depuis les runs EN; résultats FR relus à la main.
- p-values Fisher calculées à la main, pas par script.
- Greedy : gen_kwargs à confirmer dans D:\LoFT\tests\results\en_greedy_d2.json.
- Défauts Google FR (test1) : temperature non fixée, effective non vérifiée.

## Fichiers résultats
D:\LoFT\tests\results\*.json (un par tag).
