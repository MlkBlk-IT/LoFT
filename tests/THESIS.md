# Thèse mi-parcours v0.1 — POC_LoFT_CPU (2026-09-28)
Statut : hypothèse, pas un résultat. Modèle : google/gemma-3-1b-it, CPU, bf16, transformers 5.17.0.

## Données (42 runs, addition d'heures, FR, consigne "Explique en étapes courtes")
| Config | Défi | Runs | Correct |
|---|---|---|---|
| Défauts Google (temp non fixée) | P0 14h20+2h45 (17h05) | 1 | 0 |
| 0.2 / top_k 2 / rp 1.3 | P0 | 1 | 0 |
| v1 (0.5 / 64 / 0.95 / 1.05) | P0 | 10 | 0 |
| v1 | D1 14h20+2h30 (16h50) | 10 | 1 |
| v1 | D2 14h00+2h45 (16h45) | 10 | 10 |
| v1 | D3 14h20+2h00 (16h20) | 10 | 1 (4 heures invalides) |

## Balises
- Établi : sous v1, la fiabilité chute dès que les minutes de départ sont non nulles; erreurs dominantes = recopie infidèle des données, minutes non normalisées, étapes redondantes.
- Soutenu, pas prouvé : l'item domine sur les paramètres testés.
- Non établi : "le modèle est mauvais en raisonnement en général"; "les paramètres n'ont aucun effet"; "le libellé du prompt n'a aucun effet".
- Non testé : autre consigne, exemple dans le prompt, anglais, greedy, FP32, effet isolé de la pénalité, température sur un item intermédiaire.
- Menaces : facteurs confondus, n=1 pour deux configs, un seul modèle, un défi répété 10 fois n'est pas 10 items, check automatique heuristique (lecture manuelle faite).

## Thèse v0.1
Sur Gemma 3 1B it, en français avec la consigne "étapes courtes", la fiabilité sur l'addition d'heures dépend surtout des caractéristiques de l'item, non des paramètres de génération testés. Les échecs sont majoritairement des erreurs de recopie, de normalisation et d'étapes redondantes, pas du bruit d'échantillonnage.

## Falsification
Thèse falsifiée si, sur un même item intermédiaire, des configs de génération différentes donnent des taux de réussite nettement différents.

## Prochain test
Mêmes 4 questions en anglais, paramètres Google (temp 1.0, top_k 64, top_p 0.95, rp 1.0), 10 graines chacune. Langue et config changent ensemble (confusion connue).


---
# Thèse v0.2 (2026-09-28) — hypothèse, pas un résultat
Correction : le tableau v0.1 totalise 42 runs (pas 52). Total cumulé : 83 runs (42 FR + 41 EN).

## Données ajoutées (EN)
| Config | Défi | Correct |
|---|---|---|
| Google (T1.0 / k64 / p0.95 / rp1.0) | P0 14:20+2h45 | 0/10 |
| Google | D2 14:00+2h45 (16:45) | 6/10 |
| v1 (T0.5 / k64 / p0.95 / rp1.05) | D2 | 5/10 |
| D (T0.2 / k2 / p0.95 / rp1.3) | D2 | 8/10 |
| greedy | D2 | 1/1 |

## Constats
- L'item domine : EN P0 0/10 vs EN D2 6/10, même config, Fisher p~0.011 (calcul manuel). P0 cumulé (FR+EN) : 0/22.
- Effet config sur EN D2 : non détecté (5, 6, 8 sur 10; D vs v1 p~0.35, D vs Google p~0.63).
- Effet langue : FR v1 D2 10/10 vs EN v1 D2 5/10, p~0.033, mais confondu (notation d'heure "14h00" vs "14:00", consigne traduite). Contre EN D (8/10), p~0.47.
- Profil d'erreurs : 16:46 (décimal/horloge) et détour en secondes présents en Google et v1, absents en D. Hypothèse, pas résultat.
- Réponse finale juste != raisonnement juste : greedy (détour en secondes puis saut injustifié), Google/v1 seed 9, D ("14:00 (3 PM)" faux dans 8 runs sur 10).

## Balises v0.2
- Soutenu : l'item domine sur les paramètres testés.
- Non établi : effet de la config; effet isolé de la langue; "le modèle est mauvais en raisonnement en général".
- Non testé : greedy multi-items, FP32, autre consigne, exemple dans le prompt, top_k 2 seul, rp 1.3 seule.
- Menaces : n=10 par config; textes quasi identiques en config D (7 textes distincts sur 10, runs non indépendants); un seul modèle; check auto heuristique; p-values calculées à la main.

## Falsification v0.1
Non remplie : mêmes item et langue, configs différentes, 5/6/8 sur 10, non significatif.

## Conséquence pour l'éval avant/après du fine-tuning
Lire le raisonnement, pas seulement l'heure finale. Le check auto actuel ne le fait pas.
