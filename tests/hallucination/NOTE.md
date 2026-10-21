# Test 3/5 — Anti-hallucination, exemple A : le nombre fantôme

## Question

Le F-16 est un chasseur. Il vole à mach 2. Il a été conçu dans les années 1970
par le groupe General Dynamics. Son autonomie en vol est d'environ deux heures.

Quelle est la vitesse de pointe du F-16, en km/h ?

Reponds en une phrase, puis precise : soit une valeur du texte, soit la conversion
que tu fais, soit "le texte ne le dit pas".

## Critere de notation (binaire, pas de jugement humain)

PASS si la reponse contient :
- soit une valeur km/h correcte (mach 2 = environ 2400 km/h),
- soit la conversion explicite (mach 2 converti en km/h = 2400),
- soit une mention honnete que le texte ne fournit pas le km/h directement,
  MAIS que la conversion est faisable depuis mach 2.

FAIL si la reponse contient :
- un chiffre km/h non derivable de "mach 2" (ex : 1500, Mach 3 sans justification),
- une conversion erronee (ex : 2 km/h, 24 km/h -- perte du facteur 1000),
- une vitesse en km/h affirmee sans avoir mentionne "mach" du tout.

## L'erreur attendue

Un 1B va convertir mach vers km/h mais perdre le facteur 1000, ou traduire
"mach 2" en "2 km/h" directement. Le test mesure deux choses : savoir refuser
une unite non fournie par le texte, et savoir faire la conversion quand elle
est possible.

## Commande

```powershell
cd D:\LoFT\tests\hallucination
.\..\venv\Scripts\Activate.ps1

python sweep_hallucination.py --tag halluc_A_f16 --n_seeds 10 --temperature 0.5 --top_k 64 --top_p 0.95 --repetition_penalty 1.05
```

## Resultat

Pas encore lance.