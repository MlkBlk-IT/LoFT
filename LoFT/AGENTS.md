# LoFT — package interne

Repo git imbriqué dans `D:\LoFT`. Contient le code réel de la chaîne.

## Modules

| Fichier | Rôle |
|---|---|
| `train.py` | entraînement LoRA CPU. **Sauvegarde obligatoire de l'original avant réécriture.** |
| `train_original_backup.py` | copie de sauvegarde du `train.py` d'origine |
| `merge.py` | fusion d'adaptateurs LoRA |
| `chat.py` | inference avec chat template |
| `quantize.py` | quantification GGUF |
| `export.py` | export de l'adaptateur |
| `test_adapter.py` | vérification de l'adaptateur |
| `utils.py` | helpers |
| `train_config.yaml` | hyperparamètres |

## Règles

- **Chat template Qwen** : doit correspondre à la source officielle. Toute
  modification du template est un changement de comportement du modèle, pas un
  détail cosmétique — le dire explicitement avant de le changer.
- `train.py` a déjà été réécrit plusieurs fois. Avant toute édition : lire le
  fichier courant, ne pas supposer qu'on connaît sa structure.
- Les adaptateurs vivent dans `D:\LoFT\runs\<run>\adapter\`. Un run =
  un dossier, un `run_manifest.json`.
- Les modèles mergés vont dans `merged_models\`. Ne jamais écraser un adapter
  existant — créer un nouveau run.
- Code et commentaires en anglais.
