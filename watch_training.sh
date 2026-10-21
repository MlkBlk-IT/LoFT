#!/usr/bin/env bash
# Surveille la fin du training Qwen3-1.7B et envoie une notification Telegram.
# ATTENDRE = fichier adapter_model.safetensors OU fin du process.
set -u
ADAPTER="/d/LoFT/runs/autodistill_17b/adapter/adapter_model.safetensors"
MANIFEST="/d/LoFT/runs/autodistill_17b/run_manifest.json"
PID=11424
DEADLINE=$(( $(date +%s) + 6 * 3600 ))   # abandon apres 6 h

while [ ! -f "$ADAPTER" ]; do
  if ! tasklist //FI "PID eq $PID" 2>/dev/null | grep -q "$PID"; then
    if [ -f "$MANIFEST" ]; then
      hermes send --to telegram "Training Qwen3-1.7B termine (manifeste present). Lance le compare : cd D:/LoFT ; python compare.py --dataset tests/data/dataset_v1.json --base models/qwen3-1.7b --adapter runs/autodistill_17b/adapter --tag kimi_s17b --n_seeds 1 --max_new_tokens 500" >/dev/null 2>&1
      echo "OK_TERMINE"
    else
      hermes send --to telegram "Training Qwen3-1.7B ARRETE sans manifeste. Verifie D:/LoFT/LoFT dans le terminal." >/dev/null 2>&1
      echo "ECHEC"
    fi
    exit 0
  fi
  [ "$(date +%s)" -gt "$DEADLINE" ] && { hermes send --to telegram "Watchdog training 1.7B : delai 6h depasse, process toujours actif." >/dev/null 2>&1; echo "TIMEOUT"; exit 0; }
  sleep 20
done

sleep 3   # laisse le temps au fichier d'etre complet
hermes send --to telegram "Training Qwen3-1.7B FINI. Adaptateur : $(du -m "$ADAPTER" | cut -f1) Mo. Lance le compare (option A, dataset Kimi) : cd D:/LoFT ; python compare.py --dataset tests/data/dataset_v1.json --base models/qwen3-1.7b --adapter runs/autodistill_17b/adapter --tag kimi_s17b --n_seeds 1 --max_new_tokens 500" >/dev/null 2>&1
echo "OK_TERMINE"