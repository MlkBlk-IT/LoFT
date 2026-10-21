"""pipeline_nuit.py -- Pipeline complet Qwen3-1.7B, de bout en bout.

Enchaîne sans intervention :
    1. TRAINING sur dataset_autodistill_b  (traces natives du 0.6B de base)
    2. COMPARE sur les 4 datasets Teachers, 1 seed
    3. COMPARE sur un held-out de l'autodistill (mesure de généralisation)

Ecrit un journal JSONL incremental : si le run s'arrete, tout ce qui est
fait est conserve.

Usage (PowerShell):
  cd D:\\LoFT
  python pipeline_nuit.py --stage autodistill
  python pipeline_nuit.py --stage eval_teachers
  python pipeline_nuit.py --stage heldout
  python pipeline_nuit.py --stage all
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = r"D:\LoFT"
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
MODEL = "D:/LoFT/models/qwen3-1.7b"
RUNS = os.path.join(ROOT, "runs")
TESTS = os.path.join(ROOT, "tests")
JOURNAL = os.path.join(RUNS, "pipeline_nuit.jsonl")

# 4 datasets Teachers pour l'evaluation (identiques a ceux du 0.6B)
EVAL = [
    ("kimi",    os.path.join(TESTS, "data", "dataset_v1.json"),         "kimi_s17b"),
    ("grok",    os.path.join(TESTS, "data", "dataset_grok_v1.json"),    "grok_s17b"),
    ("mistral", os.path.join(TESTS, "data", "dataset_mistral_v1.json"), "mistral_s17b"),
    ("gpt",     os.path.join(TESTS, "data", "dataset_gpt_v1.json"),      "gpt_s17b"),
]
AUTODISTILL = os.path.join(TESTS, "data", "dataset_autodistill_b.json")

# Hyperparametres. 1024 = minimum qui garde les 54 items (512 n'en garde que 17).
HP = ["--num_train_epochs", "3", "--max_length", "1024", "--seed", "42",
      "--enable_thinking", "1"]


def log(stage, status, **kw):
    rec = {"ts": datetime.now().isoformat(timespec="seconds"),
           "stage": stage, "status": status, **kw}
    os.makedirs(RUNS, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[journal] %s" % json.dumps(rec, ensure_ascii=False), flush=True)


def run(cmd, label, timeout=None):
    """Lance un sous-processus, streame sa sortie dans un log."""
    os.makedirs(RUNS, exist_ok=True)
    logf = os.path.join(RUNS, "%s.log" % label)
    print("\n" + "=" * 78, flush=True)
    print(">>> %s" % label, flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as out:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=out,
                             stderr=subprocess.STDOUT, text=True,
                             bufsize=1)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -9
    dt = time.time() - t0
    print("<<< %s : rc=%s en %.0f s (%.1f min) -> %s"
          % (label, rc, dt, dt / 60, logf), flush=True)
    return rc, dt, logf


def stage_train(out_dir, dataset, label):
    cmd = [PY, os.path.join("LoFT", "LoFT", "train.py"),
           "--model_name", MODEL,
           "--dataset_path", dataset,
           "--output_dir", out_dir] + HP
    rc, dt, logf = run(cmd, label)
    adapter = os.path.join(out_dir, "adapter", "adapter_model.safetensors")
    ok = rc == 0 and os.path.exists(adapter)
    log("train:" + label, "OK" if ok else "ECHEC",
        rc=rc, seconds=round(dt, 1), adapter=adapter, log=logf)
    return ok


def stage_compare(adapter, dataset, tag):
    cmd = [PY, os.path.join(ROOT, "compare.py"),
           "--dataset", dataset,
           "--base", MODEL,
           "--adapter", adapter,
           "--tag", tag,
           "--n_seeds", "1",
           "--max_new_tokens", "500"]
    rc, dt, logf = run(cmd, "compare_" + tag)
    out = os.path.join(TESTS, "results", "compare_%s.json" % tag)
    log("compare:" + tag, "OK" if rc == 0 else "ECHEC",
        rc=rc, seconds=round(dt, 1), json=out, log=logf)
    return rc == 0


def resume_compare(tag):
    """Relit un JSON de compare s'il existe deja (reprise apres interruption)."""
    p = os.path.join(TESTS, "results", "compare_%s.json" % tag)
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    items = d.get("items", [])
    if not items:
        return None
    completes = all(i.get("ft_score") for i in items)
    return {"tag": tag, "items": len(items), "complet": completes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["autodistill", "eval_teachers", "heldout", "all"])
    ap.add_argument("--adapter", default=os.path.join(RUNS, "autodistill_17b", "adapter"))
    ap.add_argument("--timeout_train", type=int, default=14400)   # 4 h
    ap.add_argument("--timeout_eval", type=int, default=43200)    # 12 h
    args = ap.parse_args()

    print("PIPELINE Qwen3-1.7B -- %s" % args.stage)
    print("modele   : %s" % MODEL)
    print("adapter  : %s" % args.adapter)
    print("journal  : %s" % JOURNAL)
    t_start = time.time()

    # --- STAGE 1 : training sur les traces natives --------------------
    if args.stage in ("autodistill", "all"):
        out_dir = os.path.join(RUNS, "autodistill_17b")
        if os.path.exists(os.path.join(out_dir, "adapter", "adapter_model.safetensors")):
            print("\n[skip] adapter autodistill deja present")
            log("train:autodistill_17b", "DEJA_FAIT")
        else:
            stage_train(out_dir, AUTODISTILL, "train_autodistill_17b")

    # --- STAGE 2 : compare sur les 4 datasets Teachers ----------------
    if args.stage in ("eval_teachers", "all"):
        if not os.path.exists(os.path.join(args.adapter, "adapter_model.safetensors")):
            print("\n[ABORT] adapter absent : %s" % args.adapter)
            log("eval_teachers", "ABORT_ADAPTER_ABSENT")
            return 1
        for teacher, ds, tag in EVAL:
            r = resume_compare(tag)
            if r and r["complet"]:
                print("\n[skip] %s deja complet (%d items)" % (tag, r["items"]))
                log("compare:" + tag, "DEJA_FAIT", items=r["items"])
                continue
            stage_compare(args.adapter, ds, tag)

    # --- STAGE 3 : held-out de l'autodistill --------------------------
    if args.stage in ("heldout", "all"):
        r = resume_compare("autodistill_heldout_17b")
        if r and r["complet"]:
            print("\n[skip] heldout deja complet")
            log("compare:autodistill_heldout_17b", "DEJA_FAIT", items=r["items"])
        else:
            stage_compare(args.adapter, AUTODISTILL, "autodistill_heldout_17b")

    dt = time.time() - t_start
    print("\n" + "=" * 78)
    print("PIPELINE TERMINE en %.1f min (%.2f h)" % (dt / 60, dt / 3600))
    print("=" * 78)
    log("pipeline:%s" % args.stage, "TERMINE", minutes=round(dt / 60, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())