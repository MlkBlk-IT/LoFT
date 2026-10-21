import argparse, json, re, time, os
import torch, psutil, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

P = "D:/LoFT/models/gemma-3-1b-it"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--prompt_file", required=True)
ap.add_argument("--n_seeds", type=int, default=10)
ap.add_argument("--max_new_tokens", type=int, default=512)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top_k", type=int, default=64)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--repetition_penalty", type=float, default=1.0)
a = ap.parse_args()

PROMPT = open(a.prompt_file, encoding="utf-8").read().strip()
gen = dict(max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temperature,
           top_k=a.top_k, top_p=a.top_p, repetition_penalty=a.repetition_penalty)

tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16)
m.eval()
enc = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
)
n_in = enc["input_ids"].shape[1]

def n_sent(t):
    return len([s for s in re.split(r"(?<=[.!?])\s+|\n+", t.strip()) if s.strip()])

runs = []
for seed in range(a.n_seeds):
    torch.manual_seed(seed)
    t1 = time.perf_counter()
    with torch.no_grad():
        out = m.generate(**enc, **gen)
    dt = time.perf_counter() - t1
    new = out[0, n_in:]
    ans = tok.decode(new, skip_special_tokens=True)
    nd = len(re.findall(r"\d", ans))
    ns = n_sent(ans)
    r = {
        "seed": seed, "new_tokens": int(new.shape[0]),
        "stopped_on_eos": int(new[-1]) in (1, 106),
        "generate_s": round(dt, 2), "tokens_per_s": round(new.shape[0] / dt, 2),
        "n_digits": nd, "n_sentences": ns,
        "digits_ok": nd == 0, "sentences_ok": ns <= 3, "both_ok": nd == 0 and ns <= 3,
        "answer": ans,
    }
    runs.append(r)
    print("--- seed %d: %d tok, digits=%d, sentences=%d, both_ok=%s" % (seed, r["new_tokens"], nd, ns, r["both_ok"]))
    print(ans)

mi = psutil.Process().memory_info()
n = len(runs)
summary = {
    "runs": n,
    "digits_ok": sum(r["digits_ok"] for r in runs),
    "sentences_ok": sum(r["sentences_ok"] for r in runs),
    "both_ok": sum(r["both_ok"] for r in runs),
    "stopped_on_eos": sum(r["stopped_on_eos"] for r in runs),
    "mean_new_tokens": round(sum(r["new_tokens"] for r in runs) / n, 1),
    "mean_tokens_per_s": round(sum(r["tokens_per_s"] for r in runs) / n, 2),
    "peak_wset_gb": round(getattr(mi, "peak_wset", 0) / 1e9, 2) if getattr(mi, "peak_wset", None) else None,
}
res = {"tag": a.tag, "prompt": PROMPT, "gen_kwargs": gen, "model_dtype": str(m.dtype),
       "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
       "summary": summary, "runs": runs}
os.makedirs("D:/LoFT/tests/results", exist_ok=True)
with open("D:/LoFT/tests/results/%s.json" % a.tag, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("=== SUMMARY ===")
print(json.dumps(summary, indent=2))
