import argparse, json, re, time, os
import torch, psutil, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

P = "D:/LoFT/models/gemma-3-1b-it"
PROMPT = "Un train part à 14h20 et roule 2h45. À quelle heure arrive-t-il ? Explique en étapes courtes."
EXPECTED = (17, 5)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--prompt", default=PROMPT)
ap.add_argument("--expected", default="17:05")
ap.add_argument("--n_seeds", type=int, default=10)
ap.add_argument("--max_new_tokens", type=int, default=512)
ap.add_argument("--temperature", type=float, default=0.5)
ap.add_argument("--top_k", type=int, default=64)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--repetition_penalty", type=float, default=1.05)
ap.add_argument("--greedy", action="store_true")
a = ap.parse_args()
PROMPT = a.prompt
EXPECTED = tuple(int(x) for x in a.expected.split(":"))

gen = dict(max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temperature,
           top_k=a.top_k, top_p=a.top_p, repetition_penalty=a.repetition_penalty)

if a.greedy:
    gen = dict(max_new_tokens=a.max_new_tokens, do_sample=False)
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16)
m.eval()
enc = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
)
n_in = enc["input_ids"].shape[1]
pat = re.compile(r"(\d{1,2})\s*(?:h|H|:)\s*(\d{2})")

runs = []
for seed in range(a.n_seeds):
    torch.manual_seed(seed)
    t1 = time.perf_counter()
    with torch.no_grad():
        out = m.generate(**enc, **gen)
    dt = time.perf_counter() - t1
    new = out[0, n_in:]
    ans = tok.decode(new, skip_special_tokens=True)
    found = pat.findall(ans)
    last = (int(found[-1][0]), int(found[-1][1])) if found else None
    valid = last is not None and last[0] < 24 and last[1] < 60
    r = {
        "seed": seed,
        "new_tokens": int(new.shape[0]),
        "stopped_on_eos": int(new[-1]) in (1, 106),
        "generate_s": round(dt, 2),
        "tokens_per_s": round(new.shape[0] / dt, 2),
        "last_time": None if last is None else "%02d:%02d" % last,
        "valid_time": valid,
        "correct": last is not None and last[1] == EXPECTED[1] and last[0] % 12 == EXPECTED[0] % 12,
        "answer": ans,
    }
    runs.append(r)
    print("--- seed %d: %d tok, %.2f tok/s, last_time=%s, valid=%s, correct=%s" % (
        seed, r["new_tokens"], r["tokens_per_s"], r["last_time"], valid, r["correct"]))
    print(ans)

mi = psutil.Process().memory_info()
n = len(runs)
summary = {
    "runs": n,
    "correct": sum(r["correct"] for r in runs),
    "invalid_time": sum(not r["valid_time"] for r in runs),
    "stopped_on_eos": sum(r["stopped_on_eos"] for r in runs),
    "mean_new_tokens": round(sum(r["new_tokens"] for r in runs) / n, 1),
    "mean_tokens_per_s": round(sum(r["tokens_per_s"] for r in runs) / n, 2),
    "peak_wset_gb": round(getattr(mi, "peak_wset", 0) / 1e9, 2) if getattr(mi, "peak_wset", None) else None,
}
res = {
    "tag": a.tag, "prompt": PROMPT, "expected": a.expected, "gen_kwargs": gen,
    "model_dtype": str(m.dtype),
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    "summary": summary, "runs": runs,
}
os.makedirs("D:/LoFT/tests/results", exist_ok=True)
with open("D:/LoFT/tests/results/%s.json" % a.tag, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("=== SUMMARY ===")
print(json.dumps(summary, indent=2))



