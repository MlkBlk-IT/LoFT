import argparse, json, time, os
import torch, psutil, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

P = "D:/LoFT/models/gemma-3-1b-it"
PROMPT = "Un train part à 14h20 et roule 2h45. À quelle heure arrive-t-il ? Explique en étapes courtes."

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_new_tokens", type=int, default=200)
ap.add_argument("--do_sample", choices=["true", "false"])
ap.add_argument("--temperature", type=float)
ap.add_argument("--top_k", type=int)
ap.add_argument("--top_p", type=float)
ap.add_argument("--repetition_penalty", type=float)
ap.add_argument("--dtype", choices=["float32", "bfloat16"])
a = ap.parse_args()

gen = {"max_new_tokens": a.max_new_tokens}
if a.do_sample:
    gen["do_sample"] = a.do_sample == "true"
for k in ("temperature", "top_k", "top_p", "repetition_penalty"):
    v = getattr(a, k)
    if v is not None:
        gen[k] = v
kw = {"dtype": getattr(torch, a.dtype)} if a.dtype else {}

t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, **kw)
m.eval()
t_load = time.perf_counter() - t0

enc = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
)
n_in = enc["input_ids"].shape[1]
torch.manual_seed(a.seed)
t1 = time.perf_counter()
with torch.no_grad():
    out = m.generate(**enc, **gen)
dt = time.perf_counter() - t1

new = out[0, n_in:]
mi = psutil.Process().memory_info()
res = {
    "tag": a.tag,
    "prompt": PROMPT,
    "gen_kwargs": gen,
    "seed": a.seed,
    "model_dtype": str(m.dtype),
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    "torch_threads": torch.get_num_threads(),
    "generation_config_object": json.loads(m.generation_config.to_json_string(use_diff=False)),
    "first_input_ids": enc["input_ids"][0, :6].tolist(),
    "ids_0_1_2_106_as_tokens": tok.convert_ids_to_tokens([0, 1, 2, 106]),
    "input_tokens": n_in,
    "new_tokens": int(new.shape[0]),
    "stopped_on_eos": int(new[-1]) in (1, 106),
    "load_s": round(t_load, 2),
    "generate_s": round(dt, 2),
    "tokens_per_s": round(new.shape[0] / dt, 2),
    "rss_gb": round(mi.rss / 1e9, 2),
    "peak_wset_gb": round(getattr(mi, "peak_wset", 0) / 1e9, 2) if getattr(mi, "peak_wset", None) else None,
    "answer": tok.decode(new, skip_special_tokens=True),
}
os.makedirs("D:/LoFT/tests/results", exist_ok=True)
with open(f"D:/LoFT/tests/results/{a.tag}.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))

