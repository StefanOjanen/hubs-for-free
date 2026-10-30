# Exploratory CPU probe: does a CAGA-residual matching collapse Llama-160m when all layers are converted,
# and how does damage grow with the number of converted layers (prefix 0..k-1)?
import json, sys, torch
sys.path.insert(0, "alignment_study")
sys.argv = [sys.argv[0], "--dev=JackFram/llama-160m"]
import importlib.util
import gqa_common as C, fuse_kv as K, zeroshot as Z
spec = importlib.util.spec_from_file_location("R", "alignment_study/llama2_round.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
spec = importlib.util.spec_from_file_location("FU", "alignment_study/llama2_caga_followup.py"); FU = importlib.util.module_from_spec(spec); spec.loader.exec_module(FU)
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False); torch.set_flush_denormal(True)
name = "JackFram/llama-160m"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").eval()
kv = K.KV(model)
S, _ = K.second_moments(model, kv, C.windows_c4(tok, 32, 512), "cpu")
win = Z.wikitext2_windows(tok, T=512)[:16]
snap = [[p.clone() for p in kv.params(l)] for l in range(kv.L)]
def restore():
    for l in range(kv.L):
        for p, s in zip(kv.params(l), snap[l]): p.copy_(s)
pairings = {"fuse_matched": [R.min_cost_pairs(R.pair_costs(kv, l, S[l].double())) for l in range(kv.L)],
            "caga_matched": [R.min_cost_pairs(FU.caga_pair_costs(kv, l, S[l].double())) for l in range(kv.L)],
            "neighbour": [[[2*i, 2*i+1] for i in range(kv.nkv // 2)] for _ in range(kv.L)]}
out = {"base": Z.perplexity(model, win, "cpu")}
print("base", out["base"], flush=True)
for pn, g in pairings.items():
    for cons in ("fuse", "caga"):
        for k in (1, 2, 4, 8, kv.L):
            gk = [g[l] if l < k else [[h] for h in range(kv.nkv)] for l in range(kv.L)]
            R.convert(kv, S, gk, cons)
            p = Z.perplexity(model, win, "cpu"); restore()
            out[f"{pn}/{cons}/k{k}"] = p; print(pn, cons, k, round(p, 2), flush=True)
json.dump({"pairings": pairings, "ppl": out}, open(sys.argv[0].replace(".py", ".json"), "w"), indent=1)
