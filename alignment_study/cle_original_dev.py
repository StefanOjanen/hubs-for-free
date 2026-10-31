# Development baseline (exploratory, labelled): is round 30's rebalancing gain specific to fused models? The same
# equalization (fuse_balance.py with every key-value head in a group of its own) applied to the original, unconverted
# model; WikiText-2 perplexity with the attention projections rounded to 4 and 3 bits per output channel, before and
# after. If the original model gains about as much, the effect is generic; if not, it comes from the fusion's
# imbalance.
#   python alignment_study/cle_original_dev.py --model=TinyLlama/TinyLlama_v1.1 [--pplwin=8]
import json
import os
import sys
import torch

sys.path.insert(0, "alignment_study")
import fuse_kv as K
import fuse_balance as B
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)
torch.set_grad_enabled(False)
NAME = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "TinyLlama/TinyLlama_v1.1")
N_PPL = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pplwin=")), 8))
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
kv = K.KV(model)
win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
ppl = lambda: Z.perplexity(model, win, DEV, bs=2)
snap = lambda: {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
put = lambda s: [p.copy_(t) for l in range(kv.L) for p, t in zip(kv.params(l), s[l])]
res = {"model": NAME, "exploratory": True, "ppl_windows": N_PPL}
for key in ("original", "equalized"):
    if key == "equalized":
        B.balance(kv, [[[i] for i in range(kv.nkv)] for _ in range(kv.L)])
    r = {"ppl": ppl()}
    s = snap()
    for bits in (4, 3):
        put(s)
        for l in range(kv.L):
            a = kv.attn(l)
            for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                m.weight.copy_(P.rtn(m.weight, bits))
        r[f"w{bits}"] = ppl()
    put(s)
    res[key] = r
    print(key, r, flush=True)
os.makedirs("alignment_study/balance_dev", exist_ok=True)
json.dump(res, open(f"alignment_study/balance_dev/cle_original_{NAME.split('/')[-1]}.json", "w"), indent=1)
