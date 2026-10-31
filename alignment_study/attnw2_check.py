# Check of the round 32 instruments (attnw2_common.py) against the development script they condense: on Llama-160m with
# the development protocol (32 C4 documents of 1,024 tokens, 20 WikiText-2 windows of 2,048), the plain fusion through
# fuse_kv.second_moments and fuse_kv.convert, and the sensitivity arm through attnw2_common.metrics and fuse_layer
# (query energies per plane instead of the d x d S_u), against attnw_mix_dev's plain and sens values.
#   python alignment_study/attnw2_check.py [--model=JackFram/llama-160m] [--pplwin=20]
import gc
import json
import os
import sys
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import kvfuse
import zeroshot as Z
import attnw2_common as W
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "JackFram/llama-160m")
N_PPL = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pplwin=")), 20))
torch.set_grad_enabled(False)

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(NAME)
    load = lambda: AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    model = load(); kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, 32, 1024)
    S, _ = K.second_moments(model, kv, cal, DEV)
    groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    dev = json.load(open(f"alignment_study/attnw_mix_dev/{NAME.split('/')[-1]}.json"))
    res = {"model": NAME, "dev_plain": dev["arms"]["plain"]["ppl"], "dev_sens": dev["arms"]["sens"]["ppl"]}
    K.convert(kv, S, groups, "fuse")
    res["plain"] = Z.perplexity(model, win, DEV, bs=2)
    del model, kv, S; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    model = load(); kv = K.KV(model)
    Sw, Sb, E = W.metrics(model, kv, cal, list(range(kv.L)), DEV)
    for l in range(kv.L):
        W.fuse_layer(kv, l, groups[l], Sw[l], E[l], Sb[l])
    res["sens"] = Z.perplexity(model, win, DEV, bs=2)
    res["plain_rel_diff"] = res["plain"] / res["dev_plain"] - 1
    res["sens_rel_diff"] = res["sens"] / res["dev_sens"] - 1
    os.makedirs("alignment_study/attnw2_check", exist_ok=True)
    json.dump(res, open(f"alignment_study/attnw2_check/{NAME.split('/')[-1]}.json", "w"), indent=1)
    print(res, flush=True)
    print("ATTNW2_CHECK_DONE", flush=True)
