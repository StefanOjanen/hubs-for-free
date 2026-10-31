# Development (exploratory, labelled): how much calibration does the fusion need? The second moment from the first
# 16, 32, 64 and 128 C4 validation documents of 2,048 tokens; from each, the matched pairs and the fused model (plain
# fit); WikiText-2 perplexity on 8 windows of 2,048 tokens. Pairing and fitting are separated: each smaller moment is
# also used with the 128-document pairs, and the fraction of layers whose matching agrees with the 128-document one
# is recorded. Models small enough to hold a weight snapshot in memory (float32).
#   python alignment_study/calib_sens_dev.py --model=allenai/OLMo-1B-hf [--sizes=16,32,64,128] [--pplwin=8]
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "allenai/OLMo-1B-hf")
SIZES = sorted((int(s) for s in arg("sizes", "16,32,64,128").split(",")), reverse=True)
N_PPL, T_CAL = int(arg("pplwin", 8)), 2048
OUT = f"alignment_study/calib_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    cal = C.windows_c4(tok, SIZES[0], T_CAL)
    snap = {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
    put = lambda: [p.copy_(t) for l in range(kv.L) for p, t in zip(kv.params(l), snap[l])]
    res = {"model": NAME, "exploratory": True, "ppl_windows": N_PPL, "calibration_tokens": T_CAL, "base": Z.perplexity(model, win, DEV, bs=2), "sizes": {}}
    groups_full = None
    for n in SIZES:
        t1 = time.time()
        S, ntok = K.second_moments(model, kv, cal[:n], DEV)
        groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
        if groups_full is None:
            groups_full = groups
        r = {"tokens": ntok, "layers_agreeing_with_full": sum(g == gf for g, gf in zip(groups, groups_full)), "layers": kv.L}
        K.convert(kv, S, groups, "fuse"); r["ppl_own_pairs"] = Z.perplexity(model, win, DEV, bs=2); put()
        if groups is not groups_full:
            K.convert(kv, S, groups_full, "fuse"); r["ppl_full_pairs"] = Z.perplexity(model, win, DEV, bs=2); put()
        else:
            r["ppl_full_pairs"] = r["ppl_own_pairs"]
        r["seconds"] = round(time.time() - t1, 1); res["sizes"][str(n)] = r
        log(f"{NAME}: {n} documents ({ntok} tokens): ppl {r['ppl_own_pairs']:.3f} with its pairs, {r['ppl_full_pairs']:.3f} with the {SIZES[0]}-document pairs; "
            f"{r['layers_agreeing_with_full']} of {kv.L} layers agree")
        json.dump(res, open(OUT, "w"), indent=1)
        del S
    res["restore_check_ppl"] = Z.perplexity(model, win, DEV, bs=2)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"unconverted {res['base']:.3f}, restore check {res['restore_check_ppl']:.3f}")
    log("CALIB_SENS_DONE")
