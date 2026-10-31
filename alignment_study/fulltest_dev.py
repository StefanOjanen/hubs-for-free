# Development (labelled): the plain fusion's perplexity on the whole WikiText-2 test set at 2,048 tokens for the 7B
# models whose registered values are on 8 windows (round 28's protocol, which Llama-2-7B already has: 16.78).
# Round 28's calibration (128 C4 validation documents of 2,048 tokens); the pairing of the registered rounds (the
# residual's matching, neighbouring pairs for Qwen2.5-7B); bfloat16; the unconverted model on the same windows.
#   python alignment_study/fulltest_dev.py [--only=<model>]
import gc
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

MODELS = {"mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
          "mistralai/Mistral-7B-Instruct-v0.2": "63a8b081895390a26e140280378bc85ec8bce07a",
          "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796"}
N_CAL, T_CAL = 128, 2048
OUT = "alignment_study/fulltest_dev"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    for name, rev in MODELS.items():
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if (only and name not in only) or os.path.exists(path):
            continue
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
        kv = K.KV(model)
        win = Z.wikitext2_windows(tok, T=2048)
        res = {"model": name, "revision": rev, "exploratory": True, "calibration": [N_CAL, T_CAL], "windows": int(win.shape[0]), "tokens_per_window": 2048,
               "base_full_2048": Z.perplexity(model, win, DEV, bs=2), "base_8_windows": None}
        res["base_8_windows"] = Z.perplexity(model, win[:8], DEV, bs=2)
        S, _ = K.second_moments(model, kv, C.windows_c4(tok, N_CAL, T_CAL), DEV)
        if kv.has_bias(0):
            groups, kind = [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)], "neighbour"
        else:
            groups, kind = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)], "fusion-residual matching"
        K.convert(kv, S, groups, "fuse")
        res.update({"pairing": kind, "plain_8_windows": Z.perplexity(model, win[:8], DEV, bs=2), "plain_full_2048": Z.perplexity(model, win, DEV, bs=2),
                    "seconds": round(time.time() - t0, 1)})
        json.dump(res, open(path, "w"), indent=1)
        log(f"{name}: unconverted {res['base_full_2048']:.3f} (8 windows {res['base_8_windows']:.3f}); plain fusion {res['plain_full_2048']:.3f} "
            f"(8 windows {res['plain_8_windows']:.3f}) on {res['windows']} windows")
        del model, kv, S; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
    log("FULLTEST_DONE")
