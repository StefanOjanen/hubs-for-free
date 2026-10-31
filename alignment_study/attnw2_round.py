# PREREGISTRATION32: the fusion fitted in the metrics the attention computation implies. The weighted fusion of rounds
# 26 to 30 fits each group's shared key and value head in the metric of S = E[x x^T], every token alike; round 31's
# replacement, each token weighted by the attention it receives, failed (0 of 3). This round fits the keys in S_w (each
# token weighted by the first-order output sensitivity of its logits), the query energies by the same sensitivity at
# each query position, and the values in S_bar (the second moment of the attention-mixed inputs, exact for the value
# error with the attention fixed): the arm picked by the rule committed before its development values were read
# (PREREGISTRATION32_rule.md; attnw_mix_dev/rule_result.json). Each model is converted to half its key-value heads
# twice, with S throughout and with this arm, the pairs fixed at the fusion's matching under S (neighbouring pairs
# where the projections carry biases); WikiText-2 perplexity on the first 8 windows of 2,048 tokens, for Llama-2-7B
# also on the whole test set. Instruments: attnw2_common.py (checked against attnw_mix_dev.py by attnw2_check.py).
#   python alignment_study/attnw2_round.py --registered=<freeze permalink> [--only=<model>]
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
import attnw2_common as W
from scale_round_v2 import DEV
from hubsfree.adapters import device_label
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


REGISTERED = arg("registered")
if not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION32.md>")
NEW = ["HuggingFaceTB/SmolLM2-1.7B", "unsloth/Llama-3.2-1B", "mistralai/Mistral-7B-v0.1"]
SEEN = ["allenai/OLMo-1B-hf", "h2oai/h2o-danube-1.8b-base", "mistralai/Mistral-7B-Instruct-v0.2", "NousResearch/Llama-2-7b-hf",
        "Qwen/Qwen2.5-7B"]
REVISIONS = {"HuggingFaceTB/SmolLM2-1.7B": "effd688a12921b4cc83e3312b6feb579f70f9c71",
             "unsloth/Llama-3.2-1B": "9535bd9b1d1dea6acafbdc4813b728796aeb28da",
             "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
             "allenai/OLMo-1B-hf": "aee7752d9c08ee4775e9b0091426d8410e8f6a89",
             "h2oai/h2o-danube-1.8b-base": "9f9de98d6afb833c23c16a74e88cb4c68825c9a5",
             "mistralai/Mistral-7B-Instruct-v0.2": "63a8b081895390a26e140280378bc85ec8bce07a",
             "NousResearch/Llama-2-7b-hf": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
             "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796"}
FULL_TEST = {"NousResearch/Llama-2-7b-hf"}
OUT = "alignment_study/attnw2"
BF16_ABOVE, N_CAL, T_CAL, N_PPL = 4e9, 128, 2048, 8
torch.set_grad_enabled(False)


def load(name):
    m = AutoModelForCausalLM.from_pretrained(name, revision=REVISIONS[name], dtype=torch.bfloat16, attn_implementation="sdpa")
    n = sum(p.numel() for p in m.parameters())
    if n < BF16_ABOVE:
        m = m.float()
    return m.eval().to(DEV), n


def free():
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()


def run_model(name, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, revision=REVISIONS[name])
    model, nparam = load(name)
    kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)
    w8 = win[:N_PPL]
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    res = {"model": name, "tier": "new" if name in NEW else "round 31", "params": nparam, "dtype": str(next(model.parameters()).dtype),
           "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "query_stride": W.QSTRIDE, "base": Z.perplexity(model, w8, DEV, bs=2)}
    S, _ = K.second_moments(model, kv, cal, DEV)
    if kv.has_bias(0) or kv.r != kv.hd:
        groups, kind = [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)], "neighbour"
    else:
        groups, kind = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)], "fusion-residual matching"
    res["pairing"], res["groups"] = kind, groups
    t = time.time()
    K.convert(kv, S, groups, "fuse")
    res["plain"] = {"ppl": Z.perplexity(model, w8, DEV, bs=2)}
    if name in FULL_TEST:
        res["plain"]["ppl_full_2048"] = Z.perplexity(model, win, DEV, bs=2)
    res["plain"]["seconds"] = round(time.time() - t, 1)
    log(f"  {name}: unconverted {res['base']:.3f}, plain {res['plain']['ppl']:.3f}" + (f", whole test {res['plain']['ppl_full_2048']:.3f}" if name in FULL_TEST else ""))
    del model, kv, S; free()
    model, _ = load(name); kv = K.KV(model)
    t = time.time()
    Sw, Sb, E = W.metrics(model, kv, cal, list(range(kv.L)), DEV)
    res["metrics_seconds"] = round(time.time() - t, 1)
    for l in range(kv.L):
        W.fuse_layer(kv, l, groups[l], Sw.pop(l), E.pop(l), Sb.pop(l))
    res["sens"] = {"ppl": Z.perplexity(model, w8, DEV, bs=2)}
    if name in FULL_TEST:
        res["sens"]["ppl_full_2048"] = Z.perplexity(model, win, DEV, bs=2)
    res["sens"]["seconds"] = round(time.time() - t, 1)
    log(f"  {name}: sensitivity arm {res['sens']['ppl']:.3f}" + (f", whole test {res['sens']['ppl_full_2048']:.3f}" if name in FULL_TEST else ""))
    del model, kv, Sw, Sb, E; free()
    res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    only = arg("only")
    for name in NEW + SEEN:
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if (only and name != only) or os.path.exists(path):
            continue
        res = run_model(name, log)
        res["registered"] = REGISTERED
        json.dump(res, open(path, "w"), indent=1)
        free()
    log("ATTNW2_ROUND_DONE")
