# Development for a possible round on sequential conversion (exploratory, labelled; not a registered result).
# Converts a multi-head llama-layout model to half its key-value heads with the weighted fusion, fitted
#   oneshot      on the unconverted model's attention inputs (rounds 26 to 28)
#   seq          layer by layer on the inputs the converted layers below produce (fuse_seq.py)
#   qep<alpha>   seq, with each layer's projections first corrected toward the unconverted model's input
# on pairs matched by the fusion's residual (on each layer's statistics), on fixed pairs from a file, or on
# neighbouring pairs; WikiText-2 perplexity over windows of 2,048 tokens.
#   python alignment_study/seqfuse_dev.py --model=JackFram/llama-160m [--arms=oneshot/matched,seq/matched,...]
#       [--calib=128] [--calib-len=2048] [--pplwin=N] [--pairs-file=path] [--revision=...] [--dtype=bfloat16]
import gc
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import fuse_seq as Q
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME, REVISION = arg("model", "JackFram/llama-160m"), arg("revision")
ARMS = arg("arms", "oneshot/matched,seq/matched,seq/oneshot_pairs,qep0.5/matched").split(",")
N_CAL, T_CAL = int(arg("calib", 128)), int(arg("calib-len", 2048))
N_PPL = int(arg("pplwin")) if arg("pplwin") else None
DTYPE = getattr(torch, arg("dtype", "bfloat16"))
OUT = f"alignment_study/seqfuse_dev/{NAME.split('/')[-1]}{'_' + arg('tag') if arg('tag') else ''}.json"
torch.set_grad_enabled(False)


def load():
    return AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=DTYPE, attn_implementation="sdpa").eval().to(DEV)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"model": NAME, "revision": REVISION, "exploratory": True,
                                                            "calibration": [N_CAL, T_CAL], "ppl_windows": int(win.shape[0]), "arms": {}}
    fixed = json.load(open(arg("pairs-file")))["groups"] if arg("pairs-file") else None
    model = load(); kv = K.KV(model)
    if "stream_check" not in res:                     # the layer-by-layer stream reproduces the model's hidden states
        x = cal[:1].to(DEV)
        ref = model.model(input_ids=x, use_cache=False, output_hidden_states=True).hidden_states
        h, diffs = model.model.embed_tokens(x), []
        for l in range(kv.L):
            h = Q.run_layer(model, kv.blocks[l], h)
            if l < kv.L - 1:
                diffs.append(float((h.float() - ref[l + 1].float()).abs().max() / ref[l + 1].float().abs().max()))
        res["stream_check"] = {"max_rel_diff": max(diffs)}
        log(f"stream check: largest difference relative to the largest activation {max(diffs):.2e}")
    if "base" not in res:
        res["base"] = Z.perplexity(model, win, DEV, bs=2)
        log(f"{NAME}: unconverted ppl {res['base']:.3f} on {win.shape[0]} windows")
    oneshot_pairs = None
    for arm in ARMS:
        if arm in res["arms"]:
            log(f"{arm}: cached ppl {res['arms'][arm]['ppl']:.3f}"); continue
        fit, pairing = arm.split("/")
        t = time.time()
        neighbour = lambda l: [[i, i + 1] for i in range(0, kv.nkv, 2)]
        if fit == "oneshot":
            S, _ = K.second_moments(model, kv, cal, DEV)
            groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) if pairing == "matched"
                      else fixed[l] if pairing == "file" else neighbour(l) for l in range(kv.L)]
            K.convert(kv, S, groups, "fuse")
            del S
        else:
            alpha = float(fit[3:]) if fit.startswith("qep") else 0.0
            if pairing == "oneshot_pairs" and oneshot_pairs is None:
                oneshot_pairs = res["arms"]["oneshot/matched"]["groups"]
            pick = {"matched": lambda l, S: kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S)),
                    "oneshot_pairs": lambda l, S: oneshot_pairs[l],
                    "file": lambda l, S: fixed[l],
                    "neighbour": lambda l, S: neighbour(l)}[pairing]
            groups = Q.sequential_convert(model, kv, cal, DEV, pick, alpha=alpha, log=None)
        ppl = Z.perplexity(model, win, DEV, bs=2)
        res["arms"][arm] = {"ppl": ppl, "groups": [[list(G) for G in g] for g in groups], "seconds": round(time.time() - t, 1)}
        json.dump(res, open(OUT, "w"), indent=1)
        log(f"{arm}: ppl {ppl:.3f} ({time.time() - t:.0f}s)")
        del model, kv                      # kv holds the model too: free both before loading the next copy
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        model = load(); kv = K.KV(model)
    log("SEQFUSE_DEV_DONE")
