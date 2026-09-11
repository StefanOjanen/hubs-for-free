# HEAD-MERGING ROUND (PREREGISTRATION7.md). Per-layer head-merging tolerance
# on Qwen2.5-1.5B, Qwen2.5-3B and Qwen2.5-7B, with the design calibrated on
# Qwen2.5-0.5B (merge_dev_calibration.py, labeled). Same merge operation,
# selection and evaluation sets as the calibration; N_EVAL raised to 32
# windows (16,384 tokens) and per-window losses stored for a paired
# bootstrap. Secondary pair sweep on Qwen2.5-1.5B: every within-group pair
# in six layers. One JSON per model as it completes; finished models are
# skipped on rerun.
#   python alignment_study/merge_round.py [--purge-large]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
SWEEP = {"Qwen/Qwen2.5-1.5B": [3, 8, 13, 18, 23, 27]}
N_EVAL, T_EVAL, BATCH = 32, 512, 4
OUT = "alignment_study/merge"
PURGE_LARGE = "--purge-large" in sys.argv
M.N_EVAL, M.T_EVAL, M.BATCH = N_EVAL, T_EVAL, BATCH
torch.set_grad_enabled(False)


def window_nll(model, X):
    """Mean NLL per evaluation window (nats/token)."""
    out = []
    for b in range(0, X.shape[0], BATCH):
        x = X[b:b + BATCH].to(DEV)
        logits = model(input_ids=x).logits[:, :-1].float()
        tgt = x[:, 1:]
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(x.shape[0], -1)
        out += nll.mean(1).tolist()
        del logits
    return out


def run_model(name):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    cfg = model.config
    L, n, nkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n; group = n // nkv
    print(name, "DEVICE", DEV, "DTYPE", dt, f"layers {L} heads {n} kv {nkv} head_dim {hd}", flush=True)
    sel = M.selection_windows(tok); X = M.eval_windows(tok)
    cos = M.pair_cosines(model, sel, L, n)
    base_w = window_nll(model, X); base = float(np.mean(base_w))
    print(f"baseline NLL {base:.4f} over {X.numel()} tokens ({time.time()-t0:.0f}s)", flush=True)
    rg = np.random.default_rng(0)
    within = [(i, j) for i in range(n) for j in range(i + 1, n) if i // group == j // group]
    short = name.split("/")[-1]
    rerun = {r["layer"]: r for r in json.load(open(f"alignment_study/rerun/{short}.json"))["protocol_A"]["rows"]}
    rows, bk = [], {}
    for l in range(L):
        attn = model.model.layers[l].self_attn
        c = {p: cos[l][p] for p in within}
        top = max(c, key=c.get); bot = min(c, key=c.get); rnd = within[rg.integers(len(within))]
        row = {"layer": l, "sharedE": rerun[l]["sharedE"], "smass": rerun[l]["smass"], "r1_real": rerun[l]["r1_real"], "cv_gn": rerun[l]["cv_gn"],
               "top_pair": list(top), "top_cos": round(float(c[top]), 4), "bottom_pair": list(bot), "bottom_cos": round(float(c[bot]), 4),
               "random_pair": list(rnd), "random_cos": round(float(c[rnd]), 4), "median_within_cos": round(float(np.median(list(c.values()))), 4)}
        for tag, (i, j) in (("top", top), ("random", rnd), ("bottom", bot)):
            M.merge(attn, i, j, hd, bk); w = window_nll(model, X); M.restore(attn, i, j, hd, bk)
            row[f"dloss_{tag}"] = round(float(np.mean(w)) - base, 6)
            row[f"dloss_{tag}_windows"] = [round(a - b, 6) for a, b in zip(w, base_w)]
        rows.append(row)
        print(f"  L{l:2d} sE {row['sharedE']:.2f} | top {top} cos {row['top_cos']:.3f} dloss {row['dloss_top']:+.4f} | random {rnd} dloss {row['dloss_random']:+.4f} | bottom {bot} cos {row['bottom_cos']:.3f} dloss {row['dloss_bottom']:+.4f} | {time.time()-t0:.0f}s", flush=True)
    sweep = []
    for l in SWEEP.get(name, []):
        attn = model.model.layers[l].self_attn
        for (i, j) in within:
            M.merge(attn, i, j, hd, bk); w = window_nll(model, X); M.restore(attn, i, j, hd, bk)
            sweep.append({"layer": l, "pair": [i, j], "cos": round(float(cos[l][i, j]), 4), "dloss": round(float(np.mean(w)) - base, 6)})
        print(f"  sweep layer {l}: {len(within)} pairs done ({time.time()-t0:.0f}s)", flush=True)
    check = float(np.mean(window_nll(model, X)))
    del model; release_memory(DEV)
    return {"model": name, "dtype": dt, "device": DEV, "n_heads": n, "n_kv_heads": nkv, "n_sel": M.N_SEL, "t_sel": M.T_SEL,
            "n_eval": N_EVAL, "t_eval": T_EVAL, "eval_tokens": int(X.numel()), "baseline_nll": round(base, 6),
            "baseline_windows": [round(v, 6) for v in base_w], "restore_check_abs_diff": round(abs(check - base), 8),
            "predictor": "sharedE from alignment_study/rerun (protocol A, T = 64, 48 windows)", "rows": rows, "pair_sweep": sweep,
            "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name in MODELS:
        short = name.split("/")[-1]
        path = f"{OUT}/{short}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        try:
            res = run_model(name)
            json.dump(res, open(path, "w"), indent=1)
            print(f"MODEL_DONE {short} layers={len(res['rows'])} dtype={res['dtype']} {res['runtime_s']}s", flush=True)
            if PURGE_LARGE:
                from hubsfree.adapters import param_count
                if param_count(name) >= 3e9:
                    import shutil
                    from huggingface_hub.constants import HF_HUB_CACHE
                    d = os.path.join(HF_HUB_CACHE, "models--" + name.replace("/", "--")); shutil.rmtree(d, ignore_errors=True); print("PURGED cache", d, flush=True)
        except Exception as e:
            import traceback
            print(f"MODEL_FAILED {short}: {type(e).__name__}: {str(e)[:300]}", flush=True); print(traceback.format_exc()[-800:], flush=True)
    print("MERGE_ROUND_DONE", flush=True)
