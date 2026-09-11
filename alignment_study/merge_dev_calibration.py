# T5.1 development calibration (labeled, not registered): per-layer
# head-merging tolerance on the development model, Qwen2.5-0.5B, to size the
# thresholds of PREREGISTRATION7 before it is frozen.
#
# Design. For each layer, candidate pairs are query heads in the same KV
# group (grouped-query attention shares K and V within a group, so a
# within-group merge touches only the two heads' query and output slices).
# The generator cosine cos(G_i, G_j) is averaged over SELECTION windows
# (WikiText-103 validation texts of >= 256 tokens, T = 256). Three pairs per
# layer: the top-cosine pair, the bottom-cosine pair, and a seeded random
# pair. Merging a pair (i, j): both heads' q_proj rows (and biases) are set
# to their average and both o_proj column blocks to their average, so the
# merged heads compute one attention pattern whose output is projected by
# W_o,i + W_o,j. The loss change is the mean next-token NLL over EVALUATION
# windows (WikiText-103 test split, concatenated and cut into 512-token
# windows, disjoint from selection) minus the unmerged loss. The predictor
# is the layer's shared-energy fraction from the committed rerun
# (alignment_study/rerun/<model>.json, protocol A, T = 64, 48 windows).
# Output: merge_dev_calibration.json.
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import generators
from scale_round_v2 import DEV
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

NAME = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "Qwen/Qwen2.5-0.5B"
N_SEL, T_SEL, N_EVAL, T_EVAL, BATCH = 16, 256, 24, 512, 4
OUT = sys.argv[2] if len(sys.argv) > 2 else "alignment_study/merge_dev_calibration.json"
torch.set_grad_enabled(False)


def selection_windows(tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    pool = [t for t in ds["text"] if len(t.split()) > 40]
    ok = [t for t in pool if len(tok(t)["input_ids"]) >= T_SEL]
    stride = max(1, len(ok) // N_SEL)
    return [tok(ok[i * stride], return_tensors="pt", truncation=True, max_length=T_SEL)["input_ids"] for i in range(N_SEL)]


def eval_windows(tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    n = ids.numel() // T_EVAL
    picks = np.linspace(0, n - 1, N_EVAL).round().astype(int)
    return torch.stack([ids[p * T_EVAL:(p + 1) * T_EVAL] for p in picks])


def mean_nll(model, X):
    tot, cnt = 0.0, 0
    for b in range(0, X.shape[0], BATCH):
        x = X[b:b + BATCH].to(DEV)
        out = model(input_ids=x, labels=x)
        ntok = x.shape[0] * (x.shape[1] - 1)
        tot += float(out.loss) * ntok; cnt += ntok
    return tot / cnt


def pair_cosines(model, sel, L, n):
    """Mean over selection windows of cos(G_i, G_j) per layer and head pair."""
    acc = np.zeros((L, n, n)); k = 0
    for x in sel:
        out = model(input_ids=x.to(DEV), output_attentions=True)
        for l in range(L):
            G = generators(out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64))
            V = G.reshape(n, -1); V = V / np.linalg.norm(V, axis=1, keepdims=True)
            acc[l] += V @ V.T
        k += 1
        del out
    return acc / k


def merge(attn, i, j, hd, backup):
    q, o = attn.q_proj, attn.o_proj
    si, sj = slice(i * hd, (i + 1) * hd), slice(j * hd, (j + 1) * hd)
    backup.clear()
    backup["qw_i"], backup["qw_j"] = q.weight[si].clone(), q.weight[sj].clone()
    backup["ow_i"], backup["ow_j"] = o.weight[:, si].clone(), o.weight[:, sj].clone()
    avg = (q.weight[si] + q.weight[sj]) / 2; q.weight[si] = avg; q.weight[sj] = avg
    if q.bias is not None:
        backup["qb_i"], backup["qb_j"] = q.bias[si].clone(), q.bias[sj].clone()
        avgb = (q.bias[si] + q.bias[sj]) / 2; q.bias[si] = avgb; q.bias[sj] = avgb
    avgo = (o.weight[:, si] + o.weight[:, sj]) / 2; o.weight[:, si] = avgo; o.weight[:, sj] = avgo


def restore(attn, i, j, hd, backup):
    q, o = attn.q_proj, attn.o_proj
    si, sj = slice(i * hd, (i + 1) * hd), slice(j * hd, (j + 1) * hd)
    q.weight[si] = backup["qw_i"]; q.weight[sj] = backup["qw_j"]
    o.weight[:, si] = backup["ow_i"]; o.weight[:, sj] = backup["ow_j"]
    if q.bias is not None:
        q.bias[si] = backup["qb_i"]; q.bias[sj] = backup["qb_j"]


def run(name):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    cfg = model.config
    L, n, nkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n; group = n // nkv
    print(name, "DEVICE", DEV, "DTYPE", dt, f"layers {L} heads {n} kv {nkv} head_dim {hd}", flush=True)
    sel = selection_windows(tok); X = eval_windows(tok)
    cos = pair_cosines(model, sel, L, n)
    base = mean_nll(model, X)
    print(f"baseline NLL {base:.4f} nats/token over {X.numel()} tokens ({time.time()-t0:.0f}s)", flush=True)
    rg = np.random.default_rng(0)
    within = [(i, j) for i in range(n) for j in range(i + 1, n) if i // group == j // group]
    short = name.split("/")[-1]
    rerun = {r["layer"]: r for r in json.load(open(f"alignment_study/rerun/{short}.json"))["protocol_A"]["rows"]}
    rows = []
    for l in range(L):
        attn = model.model.layers[l].self_attn
        c = {p: cos[l][p] for p in within}
        top = max(c, key=c.get); bot = min(c, key=c.get); rnd = within[rg.integers(len(within))]
        row = {"layer": l, "sharedE": rerun[l]["sharedE"], "smass": rerun[l]["smass"], "r1_real": rerun[l]["r1_real"],
               "top_pair": list(top), "top_cos": round(float(c[top]), 4), "bottom_pair": list(bot), "bottom_cos": round(float(c[bot]), 4),
               "random_pair": list(rnd), "random_cos": round(float(c[rnd]), 4), "median_within_cos": round(float(np.median(list(c.values()))), 4)}
        bk = {}
        for tag, (i, j) in (("top", top), ("random", rnd), ("bottom", bot)):
            merge(attn, i, j, hd, bk); row[f"dloss_{tag}"] = round(mean_nll(model, X) - base, 5); restore(attn, i, j, hd, bk)
        rows.append(row)
        print(f"  L{l:2d} sE {row['sharedE']:.2f} | top {top} cos {row['top_cos']:.3f} dloss {row['dloss_top']:+.4f} | random {rnd} cos {row['random_cos']:.3f} dloss {row['dloss_random']:+.4f} | bottom {bot} cos {row['bottom_cos']:.3f} dloss {row['dloss_bottom']:+.4f}", flush=True)
    check = mean_nll(model, X)
    se = [r["sharedE"] for r in rows]; dt_ = [r["dloss_top"] for r in rows]; layers = list(range(L))
    def partial(x, y, z):
        rx = np.polyfit(z, x, 1); ry = np.polyfit(z, y, 1)
        return float(spearmanr(np.asarray(x) - np.polyval(rx, z), np.asarray(y) - np.polyval(ry, z))[0])
    summ = {"baseline_nll": round(base, 5), "restore_check_abs_diff": round(abs(check - base), 7),
            "spearman_sharedE_dloss_top": round(float(spearmanr(se, dt_)[0]), 3),
            "spearman_sharedE_dloss_top_depth_partialled": round(partial(se, dt_, layers), 3),
            "spearman_layer_dloss_top": round(float(spearmanr(layers, dt_)[0]), 3),
            "spearman_sharedE_dloss_random": round(float(spearmanr(se, [r["dloss_random"] for r in rows])[0]), 3),
            "frac_top_below_random": round(float(np.mean([r["dloss_top"] < r["dloss_random"] for r in rows])), 3),
            "frac_top_below_bottom": round(float(np.mean([r["dloss_top"] < r["dloss_bottom"] for r in rows])), 3),
            "median_dloss": {k: round(float(np.median([r[f"dloss_{k}"] for r in rows])), 5) for k in ("top", "random", "bottom")},
            "runtime_s": round(time.time() - t0, 1)}
    print("SUMMARY", json.dumps(summ), flush=True)
    del model; release_memory(DEV)
    return {"model": name, "dtype": dt, "device": DEV, "n_sel": N_SEL, "t_sel": T_SEL, "n_eval": N_EVAL, "t_eval": T_EVAL,
            "eval_tokens": int(X.numel()), "predictor": "sharedE from alignment_study/rerun (protocol A)", "rows": rows, "summary": summ}


if __name__ == "__main__":
    print("DEVICE", device_label(DEV), flush=True)
    res = run(NAME)
    json.dump(res, open(OUT, "w"), indent=1)
    print("MERGE_DONE", OUT, flush=True)
