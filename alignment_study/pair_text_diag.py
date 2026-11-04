# Development diagnostic (exploratory, labelled): is a merged pair's damage text-dependent? In layer 0 of
# Mistral-7B-Instruct-v0.2 the pair (1, 5) fused alone costs 0.101 nats of KL on the WikiText-2 ranking windows and
# 0.001 on eight held-out C4 windows (residual_select_dev/*_pairs*.json), and every matching containing it ranks in the
# worst fifth of 105 on WikiText, so a selector measured on C4 cannot see it. For each listed pair of one layer, fused
# alone with every other head untouched: the per-position KL(unconverted || fused) on the WikiText-2 ranking windows and
# on held-out C4 windows (documents after the calibration and check documents); per window the mean, the share of the
# total carried by the top one percent of positions, the twenty positions with the largest KL (the input token and the
# token being predicted), and the mean over positions whose predicted token contains a character class ("@", "=", a
# digit, a newline) against the rest. Writes alignment_study/pair_text_diag/<model>_L<layer>.json.
#   python alignment_study/pair_text_diag.py --model=mistralai/Mistral-7B-Instruct-v0.2 --layer=0 --pairs=1-5,0-1,0-3 [--windows=2]
import gc
import json
import math
import os
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
sys.argv = [argv0[0], "--model=none"]
import residual_select_dev as RS                      # QC (revisions, load), the fusion's instruments
sys.argv = argv0
QC, C, F, K, kvfuse, Z, DEV = RS.QC, RS.C, RS.F, RS.K, RS.kvfuse, RS.Z, RS.DEV


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model")
if not NAME:
    sys.exit("--model=<name>")
LAYER, N_WIN, N_CAL, T_CAL = int(arg("layer", 0)), int(arg("windows", 2)), 128, 2048
PAIRS = [tuple(int(x) for x in p.split("-")) for p in arg("pairs", "1-5,0-1,0-3").split(",")]
OUT = f"alignment_study/pair_text_diag/{NAME.split('/')[-1]}_L{LAYER}.json"
CLASSES = {"at_sign": lambda s: "@" in s, "equals": lambda s: "=" in s, "digit": lambda s: any(ch.isdigit() for ch in s), "newline": lambda s: "\n" in s}
torch.set_grad_enabled(False)


def logprobs(model, w):
    """Log-probabilities of the next token at every position but the last, on the CPU."""
    x = torch.as_tensor(w).view(1, -1).to(DEV)
    lp = torch.log_softmax(model(input_ids=x, use_cache=False).logits[0, :-1].float(), dim=-1).cpu()
    return lp


def describe(tok, w, kl, top=20):
    w = torch.as_tensor(w).tolist(); kl = kl.numpy(); T = len(kl)
    order = np.argsort(-kl); k = max(1, math.ceil(0.01 * T))
    tops = [{"position": int(i), "input_token": tok.decode([w[i]]), "predicted_token": tok.decode([w[i + 1]]), "kl": float(kl[i])} for i in order[:top]]
    cls = {}
    for name, fn in CLASSES.items():
        mask = np.array([fn(tok.decode([w[i + 1]])) for i in range(T)])
        cls[name] = {"positions": int(mask.sum()), "mean_kl": float(kl[mask].mean()) if mask.any() else None, "mean_kl_rest": float(kl[~mask].mean()) if (~mask).any() else None,
                     "share_of_total": float(kl[mask].sum() / kl.sum()) if kl.sum() > 0 else None}
    return {"mean_kl": float(kl.mean()), "top_1pct_share": float(np.sort(kl)[::-1][:k].sum() / kl.sum()) if kl.sum() > 0 else None, "max_kl": float(kl.max()), "top": tops, "classes": cls}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = QC.AutoTokenizer.from_pretrained(NAME, revision=QC.REVISIONS.get(NAME))
    model = QC.load(NAME); kv = K.KV(model); n = kv.nkv
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    texts = {"wikitext": Z.wikitext2_windows(tok, T=2048)[:N_WIN], "c4": C.windows_c4(tok, N_CAL + 1 + N_WIN, T_CAL)[-N_WIN:]}
    S, ntok = K.second_moments(model, kv, cal, DEV)
    base = {name: [logprobs(model, w) for w in X] for name, X in texts.items()}
    log(f"{NAME}: layer {LAYER} of {kv.L}, {n} KV heads; moments from {ntok} tokens; {N_WIN} windows of each text; pairs {PAIRS}")
    S64 = S[LAYER].double(); cost = kvfuse.pair_costs(kv, LAYER, S64); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
    orig = [p.detach().cpu().clone() for p in kv.params(LAYER)]
    def put():
        for p, t in zip(kv.params(LAYER), orig):
            p.copy_(t.to(p.device))
    res = {"model": NAME, "revision": QC.REVISIONS.get(NAME), "exploratory": True, "layer": LAYER, "windows": N_WIN, "pairs": {}}
    for i, j in PAIRS:
        put(); K.fuse_kv_group(kv, LAYER, [i, j], S64, roots, qw=True, ow=True, Lsq=Lsq)
        entry = {"residual": float(cost[i, j]), "texts": {}}
        for name, X in texts.items():
            per = []
            for w, blp in zip(X, base[name]):
                lp = logprobs(model, w); kl = (blp.exp() * (blp - lp)).sum(-1)
                per.append(describe(tok, w, kl))
            entry["texts"][name] = {"mean_kl": float(np.mean([p["mean_kl"] for p in per])), "windows": per}
        res["pairs"][f"{i}-{j}"] = entry
        json.dump(res, open(OUT, "w"), indent=1)
        log(f"pair ({i}, {j}): " + "; ".join(f"{name} KL {e['mean_kl']:.4f} (top 1 percent of positions carry {np.mean([w['top_1pct_share'] for w in e['windows']]):.2f}; "
            + ", ".join(f"{c} {np.mean([w['classes'][c]['share_of_total'] or 0 for w in e['windows']]):.2f}" for c in CLASSES) + ")" for name, e in entry["texts"].items()))
    put()
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(OUT, "w"), indent=1)
    log("PAIR_TEXT_DIAG_DONE")
