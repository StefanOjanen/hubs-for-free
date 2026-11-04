# Development diagnostic (exploratory, labelled): does a merged pair's damage follow the sequence start rather than the
# text? pair_text_diag.py found the whole cost of pairs (1, 5) and (0, 1) in layer 0 of Mistral-7B-Instruct-v0.2 in the
# first 25 positions of WikiText-2's first window, which opens the test file with a bare space and two newlines; the
# second WikiText window (mid-document) and both C4 windows are quiet, and no window carries the beginning-of-sequence
# token. For each listed pair fused alone: the per-position KL(unconverted || fused) on the same windows under four
# variants, as they are, with the beginning-of-sequence token prepended, with the first three tokens dropped, and with
# WikiText's three opening tokens prepended; per variant the mean over all positions and over positions 0 to 31, 32 to
# 127 and 128 on, the maximum and its position, and the full per-position KL (rounded) for plotting. Writes
# alignment_study/pair_start_diag/<model>_L<layer>.json.
#   python alignment_study/pair_start_diag.py --model=mistralai/Mistral-7B-Instruct-v0.2 --layer=0 --pairs=1-5,0-1 [--windows=2]
import json
import os
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
sys.argv = [argv0[0], "--model=none"]
import pair_text_diag as PT                            # the texts, the fusion's instruments, logprobs
sys.argv = argv0
QC, C, F, K, kvfuse, Z, DEV = PT.QC, PT.C, PT.F, PT.K, PT.kvfuse, PT.Z, PT.DEV
arg = PT.arg
NAME = arg("model")
if not NAME:
    sys.exit("--model=<name>")
LAYER, N_WIN, N_CAL, T_CAL = int(arg("layer", 0)), int(arg("windows", 2)), 128, 2048
PAIRS = [tuple(int(x) for x in p.split("-")) for p in arg("pairs", "1-5,0-1").split(",")]
OUT = f"alignment_study/pair_start_diag/{NAME.split('/')[-1]}_L{LAYER}.json"
torch.set_grad_enabled(False)


def variants(w, bos, wt_start):
    w = torch.as_tensor(w).view(-1)
    return {"as_is": w, "bos_prepended": torch.cat([torch.tensor([bos]), w[:-1]]), "first_3_dropped": w[3:],
            "wikitext_start_prepended": torch.cat([wt_start, w[:-3]])}


def profile(kl):
    kl = kl.numpy(); T = len(kl)
    return {"mean": float(kl.mean()), "mean_0_31": float(kl[:32].mean()), "mean_32_127": float(kl[32:128].mean()), "mean_128_on": float(kl[128:].mean()),
            "max": float(kl.max()), "argmax": int(kl.argmax()), "positions": T, "kl": [round(float(x), 4) for x in kl]}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = QC.AutoTokenizer.from_pretrained(NAME, revision=QC.REVISIONS.get(NAME))
    model = QC.load(NAME); kv = K.KV(model); n = kv.nkv
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    texts = {"wikitext": Z.wikitext2_windows(tok, T=2048)[:N_WIN], "c4": C.windows_c4(tok, N_CAL + 1 + N_WIN, T_CAL)[-N_WIN:]}
    wt_start = torch.as_tensor(texts["wikitext"][0]).view(-1)[:3].clone()
    S, ntok = K.second_moments(model, kv, cal, DEV)
    W = {name: [variants(w, tok.bos_token_id, wt_start) for w in X] for name, X in texts.items()}
    base = {name: [{v: PT.logprobs(model, ids) for v, ids in vs.items()} for vs in L] for name, L in W.items()}
    log(f"{NAME}: layer {LAYER}, {n} KV heads; moments from {ntok} tokens; {N_WIN} windows of each text, four variants; pairs {PAIRS}; "
        f"wikitext start tokens {tok.convert_ids_to_tokens([int(t) for t in wt_start])}, bos {tok.bos_token_id}")
    S64 = S[LAYER].double(); cost = kvfuse.pair_costs(kv, LAYER, S64); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
    orig = [p.detach().cpu().clone() for p in kv.params(LAYER)]
    def put():
        for p, t in zip(kv.params(LAYER), orig):
            p.copy_(t.to(p.device))
    res = {"model": NAME, "revision": QC.REVISIONS.get(NAME), "exploratory": True, "layer": LAYER, "windows": N_WIN, "bos": tok.bos_token_id,
           "wikitext_start_tokens": tok.convert_ids_to_tokens([int(t) for t in wt_start]), "pairs": {}}
    for i, j in PAIRS:
        put(); K.fuse_kv_group(kv, LAYER, [i, j], S64, roots, qw=True, ow=True, Lsq=Lsq)
        entry = {"residual": float(cost[i, j]), "texts": {}}
        for name, L in W.items():
            entry["texts"][name] = []
            for wi, vs in enumerate(L):
                row = {}
                for v, ids in vs.items():
                    lp = PT.logprobs(model, ids); blp = base[name][wi][v]; kl = (blp.exp() * (blp - lp)).sum(-1)
                    row[v] = profile(kl)
                entry["texts"][name].append(row)
                log(f"pair ({i}, {j}) {name} window {wi}: " + "; ".join(f"{v} mean {r['mean']:.4f} (0-31 {r['mean_0_31']:.3f}, 32-127 {r['mean_32_127']:.4f}, 128+ {r['mean_128_on']:.4f}; max {r['max']:.1f} at {r['argmax']})" for v, r in row.items()))
        res["pairs"][f"{i}-{j}"] = entry
        json.dump(res, open(OUT, "w"))
    put()
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(OUT, "w"))
    log("PAIR_START_DIAG_DONE")
