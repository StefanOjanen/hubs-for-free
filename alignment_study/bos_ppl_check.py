# Development check (exploratory, labelled): how much of a conversion's perplexity cost on the registered WikiText-2
# windows is the sequence-start effect? pair_start_diag.py found that merging layer 0's sink-serving pairs costs nothing
# when the beginning-of-sequence token opens the window and up to 0.2 nats per position when an ordinary token must act
# as the sink; the project's perplexity windows (the first 8 WikiText-2 windows of 2,048 tokens, sliced from the test
# text) carry no beginning-of-sequence token. For one model, the unconverted model and the plain fusion at half the
# key-value heads (the paper's conversion, pairs by the residual's matching): perplexity on the 8 windows as they are and
# with the beginning-of-sequence token prepended (the window's last token dropped, so the length is the same), and the
# mean negative log-likelihood over positions 0 to 31, 32 to 127 and 128 on under both. Writes
# alignment_study/bos_ppl/<model>.json.
#   python alignment_study/bos_ppl_check.py --model=mistralai/Mistral-7B-Instruct-v0.2 [--windows=8]
import json
import os
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
NAME = next((a.split("=", 1)[1] for a in argv0 if a.startswith("--model=")), None)
if not NAME:
    sys.exit("--model=<name>")
sys.argv = [argv0[0], f"--model={NAME}"]
import retrieval_protect_dev as R                     # load (bfloat16, pinned revisions), K, C, Z, DEV, groups_with
sys.argv = argv0
K, C, Z, kvfuse, DEV = R.K, R.C, R.Z, R.kvfuse, R.DEV
N_WIN = int(next((a.split("=", 1)[1] for a in argv0 if a.startswith("--windows=")), 8))
OUT = f"alignment_study/bos_ppl/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


def nll_profile(model, X):
    """Per-window mean NLL and the means over position bands; perplexity from the pooled mean."""
    tot, bands, per = [], {"0_31": [], "32_127": [], "128_on": []}, []
    for w in range(X.shape[0]):
        x = X[w:w + 1].to(DEV); logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none").cpu().numpy()
        per.append(float(nll.mean())); tot.append(nll)
        bands["0_31"].append(float(nll[:32].mean())); bands["32_127"].append(float(nll[32:128].mean())); bands["128_on"].append(float(nll[128:].mean()))
    allnll = np.concatenate(tot)
    return {"perplexity": float(np.exp(allnll.mean())), "per_window_nll": per, "band_nll": {k: float(np.mean(v)) for k, v in bands.items()}}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    tok = R.AutoTokenizer.from_pretrained(NAME, revision=R.REV)
    win = Z.wikitext2_windows(tok, T=2048)[:N_WIN]
    win = torch.as_tensor(win) if not torch.is_tensor(win) else win
    bos = torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype)
    win_bos = torch.cat([bos, win[:, :-1]], dim=1)
    cal = C.windows_c4(tok, R.ND.N_CAL, R.ND.T_CAL)
    model = R.load(); kv = K.KV(model); n, L = kv.nkv, kv.L
    res = {"model": NAME, "revision": R.REV, "exploratory": True, "windows": N_WIN, "bos": tok.bos_token_id, "arms": {}}
    res["arms"]["unconverted"] = {"as_is": nll_profile(model, win), "bos_prepended": nll_profile(model, win_bos)}
    log(f"{NAME}, unconverted: {res['arms']['unconverted']}")
    json.dump(res, open(OUT, "w"), indent=1)
    S, ntok = K.second_moments(model, kv, cal, DEV)
    costs = {l: kvfuse.pair_costs(kv, l, S[l].double()) for l in range(L)}
    groups = R.groups_with(kv, costs, set()); K.convert(kv, S, groups, "fuse")
    res["arms"]["fusion"] = {"as_is": nll_profile(model, win), "bos_prepended": nll_profile(model, win_bos), "kv_heads_kept": sum(len(g) for g in groups)}
    log(f"{NAME}, fusion: {res['arms']['fusion']}")
    for v in ("as_is", "bos_prepended"):
        res[f"ratio_{v}"] = res["arms"]["fusion"][v]["perplexity"] / res["arms"]["unconverted"][v]["perplexity"]
    log(f"ratio fused/unconverted: as is {res['ratio_as_is']:.3f}, bos prepended {res['ratio_bos_prepended']:.3f}")
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(OUT, "w"), indent=1)
    log("BOS_PPL_DONE")
