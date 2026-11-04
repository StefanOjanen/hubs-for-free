# Development test (exploratory, labelled; design in private/drafts/DEV_token_first_calibration.md, drafted by the
# parallel session 2026-11-04): does fitting the fusion's second moments on token-first calibration windows lower the
# token-first cost? Every calibration window of every round carries no beginning-of-sequence token
# (gqa_common.windows_c4), so the sink-serving heads are fitted in the sinkless regime deployment never shows them;
# pair_start_diag.py and bos_ppl_check.py found that regime behind 21 percent of Mistral-7B-Instruct's fused cost and
# most of its retrieval loss, and 3 percent of Mistral-7B-v0.1's. Prediction stated before running: token-first
# calibration lowers the Instruct's token-first perplexity by more than the base's. Per model, a 2 x 2: moments on
# token-less and token-first windows (the flag bos=True of windows_c4: the token prepended, the length kept), the pairs
# fixed to the token-less fit's residual matching so only the fit changes, each conversion at half the key-value heads
# evaluated on the 8 registered WikiText-2 windows as they are and with the token prepended (perplexity and the
# negative log-likelihood over positions 0 to 31, 32 to 127 and 128 on), plus the registered needle protocol (ten codes
# at five depths, 50 instances, needle_dev's decoding) with the token prepended, on both conversions. Writes
# alignment_study/token_first_calib_dev/<model>.json after every arm.
#   python alignment_study/token_first_calib_dev.py --model=mistralai/Mistral-7B-Instruct-v0.2 [--calib=128] [--calib-len=2048] [--windows=8] [--no-needle]
import gc
import json
import os
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
arg = lambda n, d=None: next((a.split("=", 1)[1] for a in argv0 if a.startswith(f"--{n}=")), d)
NAME = arg("model")
if not NAME:
    sys.exit("--model=<name>")
sys.argv = [argv0[0], f"--model={NAME}"]
import retrieval_protect_dev as R                     # load (bfloat16, pinned revisions), K, C, Z, DEV, groups_with, ND
sys.argv = argv0
K, C, Z, kvfuse, DEV, ND = R.K, R.C, R.Z, R.kvfuse, R.DEV, R.ND
N_CAL, T_CAL, N_WIN, NEEDLE = int(arg("calib", 128)), int(arg("calib-len", 2048)), int(arg("windows", 8)), "--no-needle" not in argv0
OUT = f"alignment_study/token_first_calib_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


def nll_profile(model, X):
    tot, bands, per = [], {"0_31": [], "32_127": [], "128_on": []}, []
    for w in range(X.shape[0]):
        x = X[w:w + 1].to(DEV); logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none").cpu().numpy()
        per.append(float(nll.mean())); tot.append(nll)
        bands["0_31"].append(float(nll[:32].mean())); bands["32_127"].append(float(nll[32:128].mean())); bands["128_on"].append(float(nll[128:].mean()))
    allnll = np.concatenate(tot)
    return {"perplexity": float(np.exp(allnll.mean())), "per_window_nll": per, "band_nll": {k: float(np.mean(v)) for k, v in bands.items()}}


def needle_eval_bos(model, tok, context_ids):
    """The registered needle protocol with the beginning-of-sequence token prepended to every prompt."""
    inst = []
    for colour, code in ND.FACTS:
        needle = f" The secret code for the {colour} door is {code}. "
        question = f" Based on the content of the book, Question: What is the secret code for the {colour} door?\nAnswer:"
        for depth in ND.DEPTHS:
            ids = ND.build(tok, context_ids, needle, question, depth)
            ids = torch.cat([torch.tensor([[tok.bos_token_id]], dtype=ids.dtype), ids], dim=1)
            a = ND.answer(model, tok, ids)
            inst.append({"colour": colour, "code": code, "depth": depth, "answer": a, "hit": code in a})
    return {"hits": sum(i["hit"] for i in inst), "of": len(inst), "by_depth": {str(d): sum(i["hit"] for i in inst if i["depth"] == d) for d in ND.DEPTHS}, "instances": inst}


def evaluate(model, tok, win, win_bos, context_ids, needle):
    out = {"as_is": nll_profile(model, win), "bos_prepended": nll_profile(model, win_bos)}
    if needle:
        out["needle_bos"] = needle_eval_bos(model, tok, context_ids)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    tok = R.AutoTokenizer.from_pretrained(NAME, revision=R.REV)
    if tok.bos_token_id is None:
        print(f"{NAME}: the tokenizer has no beginning-of-sequence token; the test does not apply", flush=True); sys.exit(0)
    ND.FACTS, ND.DEPTHS = R.REG_FACTS, R.REG_DEPTHS
    win = Z.wikitext2_windows(tok, T=2048)[:N_WIN]; win = torch.as_tensor(win) if not torch.is_tensor(win) else win
    win_bos = torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1)
    context_ids = tok.encode(ND.haystack_text(3 * ND.CTX), add_special_tokens=False) if NEEDLE else None
    cal_less = C.windows_c4(tok, N_CAL, T_CAL); cal_first = C.windows_c4(tok, N_CAL, T_CAL, bos=True)
    assert int(cal_first[0, 0]) == tok.bos_token_id and cal_first.shape == cal_less.shape
    model = R.load(); kv = K.KV(model); n, L = kv.nkv, kv.L
    res = {"model": NAME, "revision": R.REV, "exploratory": True, "design": "private/drafts/DEV_token_first_calibration.md", "calibration": [N_CAL, T_CAL], "windows": N_WIN,
           "bos": tok.bos_token_id, "needle": {"context_tokens": ND.CTX, "facts": len(ND.FACTS), "depths": list(ND.DEPTHS)} if NEEDLE else None, "arms": {}}
    res["arms"]["unconverted"] = evaluate(model, tok, win, win_bos, context_ids, NEEDLE)
    json.dump(res, open(OUT, "w"), indent=1); log(f"{NAME}: unconverted {res['arms']['unconverted']['as_is']['perplexity']:.3f} as is, {res['arms']['unconverted']['bos_prepended']['perplexity']:.3f} with the token" + (f"; needle with the token {res['arms']['unconverted']['needle_bos']['hits']} of {res['arms']['unconverted']['needle_bos']['of']}" if NEEDLE else ""))
    S_less, ntok = K.second_moments(model, kv, cal_less, DEV); log(f"token-less moments from {ntok} tokens")
    S_first, ntok2 = K.second_moments(model, kv, cal_first, DEV); log(f"token-first moments from {ntok2} tokens")
    costs = {l: kvfuse.pair_costs(kv, l, S_less[l].double()) for l in range(L)}
    groups = R.groups_with(kv, costs, set()); res["groups"] = groups; res["pairs_fixed_to"] = "the token-less fit's residual matching"
    for name, S in (("fit_token_less", S_less), ("fit_token_first", S_first)):
        del model, kv; R.free(); model = R.load(); kv = K.KV(model)
        K.convert(kv, S, groups, "fuse")
        res["arms"][name] = evaluate(model, tok, win, win_bos, context_ids, NEEDLE); res["arms"][name]["kv_heads_kept"] = sum(len(g) for g in groups)
        u = res["arms"]["unconverted"]; a = res["arms"][name]
        a["ratio_as_is"] = a["as_is"]["perplexity"] / u["as_is"]["perplexity"]; a["ratio_bos_prepended"] = a["bos_prepended"]["perplexity"] / u["bos_prepended"]["perplexity"]
        json.dump(res, open(OUT, "w"), indent=1)
        log(f"{NAME}, {name}: perplexity {a['as_is']['perplexity']:.3f} as is (ratio {a['ratio_as_is']:.3f}), {a['bos_prepended']['perplexity']:.3f} with the token (ratio {a['ratio_bos_prepended']:.3f}); "
            f"bands with the token {a['bos_prepended']['band_nll']}" + (f"; needle with the token {a['needle_bos']['hits']} of {a['needle_bos']['of']} {a['needle_bos']['by_depth']}" if NEEDLE else ""))
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(OUT, "w"), indent=1)
    log("TOKEN_FIRST_CALIB_DONE")
