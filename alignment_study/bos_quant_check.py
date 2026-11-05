# Development check (exploratory, labelled): does a quantization cost measured on windows without the beginning-of-sequence
# token carry a sink-start share, as the fusion's does (bos_ppl_check.py: a fifth on Mistral-7B-Instruct, three percent on
# Mistral-7B-v0.1)? The unconverted model's attention projections (q, k, v, o) rounded to 4 and 3 bits per output channel
# by round 30's asymmetric min-max round-to-nearest, nothing else changed, evaluated on the 8 registered WikiText-2 windows
# as they are and with the token prepended (the last token dropped). Question rather than prediction: the rounding is
# uniform over heads, so its share is expected below the fusion's, which falls on the sink-serving heads it merges.
# Writes alignment_study/bos_quant/<model>.json.
#   python alignment_study/bos_quant_check.py --model=mistralai/Mistral-7B-Instruct-v0.2 [--windows=8] [--bits=4,3] [--exempt-layers=all,0,0+1,mid]
# --exempt-layers runs one arm per listed set of layers left unrounded (all = none exempt; mid = the middle layer; last = the last),
# writes bos_quant/<model>_exempt.json; the design and prediction are the parallel session's (private/drafts/DEV_quant_early_layers.md).
import json
import math
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
import retrieval_protect_dev as R                     # load (bfloat16, pinned revisions), K, Z, DEV
sys.argv = argv0
K, Z, DEV = R.K, R.Z, R.DEV
N_WIN, BITS = int(arg("windows", 8)), [int(b) for b in arg("bits", "4,3").split(",")]
EXEMPT = arg("exempt-layers")                           # e.g. all,0,0+1,mid: arms with these layers left unrounded (development, 2026-11-05)
OUT = f"alignment_study/bos_quant/{NAME.split('/')[-1]}{'_exempt' if EXEMPT else ''}.json"
torch.set_grad_enabled(False)


def rtn(w, bits):
    """Asymmetric min-max round-to-nearest per output channel (round 30's rounding, llama2_quant_sink_probe.rtn with group=None)."""
    W = w.float(); lo, hi = W.amin(-1, keepdim=True), W.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8); zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(W / scale) + zero, 0, 2 ** bits - 1)
    return ((q - zero) * scale).to(w.dtype)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    tok = R.AutoTokenizer.from_pretrained(NAME, revision=R.REV)
    if tok.bos_token_id is None:
        print(f"{NAME}: the tokenizer has no beginning-of-sequence token; the check does not apply", flush=True); sys.exit(0)
    win = Z.wikitext2_windows(tok, T=2048)[:N_WIN]; win = torch.as_tensor(win) if not torch.is_tensor(win) else win
    win_bos = torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1)
    model = R.load(); kv = K.KV(model)
    res = {"model": NAME, "revision": R.REV, "exploratory": True, "windows": N_WIN, "bos": tok.bos_token_id, "rounding": "asymmetric min-max round-to-nearest per output channel of q, k, v, o", "arms": {}}
    ev = lambda: {"as_is": Z.perplexity(model, win, DEV, bs=2), "bos_prepended": Z.perplexity(model, win_bos, DEV, bs=2)}
    res["arms"]["unconverted"] = ev(); u = res["arms"]["unconverted"]
    log(f"{NAME}, unconverted: {u['as_is']:.3f} as is, {u['bos_prepended']:.3f} with the token")
    snap = {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
    def parse(spec):
        return [] if spec == "all" else [kv.L // 2] if spec == "mid" else [kv.L - 1] if spec == "last" else [int(x) for x in spec.split("+")]
    SETS = [("", [])] if not EXEMPT else [(f"_exempt_{spec}", parse(spec)) for spec in EXEMPT.split(",")]
    for bits in BITS:
        for name, ex in SETS:
            for l in range(kv.L):
                for p, t in zip(kv.params(l), snap[l]):
                    p.copy_(t)
                if l in ex:
                    continue
                a = kv.attn(l)
                for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    m.weight.copy_(rtn(m.weight, bits))
            r = ev(); r["exempt_layers"] = ex; r["ratio_as_is"] = r["as_is"] / u["as_is"]; r["ratio_bos_prepended"] = r["bos_prepended"] / u["bos_prepended"]
            r["sink_start_share"] = 1 - math.log(r["ratio_bos_prepended"]) / math.log(r["ratio_as_is"]) if r["ratio_as_is"] > 1 and r["ratio_bos_prepended"] > 1 else None
            res["arms"][f"w{bits}{name}"] = r; json.dump(res, open(OUT, "w"), indent=1)
            log(f"{NAME}, w{bits}{name}: {r['as_is']:.3f} as is (ratio {r['ratio_as_is']:.3f}), {r['bos_prepended']:.3f} with the token (ratio {r['ratio_bos_prepended']:.3f}); sink-start share "
                + (f"{100 * r['sink_start_share']:.0f} percent" if r["sink_start_share"] is not None else "undefined"))
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(OUT, "w"), indent=1)
    log("BOS_QUANT_DONE")
