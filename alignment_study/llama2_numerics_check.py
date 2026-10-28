# Development check for PREREGISTRATION28 (labeled): how much of a converted model's error is the rounding of
# rewritten weights to bfloat16, and whether a change of gauge removes it. The fusion writes each shared key
# plane as a whitened unit vector (the queries absorb the scale) and each shared value projection in a whitened
# basis (the output projections absorb the rest); both are exact in float64 but take new rounding when stored in
# bfloat16. The gauge-fixed variant computes each layer's conversion on the host in float64, then rescales each
# shared key plane by a complex number s (queries take 1/conj(s)) and multiplies each shared value projection by
# an invertible matrix M (output projections take M^-1), chosen so that groups of one give back the original
# weights; it rounds once. In exact arithmetic the function is unchanged. The conversion on the host is
# fuse_gauge.convert (the code round 28 would use).
#   python alignment_study/llama2_numerics_check.py <model> [ncal] [tcal] [ppl windows] [--float32]
import json
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import fuse_gauge as FG
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)


if __name__ == "__main__":
    name = sys.argv[1]
    ncal, tcal, nwin = (int(sys.argv[k]) if len(sys.argv) > k else dflt for k, dflt in ((2, 128), (3, 2048), (4, 16)))
    dtype = torch.float32 if "--float32" in sys.argv else torch.bfloat16
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    cal = C.windows_c4(tok, ncal, tcal)
    S, ntok = K.second_moments(model, kv, cal, DEV)
    X = Z.wikitext2_windows(tok, T=1024)[:nwin]
    x0 = cal[:1, :256].to(DEV)
    snap = [[p.detach().cpu().clone() for p in kv.params(l)] for l in range(kv.L)]

    def restore():
        for l in range(kv.L):
            for p, s_ in zip(kv.params(l), snap[l]):
                p.data = s_.to(DEV)
    ref = model(input_ids=x0).logits.float()
    base = Z.perplexity(model, X, DEV)
    log(f"{name}: {ntok} calibration tokens; unconverted perplexity {base:.4f} ({nwin} windows of 1,024)")
    out = {"model": name, "dtype": str(dtype), "calibration": [ncal, tcal], "windows": nwin, "base_ppl": base, "cases": {}}
    ones = [[[i] for i in range(kv.nkv)] for _ in range(kv.L)]
    pairs = [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)]
    for label, groups, method, how in (("ones/fuse/as_round27", ones, "fuse", "device"), ("ones/fuse/host64", ones, "fuse", "host"),
                                       ("ones/fuse/gauge", ones, "fuse", "gauge"), ("pairs/fuse/as_round27", pairs, "fuse", "device"),
                                       ("pairs/fuse/gauge", pairs, "fuse", "gauge"), ("pairs/fuse_u/as_round27", pairs, "fuse_u", "device"),
                                       ("pairs/fuse_u/gauge", pairs, "fuse_u", "gauge")):
        t = time.time()
        if how == "device":
            K.convert(kv, S, groups, method)
        else:
            FG.convert(kv, S, groups, method, gauge=(how == "gauge"))
        lg = model(input_ids=x0).logits.float()
        d = (lg - ref).abs().amax(-1)[0]
        p = Z.perplexity(model, X, DEV)
        out["cases"][label] = {"max_logit_diff": float(d.max()), "argmax_position": int(d.argmax()), "median_position_max_diff": float(d.median()), "ppl": p}
        log(f"{label:24s} max logit diff {float(d.max()):8.3f} (position {int(d.argmax())}, median over positions {float(d.median()):.3f}); ppl {p:.4f} ({time.time() - t:.0f}s)")
        restore()
    json.dump(out, open(f"alignment_study/llama2_dev/numerics_{name.split('/')[-1]}{'_float32' if dtype == torch.float32 else ''}.json", "w"), indent=1)
    log("NUMERICS_DONE")
