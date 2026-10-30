# Development for a possible round (exploratory, labelled; not a registered result): does restoring each fused
# head's output energy reduce the damage that builds up across layers? The weighted fusion is a least-squares fit, so
# each query head's fitted output O'_h C_G x carries less energy than the original O_h C_h x by exactly the fit's
# residual; with every layer converted, Llama-2-7B's residual stream ends up 4 to 12 percent smaller than the
# unconverted model's through most of the depth (llama2_followup_bisect.json, profile). This rescales every fused
# query head's output slice by s_h = sqrt(E|O_h C_h x|^2 / E|O'_h C_G x|^2) under the calibration second moment,
# and compares WikiText-2 perplexity with the plain fusion on the same pairs.
#   python alignment_study/energy_fuse_dev.py --model=JackFram/llama-160m [--calib=128] [--calib-len=2048] [--pplwin=N]
#       [--dtype=float32] [--revision=...] [--power=1.0]
import gc
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME, REVISION = arg("model", "JackFram/llama-160m"), arg("revision")
N_CAL, T_CAL = int(arg("calib", 128)), int(arg("calib-len", 2048))
N_PPL = int(arg("pplwin")) if arg("pplwin") else None
DTYPE = getattr(torch, arg("dtype", "bfloat16"))
POWER = float(arg("power", 1.0))                   # 1: restore the energy fully; 0.5: half way in log scale
OUT = f"alignment_study/energy_fuse_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


def head_energy(O, Cv, S):
    """E|O Cv x|^2 = tr(O Cv S Cv^T O^T) for an output slice O (d x hd) and value rows Cv (hd x d)."""
    M = Cv @ S @ Cv.T
    return float(((O @ M) * O).sum())


def convert(kv, S, groups, rescale):
    stats = []
    for l in range(kv.L):
        S64 = S[l].double(); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
        a, hd = kv.attn(l), kv.hd
        for G in groups[l]:
            G = list(G)
            Q = {i: list(range(i * kv.rep, (i + 1) * kv.rep)) for i in G}
            before = {h: head_energy(F.host64(a.o_proj.weight[:, h * hd:(h + 1) * hd]), F.host64(a.v_proj.weight[i * hd:(i + 1) * hd]), S64)
                      for i in G for h in Q[i]}
            K.fuse_kv_group(kv, l, G, S64, roots, qw=True, ow=True, Lsq=Lsq)
            for i in G:
                Cv = F.host64(a.v_proj.weight[i * hd:(i + 1) * hd])
                for h in Q[i]:
                    O = F.host64(a.o_proj.weight[:, h * hd:(h + 1) * hd])
                    after = head_energy(O, Cv, S64)
                    s = (before[h] / max(after, 1e-30)) ** (0.5 * POWER)
                    stats.append(after / max(before[h], 1e-30))
                    if rescale:
                        a.o_proj.weight[:, h * hd:(h + 1) * hd] = (O * s).to(a.o_proj.weight.dtype).to(a.o_proj.weight.device)
        del S64, roots, Lsq
    return stats


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    load = lambda: AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=DTYPE, attn_implementation="sdpa").eval().to(DEV)
    model = load(); kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    res = {"model": NAME, "revision": REVISION, "exploratory": True, "calibration": [N_CAL, T_CAL], "ppl_windows": int(win.shape[0]),
           "power": POWER, "base": Z.perplexity(model, win, DEV, bs=2)}
    log(f"{NAME}: unconverted ppl {res['base']:.3f}")
    S, _ = K.second_moments(model, kv, C.windows_c4(tok, N_CAL, T_CAL), DEV)
    groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    for arm, rescale in (("fuse", False), ("fuse_energy", True)):
        t = time.time()
        stats = convert(kv, S, groups, rescale)
        res[arm] = {"ppl": Z.perplexity(model, win, DEV, bs=2), "energy_kept_median": sorted(stats)[len(stats) // 2],
                    "energy_kept_min": min(stats), "seconds": round(time.time() - t, 1)}
        log(f"{arm}: ppl {res[arm]['ppl']:.3f}; fused heads keep {res[arm]['energy_kept_median']:.3f} of their output energy (median), "
            f"{res[arm]['energy_kept_min']:.3f} at least ({time.time() - t:.0f}s)")
        json.dump(res, open(OUT, "w"), indent=1)
        del model, kv
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        model = load(); kv = K.KV(model)
    res["groups"] = groups
    json.dump(res, open(OUT, "w"), indent=1)
    log("ENERGY_DEV_DONE")
