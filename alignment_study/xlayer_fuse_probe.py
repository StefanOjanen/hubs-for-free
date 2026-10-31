# Probe (exploratory, labelled): could a layer share its neighbour's key-value cache through a closed-form fit? Round
# 25 found that adjacent layers' cached keys and values are unrelated token by token, while their residual streams
# are close (cosine 0.90 to 0.94). Sharing one cache between layers l and l+1 needs layer l+1's keys and values to be
# predictable from layer l's attention input x_l. For every layer pair this measures, from second moments on
# calibration text, the share of layer l+1's key and value energy that the best linear map of x_l reproduces,
#   R^2 = 1 - tr(W (S_{l+1} - C S_l^-1 C^T) W^T) / tr(W S_{l+1} W^T),  C = E[x_{l+1} x_l^T],
# for each key-value head (values weighted by the head's output projections, keys unweighted), and, as a
# reference, the same share from x_{l+1} itself (1 by construction) and from x_{l-1} (two layers back).
#   python alignment_study/xlayer_fuse_probe.py --model=JackFram/llama-160m [--calib=32] [--calib-len=1024]
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)

NAME = arg("model", "JackFram/llama-160m")
N_CAL, T_CAL = int(arg("calib", 32)), int(arg("calib-len", 1024))
OUT = f"alignment_study/xlayer_fuse_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    d, L, hd = kv.d, kv.L, kv.hd
    S = {l: torch.zeros(d, d, dtype=torch.float64) for l in range(L)}
    X1 = {l: torch.zeros(d, d, dtype=torch.float64) for l in range(L - 1)}       # E[x_{l+1} x_l^T]
    X2 = {l: torch.zeros(d, d, dtype=torch.float64) for l in range(L - 2)}       # E[x_{l+2} x_l^T]
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    n = 0
    for w in range(cal.shape[0]):
        xs = {}
        hooks = [kv.attn(l).k_proj.register_forward_pre_hook((lambda l: lambda m, i: xs.__setitem__(l, i[0][0].float()))(l)) for l in range(L)]
        model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
        for l in range(L):
            S[l] += F.host64(xs[l].T @ xs[l])
            if l < L - 1:
                X1[l] += F.host64(xs[l + 1].T @ xs[l])
            if l < L - 2:
                X2[l] += F.host64(xs[l + 2].T @ xs[l])
        n += cal.shape[1]
    S = {l: v / n for l, v in S.items()}; X1 = {l: v / n for l, v in X1.items()}; X2 = {l: v / n for l, v in X2.items()}
    log(f"{NAME}: moments from {n} tokens")

    def shares(l_target, l_src, Xc):
        """Per KV head of layer l_target: R^2 of its values (output weighted) and keys predicted from x_{l_src}."""
        Ss = S[l_src] + 1e-6 * S[l_src].diagonal().mean() * torch.eye(d, dtype=torch.float64)
        resid_cov = S[l_target] - Xc @ torch.linalg.solve(Ss, Xc.T)                # E[x_t x_t^T] - C S^-1 C^T
        a = kv.attn(l_target)
        Wv, Wk, Wo = F.host64(a.v_proj.weight), F.host64(a.k_proj.weight), F.host64(a.o_proj.weight)
        out = []
        for i in range(kv.nkv):
            Cv, Bk = Wv[i * hd:(i + 1) * hd], Wk[i * hd:(i + 1) * hd]
            OO = sum(Wo[:, h * hd:(h + 1) * hd].T @ Wo[:, h * hd:(h + 1) * hd] for h in range(i * kv.rep, (i + 1) * kv.rep))
            tot_v = float((OO * (Cv @ S[l_target] @ Cv.T)).sum()); res_v = float((OO * (Cv @ resid_cov @ Cv.T)).sum())
            tot_k = float((Bk @ S[l_target] @ Bk.T).trace()); res_k = float((Bk @ resid_cov @ Bk.T).trace())
            out.append({"value_r2": 1 - res_v / tot_v, "key_r2": 1 - res_k / tot_k})
        return out
    res = {"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "layers": []}
    for l in range(1, L):
        prev = shares(l, l - 1, X1[l - 1])
        two = shares(l, l - 2, X2[l - 2]) if l >= 2 else None
        med = lambda xs, k: sorted(x[k] for x in xs)[len(xs) // 2]
        row = {"layer": l, "from_previous": prev, "from_two_back": two,
               "median_value_r2_previous": med(prev, "value_r2"), "median_key_r2_previous": med(prev, "key_r2"),
               "median_value_r2_two_back": med(two, "value_r2") if two else None, "median_key_r2_two_back": med(two, "key_r2") if two else None}
        res["layers"].append(row)
        log(f"layer {l}: predicted from layer {l - 1}'s input, value R^2 median {row['median_value_r2_previous']:.3f}, key {row['median_key_r2_previous']:.3f}"
            + (f"; from two back {row['median_value_r2_two_back']:.3f}, {row['median_key_r2_two_back']:.3f}" if two else ""))
    json.dump(res, open(OUT, "w"), indent=1)
    log("XLAYER_PROBE_DONE")
