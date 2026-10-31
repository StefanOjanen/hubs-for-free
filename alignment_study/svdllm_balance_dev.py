# Development (exploratory, labelled): does rebalancing help other closed-form compressions built in a whitened
# basis? SVD-LLM (Wang et al. 2025) factors each linear layer W through the Cholesky factor of its input's second
# moment, S = Lc Lc^T: W Lc = U Sigma V^T, truncated to rank k, W ~ A B with A = U_k Sigma_k^1/2 and B = Sigma_k^1/2
# V_k^T Lc^-1, so B carries the inverse whitening. Each rank component j can be rescaled (A's column j times c, B's
# row j divided by c) without changing A B; rebalancing sets every component's two norms equal. This compresses the
# MLP projections of a model to half their parameters, then rounds A and B to 4 and 3 bits per output channel, with
# SVD-LLM's split and rebalanced; WikiText-2 perplexity.
#   python alignment_study/svdllm_balance_dev.py --model=TinyLlama/TinyLlama_v1.1 [--calib=32] [--calib-len=1024] [--pplwin=8] [--keep=0.5]
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "TinyLlama/TinyLlama_v1.1")
N_CAL, T_CAL, N_PPL, KEEP = int(arg("calib", 32)), int(arg("calib-len", 1024)), int(arg("pplwin", 8)), float(arg("keep", 0.5))
OUT = f"alignment_study/balance_dev/svdllm_{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


class Factored(torch.nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.A, self.B = torch.nn.Parameter(A, requires_grad=False), torch.nn.Parameter(B, requires_grad=False)

    def forward(self, x):
        return (x @ self.B.T) @ self.A.T


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    layers = model.model.layers
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)
    res = {"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "keep": KEEP, "ppl_windows": N_PPL, "base": ppl()}
    names = ("gate_proj", "up_proj", "down_proj")
    H = {}
    def mk(key):
        def pre(m, i):
            x = i[0].detach().reshape(-1, i[0].shape[-1]).float()
            g = (x.T @ x).cpu().double()                       # on the GPU in float32, accumulated on the host
            H[key] = H.get(key, 0) + g
        return pre
    hooks = [getattr(blk.mlp, nm).register_forward_pre_hook(mk((l, nm))) for l, blk in enumerate(layers) for nm in names]
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    for w in range(cal.shape[0]):
        model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
    for h in hooks:
        h.remove()
    factors = {}
    for l, blk in enumerate(layers):
        for nm in names:
            W = getattr(blk.mlp, nm).weight.detach().cpu().double()
            out_f, in_f = W.shape
            k = int(KEEP * out_f * in_f / (out_f + in_f))
            Sx = H[(l, nm)]
            Sx = Sx + 1e-6 * Sx.diagonal().mean() * torch.eye(in_f, dtype=torch.float64)
            Lc = torch.linalg.cholesky(Sx)
            U, s, Vt = torch.linalg.svd(W @ Lc, full_matrices=False)
            A = U[:, :k] * s[:k].sqrt()
            Bm = s[:k].sqrt()[:, None] * torch.linalg.solve_triangular(Lc.T, Vt[:k].T, upper=True).T     # Sigma^1/2 V^T Lc^-1
            factors[(l, nm)] = (A, Bm)
    del H
    log(f"factored the MLP projections at rank keeping {KEEP:.2f} of their parameters")

    def install(balance, bits):
        for (l, nm), (A, Bm) in factors.items():
            A2, B2 = A.clone(), Bm.clone()
            if balance:
                c = (B2.norm(dim=1) / A2.norm(dim=0).clamp_min(1e-30)).sqrt()
                A2 = A2 * c[None, :]; B2 = B2 / c[:, None]
            A2, B2 = A2.float().to(DEV), B2.float().to(DEV)
            if bits:
                A2, B2 = P.rtn(A2, bits), P.rtn(B2, bits)
            setattr(layers[l].mlp, nm, Factored(A2, B2))
    stats = []
    for (l, nm), (A, Bm) in list(factors.items())[:3]:
        stats.append({"layer": l, "proj": nm, "A_col_norm_median": float(A.norm(dim=0).median()), "B_row_norm_median": float(Bm.norm(dim=1).median()),
                      "B_row_norm_max": float(Bm.norm(dim=1).max()), "A_col_norm_max": float(A.norm(dim=0).max())})
    res["norm_examples"] = stats
    for balance in (False, True):
        key = "balanced" if balance else "svdllm"
        r = {}
        for bits in (None, 4, 3):
            install(balance, bits)
            r[f"w{bits}" if bits else "fp32"] = ppl()
        res[key] = r
        log(f"{key}: float32 {r['fp32']:.3f}, 4 bits {r['w4']:.3f}, 3 bits {r['w3']:.3f}")
        json.dump(res, open(OUT, "w"), indent=1)
    log(f"unconverted {res['base']:.3f}; norms {stats}")
    log("SVDLLM_BALANCE_DEV_DONE")
