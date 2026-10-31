# Development (exploratory, labelled): which metric each part of the fusion should be fitted in. Round 31 fitted
# keys, values and query energies in S_att, the second moment with each token weighted by the attention it receives,
# and failed (lower in 3 of 5 models). With the attention pattern held fixed, two metrics follow from the attention
# computation itself instead:
#   values: head h's output at query t is O_h V x_bar_t with x_bar_t = sum_s p_ts x_s, so the value error is exactly
#     tr(dV S_bar dV^T) with S_bar = E_t[x_bar_t x_bar_t^T], the second moment of the attention-mixed inputs
#     (averaged over the layer's heads); S_att is its diagonal upper bound (Jensen), S assumes every query sees one token.
#   keys: to first order a logit error at (t, s) moves the output by p_ts (z_s - z_bar_t), z = the output-weighted value
#     (|z|^2 = v^T O_h^T O_h v); a key error at token s is weighted by w_s = sum_t p_ts |z_s - z_bar_t|^2 and a query
#     position t by u_t = sum_s p_ts |z_s - z_bar_t|^2 (the diagonal of the first-order error, as S_att is for values).
# Arms (key metric / query-energy metric / value metric), pairs fixed at the fusion's matching under S:
#   plain S/S/S; round31 Sa/Sa/Sa; mixv Sa/Sa/Sbar; plainmixv S/S/Sbar; sens Sw/Su/Sbar; sensk Sw/Su/S.
#   python alignment_study/attnw_mix_dev.py --model=JackFram/llama-160m [--calib=32] [--calib-len=1024] [--pplwin=20]
#       [--qstride=8] [--arms=plain,round31,mixv,plainmixv,sens,sensk]
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
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "JackFram/llama-160m")
N_CAL, T_CAL, N_PPL, QSTRIDE = int(arg("calib", 32)), int(arg("calib-len", 1024)), int(arg("pplwin", 20)), int(arg("qstride", 8))
ARMS = {"plain": ("S", "S", "S"), "round31": ("Sa", "Sa", "Sa"), "mixv": ("Sa", "Sa", "Sbar"), "plainmixv": ("S", "S", "Sbar"),
        "sens": ("Sw", "Su", "Sbar"), "sensk": ("Sw", "Su", "S")}
ARMS = {a: ARMS[a] for a in arg("arms", ",".join(ARMS)).split(",")}
OUT = f"alignment_study/attnw_mix_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


def moments(model, kv, cal):
    """S, Sa (received attention), Sbar (attention-mixed inputs), Sw (key sensitivity), Su (query sensitivity),
    float32 on the host; queries every QSTRIDE-th position."""
    d, L, hd, n = kv.d, kv.L, kv.hd, kv.n
    M = {k: {l: torch.zeros(d, d) for l in range(L)} for k in ("S", "Sa", "Sbar", "Sw", "Su")}
    lin = torch.nn.functional.linear
    R = {}                                                     # R_h R_h^T = O_h^T O_h, so |v R_h|^2 is the output energy
    for l in range(L):
        Ow = kv.attn(l).o_proj.weight
        R[l] = [torch.linalg.cholesky((lambda Oh: Oh.T @ Oh + 1e-8 * torch.eye(hd, dtype=torch.float64))(
            Ow[:, h * hd:(h + 1) * hd].cpu().double())).float().to(DEV) for h in range(n)]
    for w in range(cal.shape[0]):
        T = cal.shape[1]
        pos = torch.arange(T, device=DEV)[None]
        rows = torch.arange(QSTRIDE - 1, T, QSTRIDE, device=DEV)
        mask = torch.ones(T, T, dtype=torch.bool, device=DEV).tril()[rows]

        def mk(l):
            def pre(mod, inp):
                x = inp[0][0].float()
                a = kv.attn(l)
                cos, sin = model.model.rotary_emb(x[None], position_ids=pos)
                q = lin(x, a.q_proj.weight.float()).view(1, T, n, hd).transpose(1, 2)
                k = lin(x, a.k_proj.weight.float()).view(1, T, kv.nkv, hd).transpose(1, 2)
                v = lin(x, a.v_proj.weight.float()).view(T, kv.nkv, hd).transpose(0, 1)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                recv, ws, us = torch.zeros(T, device=x.device), torch.zeros(T, device=x.device), torch.zeros(len(rows), device=x.device)
                A = torch.zeros(T, T, device=x.device)
                for h in range(n):
                    p = torch.softmax(((q[0, h][rows] @ k[0, h // kv.rep].T) / hd ** 0.5).masked_fill(~mask, float("-inf")), -1)
                    ps = p.sum(0)
                    recv += ps
                    A += p.T @ p
                    z = v[h // kv.rep] @ R[l][h]
                    zz, zb = (z * z).sum(1), p @ z
                    zbn = (zb * zb).sum(1)
                    ws += ps * zz - 2 * (z * (p.T @ zb)).sum(1) + p.T @ zbn
                    us += p @ zz - zbn
                M["S"][l] += (x.T @ x).cpu()
                M["Sa"][l] += ((x * (recv / recv.sum() * T)[:, None]).T @ x).cpu()
                M["Sbar"][l] += (x.T @ (A @ x) / (n * len(rows))).cpu()
                M["Sw"][l] += ((x * (ws / ws.sum() * T)[:, None]).T @ x).cpu()
                xr = x[rows]
                M["Su"][l] += ((xr * (us / us.sum() * len(rows))[:, None]).T @ xr * (T / len(rows))).cpu()
            return pre
        hooks = [kv.attn(l).k_proj.register_forward_pre_hook(mk(l)) for l in range(L)]
        model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
    ntok = cal.shape[0] * cal.shape[1]
    return {k: {l: v / ntok for l, v in m.items()} for k, m in M.items()}


def snap(t, idx):
    return [t[i].clone() for i in idx]


def fuse_split(kv, groups, Sk, Sq, Sv):
    """Every group fused with the keys fitted in Sk, the query energies in Sq and the values in Sv: the frozen
    fuse_kv.fuse_kv_group run once with (Sk, Sq) for the keys and once with Sv for the values, each pass's other
    half restored."""
    hd = kv.hd
    for l in range(kv.L):
        a = kv.attn(l)
        k64, v64 = Sk[l].double(), Sv[l].double()
        rk, rv = F.layer_roots(k64, kv.d), F.layer_roots(v64, kv.d)
        for G in groups[l]:
            G = list(G)
            Qh = [h for i in G for h in range(i * kv.rep, (i + 1) * kv.rep)]
            krows = [slice(i * hd, (i + 1) * hd) for i in G]; qrows = [slice(h * hd, (h + 1) * hd) for h in Qh]
            v0, o0 = snap(a.v_proj.weight, krows), [a.o_proj.weight[:, s].clone() for s in qrows]
            K.fuse_kv_group(kv, l, G, k64, rk, qw=True, ow=True, Lsq=Sq[l].double())
            k1, q1 = snap(a.k_proj.weight, krows), snap(a.q_proj.weight, qrows)
            if Sv is Sk:
                continue
            for s, t in zip(krows, v0):
                a.v_proj.weight[s] = t
            for s, t in zip(qrows, o0):
                a.o_proj.weight[:, s] = t
            K.fuse_kv_group(kv, l, G, v64, rv, qw=True, ow=True, Lsq=v64)
            for s, t in zip(krows, k1):
                a.k_proj.weight[s] = t
            for s, t in zip(qrows, q1):
                a.q_proj.weight[s] = t


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    load = lambda: AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    model = load(); kv = K.KV(model)
    assert not kv.has_bias(0) and kv.r == kv.hd, "development models only: no biases, full rotary"
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    M = moments(model, kv, C.windows_c4(tok, N_CAL, T_CAL))
    log(f"{NAME}: moments")
    groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, M["S"][l].double())) for l in range(kv.L)]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.update({"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL, "query_stride": QSTRIDE,
                "base": Z.perplexity(model, win, DEV, bs=2)})
    res.setdefault("arms", {})
    for arm, (kk, qq, vv) in ARMS.items():
        fuse_split(kv, groups, M[kk], M[qq], M[vv])
        res["arms"][arm] = {"metrics": [kk, qq, vv], "ppl": Z.perplexity(model, win, DEV, bs=2)}
        log(f"{arm} ({kk}/{qq}/{vv}): ppl {res['arms'][arm]['ppl']:.3f}")
        json.dump(res, open(OUT, "w"), indent=1)
        del model, kv; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        model = load(); kv = K.KV(model)
    log(f"unconverted {res['base']:.3f}")
    log("ATTNW_MIX_DEV_DONE")
