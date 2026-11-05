# Development (exploratory, labelled): where does 3-bit per-channel GPTQ of Mistral-7B-Instruct's attention break the
# token-first model (gptq_bos_dev.py: 6.19 token-less, 569,348 with the token)? Hessians from token-less C4 calibration
# as there; each layer's q, k, v, o are GPTQ-quantized alone and the model is scored on 2 WikiText-2 windows of 2,048
# tokens with the token prepended and as they are; the same with damping 0.1 for the worst layer.
#   python alignment_study/gptq_bos_locate.py [--model=mistralai/Mistral-7B-Instruct-v0.2] [--pplwin=2]
import json
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM
from quant_balance_dev import gptq


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def gptq_g(W, H, bits, group=128, block=128, damp=0.01):
    """GPTQ with a scale per group of input columns, set from the error-updated weights when the group starts (the
    reference implementation's groupsize behaviour, without act-order)."""
    out_dev, out_dtype = W.device, W.dtype
    W = W.detach().cpu().float().clone(); n = W.shape[1]
    H = H.detach().cpu().double().clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += damp * H.diagonal().mean() * torch.eye(n, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Hi1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        E1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:
                g = torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :group]
                lo, hi = g.amin(1), g.amax(1)
                scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8); zero = torch.round(-lo / scale)
            w = W1[:, i]
            qi = (torch.clamp(torch.round(w / scale) + zero, 0, 2 ** bits - 1) - zero) * scale
            Q[:, i1 + i] = qi
            e = (w - qi) / Hi1[i, i]
            W1[:, i:] -= e[:, None] * Hi1[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q.to(out_dtype).to(out_dev)


NAME, N_PPL, BITS = arg("model", "mistralai/Mistral-7B-Instruct-v0.2"), int(arg("pplwin", 2)), 3
OUT = f"alignment_study/gptq_bos_dev/{NAME.split('/')[-1]}_locate.json"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    attn = [b.self_attn for b in model.model.layers]
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    win_bos = torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1)
    ev = lambda: {"bos_prepended": Z.perplexity(model, win_bos, DEV, bs=1), "as_is": Z.perplexity(model, win, DEV, bs=1)}
    H = {}
    def mk(key):
        def pre(m, i):
            x = i[0].detach().reshape(-1, i[0].shape[-1]).float()
            H[key] = H.get(key, 0) + x.T @ x
        return pre
    hooks = [h for l, a in enumerate(attn) for h in (a.q_proj.register_forward_pre_hook(mk((l, "in"))), a.o_proj.register_forward_pre_hook(mk((l, "o"))))]
    cal = C.windows_c4(tok, 32, 1024)
    for w in range(cal.shape[0]):
        model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
    for h in hooks:
        h.remove()
    res = {"model": NAME, "exploratory": True, "bits": BITS, "windows": N_PPL, "unquantized": ev(), "layers": {}}
    log(f"unquantized {res['unquantized']}")

    # whole model at groups of 128 first: if the collapse vanishes there, the per-channel regime is the cause
    saved = [[m.weight.clone() for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj)] for a in attn]
    for l, a in enumerate(attn):
        for k, m in zip("qkvo", (a.q_proj, a.k_proj, a.v_proj, a.o_proj)):
            m.weight.copy_(gptq_g(m.weight, H[(l, "o" if k == "o" else "in")], BITS))
    res["whole_gptq_g128"] = ev(); log(f"whole model, GPTQ groups of 128: {res['whole_gptq_g128']}")
    for a, ws in zip(attn, saved):
        for m, w in zip((a.q_proj, a.k_proj, a.v_proj, a.o_proj), ws):
            m.weight.copy_(w)
    del saved
    json.dump(res, open(OUT, "w"), indent=1)

    def one(l, damp=0.01, which=("q", "k", "v", "o")):
        a = attn[l]; mods = {"q": a.q_proj, "k": a.k_proj, "v": a.v_proj, "o": a.o_proj}
        saved = {k: mods[k].weight.clone() for k in which}
        for k in which:
            mods[k].weight.copy_(gptq(mods[k].weight, H[(l, "o" if k == "o" else "in")], BITS, damp=damp))
        r = ev()
        for k in which:
            mods[k].weight.copy_(saved[k])
        return r

    for l in range(len(attn)):
        res["layers"][l] = one(l); log(f"layer {l}: {res['layers'][l]}")
        json.dump(res, open(OUT, "w"), indent=1)
    worst = max(res["layers"], key=lambda l: res["layers"][l]["bos_prepended"])
    res["worst"] = worst
    res["worst_by_projection"] = {k: one(worst, which=(k,)) for k in ("q", "k", "v", "o")}
    res["worst_damp0.1"] = one(worst, damp=0.1)
    log(f"worst layer {worst}: by projection {res['worst_by_projection']}; damping 0.1 {res['worst_damp0.1']}")
    json.dump(res, open(OUT, "w"), indent=1)
    log("GPTQ_BOS_LOCATE_DONE")
