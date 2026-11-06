# Development (exploratory, labelled): does the start-token GPTQ collapse (gptq_bos_fix.py: Mistral-7B-v0.1 per-channel
# 3-bit GPTQ of q, k, v, o gives 5.56 as is and 25,321 with the beginning-of-sequence token) survive the reference
# protocol? Port of IST-DASLab/gptq llama.py (main branch, read 2026-11-06): every linear layer of every block, attention
# and MLP; layers quantized in order, each block's Hessians collected from the outputs of the already-quantized blocks
# below; 128 C4 segments of 2,048 tokens; asymmetric per-row min-max range that includes zero (quant.Quantizer);
# percdamp 0.01, block 128. Arms: per-channel; per-channel with act-order; per-channel with the sink-aware term
# H + lam * N * x_s x_s^T (x_s each linear's input at position 0 of a lone start token, propagated through the
# quantized blocks like the calibration data); groups of 128 with act-order (GPTQModel's usual layout). Perplexity on
# 8 WikiText-2 windows of 2,048 tokens as they are and with the token.
#   python alignment_study/gptq_ref_check.py [--model=mistralai/Mistral-7B-v0.1] [--nsamples=128] [--arms=a,b]
import json
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME, NS, N_PPL, BITS = arg("model", "mistralai/Mistral-7B-v0.1"), int(arg("nsamples", 128)), int(arg("pplwin", 8)), 3
ARMS = arg("arms", "channel,channel_actorder,channel_sink0.01,g128_actorder").split(",")
LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
OUT = f"alignment_study/gptq_bos_dev/{NAME.split('/')[-1]}_ref.json"
torch.set_grad_enabled(False)


def params(W, bits):
    """quant.Quantizer.find_params, perchannel, sym=False: per-row range extended to include zero."""
    lo = torch.minimum(W.amin(1), torch.zeros(1)); hi = torch.maximum(W.amax(1), torch.zeros(1))
    both = (lo == 0) & (hi == 0); lo[both] = -1; hi[both] = 1
    scale = (hi - lo) / (2 ** bits - 1)
    return scale, torch.round(-lo / scale)


def fasterquant(W, H, bits, group=-1, actorder=False, block=128, percdamp=0.01):
    """gptq.GPTQ.fasterquant (static_groups False); the column loop on the host."""
    dev, dt = W.device, W.dtype
    W = W.detach().cpu().float().clone(); H = H.detach().cpu().double().clone(); n = W.shape[1]
    maxq = 2 ** bits - 1
    scale, zero = params(W, bits)
    dead = H.diagonal() == 0
    H[dead, dead] = 1; W[:, dead] = 0
    if actorder:
        perm = torch.argsort(H.diagonal(), descending=True); W, H = W[:, perm], H[perm][:, perm]
    H += percdamp * H.diagonal().mean() * torch.eye(n, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Hi1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        E1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            if group != -1 and (i1 + i) % group == 0:
                scale, zero = params(W[:, i1 + i:i1 + i + group], bits)   # as the reference: W, updated through the previous block
            w = W1[:, i]
            q = scale * (torch.clamp(torch.round(w / scale) + zero, 0, maxq) - zero)
            Q[:, i1 + i] = q
            e = (w - q) / Hi1[i, i]
            W1[:, i:] -= e[:, None] * Hi1[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if actorder:
        Q = Q[:, torch.argsort(perm)]
    return Q.to(dt).to(dev)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    load = lambda: AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    model = load(); blocks = model.model.layers
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    win_bos = torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1)
    ev = lambda: {"as_is": Z.perplexity(model, win, DEV, bs=1), "bos_prepended": Z.perplexity(model, win_bos, DEV, bs=1)}
    cal = C.windows_c4(tok, NS, 2048)

    def first_inputs(ids):
        """Block 0's hidden states and keyword arguments for each sequence (llama.py's Catcher)."""
        got = []
        def catch(m, a, kw):
            got.append((a[0] if a else kw.pop("hidden_states"), kw)); raise ValueError
        h = blocks[0].register_forward_pre_hook(catch, with_kwargs=True)
        for s in range(ids.shape[0]):
            try:
                model(input_ids=ids[s:s + 1].to(DEV), use_cache=False)
            except ValueError:
                pass
        h.remove()
        return got

    def run(label, group, actorder, lam):
        inps, sink = first_inputs(cal), first_inputs(torch.tensor([[tok.bos_token_id]]))
        ntok = cal.numel()
        for l, b in enumerate(blocks):
            H, X = {}, {}
            def mk(n, store):
                def pre(m, i):
                    x = i[0].detach().reshape(-1, i[0].shape[-1]).float()
                    if store is H:
                        H[n] = H.get(n, 0) + x.T @ x
                    else:
                        X[n] = x[0]
                return pre
            tb = time.time()
            for store, states in ((H, inps), (X, sink)):
                hs = [b.get_submodule(n).register_forward_pre_hook(mk(n, store)) for n in LINEAR]
                for x, kw in states:
                    b(x, **kw)
                for h in hs:
                    h.remove()
            th = time.time()
            for n in LINEAR:
                m = b.get_submodule(n)
                Hn = H[n] + lam * ntok * torch.outer(X[n], X[n]) if lam else H[n]
                if lam and l == 0 and n == LINEAR[0]: log(f"sink term: |x_s|^2 {float(X[n].norm()) ** 2:.3g}, rank-one {lam * ntok * float(X[n].norm()) ** 2:.3g} against tr H {float(H[n].trace()):.3g}")
                tq = time.time(); m.weight.copy_(fasterquant(m.weight, Hn, BITS, group, actorder))
                if l == 0:
                    log(f"{label}: block 0 {n} quantized in {time.time() - tq:.0f}s (Hessians {th - tb:.0f}s)")
            del H, X
            first = lambda o: o[0] if isinstance(o, tuple) else o
            out = lambda st: [(first(b(x, **kw)), kw) for x, kw in st]
            inps, sink = out(inps), out(sink)
            if l % 8 == 7:
                log(f"{label}: block {l} quantized")
        r = ev(); log(f"{label} {r}")
        return r

    res = {"model": NAME, "exploratory": True, "protocol": "IST-DASLab/gptq llama.py port, all linear layers, sequential",
           "bits": BITS, "nsamples": NS, "windows": N_PPL, "arms": {}}
    res["unquantized"] = ev(); log(f"unquantized {res['unquantized']}")
    spec = {"channel": (-1, False, 0.0), "channel_actorder": (-1, True, 0.0), "channel_sink0.01": (-1, False, 0.01),
            "g128_actorder": (128, True, 0.0)}
    for a in ARMS:
        res["arms"][a] = run(a, *spec[a])
        # reload instead of keeping a copy of the weights: a second copy of a 7B model does not fit beside the first
        del blocks, model; torch.mps.empty_cache() if DEV == "mps" else None
        model = load(); blocks = model.model.layers
        json.dump(res, open(OUT, "w"), indent=1)
    log("GPTQ_REF_CHECK_DONE")
