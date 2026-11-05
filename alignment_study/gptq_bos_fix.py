# Development (exploratory, labelled): a sink-aware GPTQ Hessian. GPTQ of Mistral-7B-Instruct's layer-0 v_proj alone
# collapses the token-first model (gptq_bos_locate.py: 297,888) because, on the hypothesis, the beginning-of-sequence
# token's input is a direction the token-less calibration Hessian barely weights. Fix: add that token's input at each
# projection, x_s, to the Hessian with weight lam * N (N calibration tokens): H + lam * N * x_s x_s^T. x_s is position 0's
# input under a lone beginning-of-sequence token (causal attention makes it the same in every token-first sequence).
# Arms (with, at lam 0, IntactKV's first-token keys and values and a layer-0-kept variant): per-channel and groups-of-128 GPTQ of q, k, v, o in every layer, lam in {0, 0.001, 0.01}; RTN groups of 128 as
# reference; 8 WikiText-2 windows of 2,048 tokens as they are and with the token.
#   python alignment_study/gptq_bos_fix.py [--model=mistralai/Mistral-7B-Instruct-v0.2]
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
from gptq_bos_locate import gptq_g

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME, N_PPL, BITS = arg("model", "mistralai/Mistral-7B-Instruct-v0.2"), int(arg("pplwin", 8)), 3
OUT = f"alignment_study/gptq_bos_dev/{NAME.split('/')[-1]}_fix.json"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    attn = [b.self_attn for b in model.model.layers]
    mods = lambda a: (a.q_proj, a.k_proj, a.v_proj, a.o_proj)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    win_bos = torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1)
    ev = lambda: {"as_is": Z.perplexity(model, win, DEV, bs=1), "bos_prepended": Z.perplexity(model, win_bos, DEV, bs=1)}
    W0 = [[m.weight.detach().clone() for m in mods(a)] for a in attn]
    restore = lambda: [m.weight.copy_(w) for a, ws in zip(attn, W0) for m, w in zip(mods(a), ws)]

    H, X, ntok = {}, {}, 0
    def mk(key, store):
        def pre(m, i):
            x = i[0].detach().reshape(-1, i[0].shape[-1]).float()
            if store is H:
                H[key] = H.get(key, 0) + x.T @ x
            else:
                X[key] = x[0]
        return pre
    for store, inputs in ((H, C.windows_c4(tok, 32, 1024)), (X, torch.tensor([[tok.bos_token_id]]))):
        hooks = [h for l, a in enumerate(attn) for h in (a.q_proj.register_forward_pre_hook(mk((l, "in"), store)), a.o_proj.register_forward_pre_hook(mk((l, "o"), store)))]
        for w in range(inputs.shape[0]):
            model(input_ids=inputs[w:w + 1].to(DEV), use_cache=False)
            if store is H:
                ntok += inputs.shape[1]
        for h in hooks:
            h.remove()
    res = {"model": NAME, "exploratory": True, "bits": BITS, "windows": N_PPL, "calibration_tokens": ntok, "arms": {}}
    res["unquantized"] = ev(); log(f"unquantized {res['unquantized']}")
    for a in attn:
        for m in mods(a):
            m.weight.copy_(P.rtn(m.weight, BITS, group=128))
    res["arms"]["rtn_g128"] = ev(); log(f"rtn_g128 {res['arms']['rtn_g128']}"); restore()
    for lam in (0.0, 0.001, 0.01):
        for layout, fn in (("channel", gptq), ("g128", gptq_g)):
            for l, a in enumerate(attn):
                for k, m in zip("qkvo", mods(a)):
                    key = (l, "o" if k == "o" else "in")
                    Hk = H[key] + lam * ntok * torch.outer(X[key], X[key])
                    m.weight.copy_(fn(m.weight, Hk, BITS))
            label = f"gptq_{layout}_lam{lam}"
            res["arms"][label] = ev(); log(f"{label} {res['arms'][label]}")
            if lam == 0.0:
                # IntactKV's remedy (Liu et al. 2024): position 0's keys and values computed with the unquantized weights
                ih = []
                for a, ws in zip(attn, W0):
                    for m, w0 in ((a.k_proj, ws[1]), (a.v_proj, ws[2])):
                        def f(mod, i, o, w0=w0):
                            o = o.clone(); o[:, 0] = i[0][:, 0] @ w0.T
                            return o
                        ih.append(m.register_forward_hook(f))
                res["arms"][label + "_intactkv"] = ev(); log(f"{label}_intactkv {res['arms'][label + '_intactkv']}")
                for h in ih:
                    h.remove()
                # the same quantization with layer 0 left unquantized: how much of the gap sits in layer 0
                for m, w in zip(mods(attn[0]), W0[0]):
                    m.weight.copy_(w)
                res["arms"][label + "_layer0_kept"] = ev(); log(f"{label}_layer0_kept {res['arms'][label + '_layer0_kept']}")
            restore()
            json.dump(res, open(OUT, "w"), indent=1)
    log("GPTQ_BOS_FIX_DONE")
