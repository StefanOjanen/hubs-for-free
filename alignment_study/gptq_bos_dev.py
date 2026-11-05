# Development (exploratory, labelled, not preregistered): does GPTQ's calibration convention matter for the deployed
# model? Standard GPTQ calibration slices carry no beginning-of-sequence token; deployed prompts do. In token-first
# text the sink position carries massive activations that enter every Hessian. Arms: 3-bit per-channel GPTQ of q, k,
# v, o calibrated on 32 C4 windows of 1,024 tokens without the token, and on the same windows with it
# (gqa_common.windows_c4(bos=True)); round-to-nearest as reference. Each evaluated on WikiText-2 windows of 2,048 tokens
# as they are and with the token prepended (bos_quant_check.py's convention).
#   python alignment_study/gptq_bos_dev.py --model=mistralai/Mistral-7B-v0.1 [--bits=3] [--pplwin=8]
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

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME, BITS, N_PPL = arg("model", "mistralai/Mistral-7B-v0.1"), int(arg("bits", 3)), int(arg("pplwin", 8))
OUT = f"alignment_study/gptq_bos_dev/{NAME.split('/')[-1]}_w{BITS}.json"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    big = "7b" in NAME.lower()
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16 if big else torch.float32, attn_implementation="sdpa").eval().to(DEV)
    attn = [b.self_attn for b in model.model.layers]
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    win_bos = torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1)
    ev = lambda: {"as_is": Z.perplexity(model, win, DEV, bs=1), "bos_prepended": Z.perplexity(model, win_bos, DEV, bs=1)}
    W0 = [[m.weight.detach().cpu().clone() for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj)] for a in attn]
    restore = lambda: [m.weight.copy_(w.to(DEV)) for a, ws in zip(attn, W0) for m, w in zip((a.q_proj, a.k_proj, a.v_proj, a.o_proj), ws)]

    def hessians(cal):
        H = {}
        def mk(key):
            def pre(m, i):
                x = i[0].detach().reshape(-1, i[0].shape[-1]).float()
                H[key] = H.get(key, 0) + x.T @ x
            return pre
        hooks = []
        for l, a in enumerate(attn):
            hooks += [a.q_proj.register_forward_pre_hook(mk((l, "in"))), a.o_proj.register_forward_pre_hook(mk((l, "o")))]
        for w in range(cal.shape[0]):
            model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
        return H                                   # kept in place: a host copy doubled the peak and the watchdog killed the 7B run

    res = {"model": NAME, "exploratory": True, "bits": BITS, "windows": N_PPL, "calibration": [32, 1024], "arms": {}}
    res["unquantized"] = ev(); log(f"unquantized {res['unquantized']}")
    for a in attn:
        for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            m.weight.copy_(P.rtn(m.weight, BITS))
    res["arms"]["rtn"] = ev(); log(f"rtn {res['arms']['rtn']}"); restore()
    for a in attn:
        for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            m.weight.copy_(P.rtn(m.weight, BITS, group=128))
    res["arms"]["rtn_g128"] = ev(); log(f"rtn_g128 {res['arms']['rtn_g128']}"); restore()
    json.dump(res, open(OUT, "w"), indent=1)
    for label, bos in (("gptq_tokenless_calib", False), ("gptq_token_first_calib", True)):
        H = hessians(C.windows_c4(tok, 32, 1024, bos=bos))
        for l, a in enumerate(attn):
            for m in (a.q_proj, a.k_proj, a.v_proj):
                m.weight.copy_(gptq(m.weight, H[(l, "in")], BITS))
            a.o_proj.weight.copy_(gptq(a.o_proj.weight, H[(l, "o")], BITS))
        del H
        res["arms"][label] = ev(); log(f"{label} {res['arms'][label]}"); restore()
        json.dump(res, open(OUT, "w"), indent=1)
    log("GPTQ_BOS_DEV_DONE")
