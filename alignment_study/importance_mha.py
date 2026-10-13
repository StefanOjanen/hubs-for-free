# PREREGISTRATION15: preregistration 12's head-importance clauses (S2 to S4) on
# multi-head-attention decoders (GPT-2, Pythia) and on a second corpus (C4).
# Architecture-generic: attention weights from output_attentions (eager); each
# head's output read from the input of the layer's output projection
# (GPT-2 attn.c_proj, a Conv1D; Pythia attention.dense, a Linear), which is the
# head-major concatenation of the heads' outputs a_r V_h; a head is removed by
# zeroing its slice of the output projection. Head statistics on the selection
# windows (WikiText-103 validation, 16 x 256): confidence, negative entropy,
# sink mass and output norm, as in standard_common.py. Outcome: KL of
# removing each head (round 11's measure) on two evaluation sets, WikiText-103
# test and C4 validation, 8 windows of 512 tokens each.
#   python alignment_study/importance_mha.py [--dev] [--only=name]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import functional_common as F
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.stats import sink_columns
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV_RUN = "--dev" in sys.argv
MODELS = ["gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
DEV_MODELS = ["gpt2"]
OUT = "alignment_study/importance_mha_dev" if DEV_RUN else "alignment_study/importance_mha"
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
N_EVAL, T_EVAL = 8, 512
ROW0 = F.ROW0
torch.set_grad_enabled(False)


def oproj(model, l):
    """(module, kind): GPT-2 Conv1D (weight in x out) or a Linear (weight out x in)."""
    if hasattr(model, "transformer"):
        return model.transformer.h[l].attn.c_proj, "conv1d"
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers[l].attention.dense, "linear"
    return model.model.layers[l].self_attn.o_proj, "linear"


def head_slice_weight(mod, kind, h, hd):
    W = mod.weight.detach().float().cpu().numpy().astype(np.float64)
    return W[h * hd:(h + 1) * hd, :] if kind == "conv1d" else W[:, h * hd:(h + 1) * hd].T     # (hd, D) in both cases


def geometry(model):
    c = model.config
    L = getattr(c, "num_hidden_layers", None) or c.n_layer
    n = getattr(c, "num_attention_heads", None) or c.n_head
    D = getattr(c, "hidden_size", None) or c.n_embd
    return L, n, D // n


def c4_eval_windows(tok, n, T):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    out = []
    for ex in ds:
        ids = tok(ex["text"], return_tensors="pt", truncation=True, max_length=T)["input_ids"]
        if ids.shape[1] >= T:
            out.append(ids[0])
        if len(out) == n:
            break
    return torch.stack(out)


def selection_stats(model, sel, L, n, hd):
    grams = {}
    for l in range(L):
        mod, kind = oproj(model, l)
        grams[l] = [(lambda Wh: Wh @ Wh.T)(head_slice_weight(mod, kind, h, hd)) for h in range(n)]
    store, handles = {}, []
    for l in range(L):
        mod, _ = oproj(model, l)
        def pre(m, args, l=l):
            store[l] = args[0][0].detach().float().cpu().numpy().astype(np.float64)
        handles.append(mod.register_forward_pre_hook(pre))
    acc = {l: [{k: [] for k in ("conf", "neg_entropy", "sink_mass", "out_norm")} for _ in range(n)] for l in range(L)}
    for x in sel:
        out = model(input_ids=x.to(DEV), output_attentions=True)
        for l in range(L):
            A = out.attentions[l][0].float().cpu().numpy().astype(np.float64)
            S = list(sink_columns(A)); Xin = store[l]
            for h in range(n):
                a = A[h][ROW0:]
                ent = -(a * np.log(np.clip(a, 1e-30, 1.0))).sum(1).mean()
                xh = Xin[ROW0:, h * hd:(h + 1) * hd]
                on = np.sqrt(np.clip(np.einsum("rd,de,re->r", xh, grams[l][h], xh), 0, None)).mean()
                d = acc[l][h]
                d["conf"].append(a.max(1).mean()); d["neg_entropy"].append(-ent); d["sink_mass"].append(a[:, S].sum(1).mean()); d["out_norm"].append(on)
        del out
    for hnd in handles:
        hnd.remove()
    return {l: [{k: float(np.mean(v)) for k, v in d.items()} for d in acc[l]] for l in range(L)}


def ablation_sweep(model, sets, L, n, hd, log):
    res = {name: {} for name in sets}
    for l in range(L):
        mod, kind = oproj(model, l)
        for name in sets:
            res[name][str(l)] = {}
        for h in range(n):
            sl = slice(h * hd, (h + 1) * hd)
            if kind == "conv1d":
                bk = mod.weight[sl, :].clone(); mod.weight[sl, :] = 0
            else:
                bk = mod.weight[:, sl].clone(); mod.weight[:, sl] = 0
            for name, (X, base) in sets.items():
                kls, dn = F.kl_eval(model, X, base)
                res[name][str(l)][str(h)] = {"kl": float(np.mean(kls)), "dnll": float(np.mean(dn))}
            if kind == "conv1d":
                mod.weight[sl, :] = bk
            else:
                mod.weight[:, sl] = bk
        log(f"  layer {l}: {n} heads")
    return res


def run_model(name):
    t0 = time.time()
    log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L, n, hd = geometry(model)
    log(f"{name} DTYPE {dt} layers {L} heads {n} head_dim {hd}")
    M.N_SEL, M.T_SEL, M.N_EVAL, M.T_EVAL = F.N_SEL, F.T_SEL, N_EVAL, T_EVAL
    sel = M.selection_windows(tok)
    X_wiki = M.eval_windows(tok); X_c4 = c4_eval_windows(tok, N_EVAL, T_EVAL)
    stats = selection_stats(model, sel, L, n, hd)
    log("selection statistics done")
    sets = {"wikitext": (X_wiki, F.base_cache(model, X_wiki)), "c4": (X_c4, F.base_cache(model, X_c4))}
    abl = ablation_sweep(model, sets, L, n, hd, log)
    checks = {k: float(np.mean(F.kl_eval(model, X, b)[0])) for k, (X, b) in sets.items()}
    del model; release_memory(DEV)
    return {"model": name, "dtype": dt, "device": DEV, "layers": L, "heads": n, "head_dim": hd, "eval_tokens_each": int(X_wiki.numel()),
            "restore_check_kl": checks, "stats": {str(l): stats[l] for l in range(L)}, "ablation": abl, "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name in (DEV_MODELS if DEV_RUN else MODELS):
        if ONLY and name not in ONLY:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        try:
            res = run_model(name)
            json.dump(res, open(path, "w"), indent=1)
            print(f"MODEL_DONE {name} {res['runtime_s']}s", flush=True)
        except Exception as e:
            import traceback
            print(f"MODEL_FAILED {name}: {type(e).__name__}: {str(e)[:300]}", flush=True); print(traceback.format_exc()[-1500:], flush=True)
    print("IMPORTANCE_MHA_DONE", flush=True)
