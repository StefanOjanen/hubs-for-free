# RERUN ROUND (PREREGISTRATION6.md): the thirteen-model set with the fixed
# instruments. Protocol A (T = 64, 48 windows from distinct documents): the
# main statistics, with the multi-column sink set (modal column plus columns
# whose layer-level mass reaches 0.10, at most 3) and column-set surrogates.
# Protocol B (T = 256, 6 windows): the dissociation controls (wrapped targets
# primary; shift and altsink-v2 secondary), whose r1 restoration needs the
# causal width (control_redesign_toy*.py). The
# statistics come from the registered implementations (common.py,
# scale_round_v2.py, multicolumn_dev_calibration.py). One JSON per model is
# written as it completes; a rerun skips finished models.
#   python alignment_study/rerun_round.py            # all models
#   flags: --exclude=<id>,<id>  (run later under the same file)  --purge-large  (drop HF cache of >= 3B models after use)
#   python alignment_study/rerun_round.py --smoke    # dev model, 2 windows, 3 layers
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (generators, gnorms, r1, rho, shared_mode, sink_column, col_mass,
                    sink_generator, zstats, zmed_of)
from scale_round_v2 import coup_batch, sur_plain_batch, sur_alt2_batch, DEV
from multicolumn_dev_calibration import column_masses, sur_colfix_batch, multi_sink_generator
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = ["gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "TinyLlama/TinyLlama_v1.1", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
          "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", "microsoft/Phi-3-mini-4k-instruct",
          "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B", "allenai/OLMo-2-1124-7B"]
A_NWIN, A_SEQ, B_NWIN, B_SEQ = 48, 64, 6, 256
KPLAIN, KFAM, THRESH, KMAX = 12, 8, 0.10, 3
OUT = "alignment_study/rerun"
SMOKE = "--smoke" in sys.argv
PURGE_LARGE = "--purge-large" in sys.argv      # remove the HF cache of models >= 3B params after their run (disk)
EXCLUDE = set()
ONLY = None
for _a in sys.argv:
    if _a.startswith("--exclude="):
        EXCLUDE |= set(x for x in _a.split("=", 1)[1].split(",") if x)
    if _a.startswith("--only="):          # run these model ids instead of MODELS (preregistration 10 uses this)
        ONLY = [x for x in _a.split("=", 1)[1].split(",") if x]
    if _a.startswith("--out="):           # results directory (default alignment_study/rerun)
        OUT = _a.split("=", 1)[1]
torch.set_grad_enabled(False)


def windows(tok, n, seq):
    """The first n texts, at an even stride, among WikiText-103 validation
    texts of over 40 words that tokenize to at least seq tokens; each
    truncated to seq tokens, so every window has exactly T = seq."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    pool = [t for t in ds["text"] if len(t.split()) > 40]
    ok = [t for t in pool if len(tok(t)["input_ids"]) >= seq]
    stride = max(1, len(ok) // n)
    return [tok(ok[i * stride], return_tensors="pt", truncation=True, max_length=seq) for i in range(n)]


def sink_set(A):
    modal = sink_column(A)
    cm = column_masses(A)
    order = [int(c) for c in np.argsort(cm)[::-1]]
    extra = [c for c in order if c != modal and cm[c] >= THRESH][:KMAX - 1]
    return sorted([modal] + extra), modal, cm


def sur_shift_batch(A, rg, K):
    """Matched-geometry control: head-specific cyclic shift of each causal row
    by an offset drawn per draw as a permutation of 1..n (hubsfree.nulls.surrogate_shift)."""
    n, T, _ = A.shape
    S = np.repeat(A[None], K, 0)
    for d in range(K):
        offs = rg.permutation(n) + 1
        for h in range(n):
            for i in range(2, T):
                S[d, h, i, :i] = np.roll(A[h, i, :i], int(offs[h]) % i)
    return S


def sur_wrapped_batch(A, rg, K):
    """Wrapped-target control (altsink-v3): per draw a random permutation of
    1..n assigns each head a target column; in every row i >= 2 the row
    maximum is swapped into the target (wrapped to 1 + (t - 1) mod (i - 1)
    when t >= i) and the other causal entries are permuted, so every row of
    every head receives the same treatment (control_redesign_toy.ctrl_wrapped)."""
    n, T, _ = A.shape
    S = np.repeat(A[None], K, 0)
    rows = np.arange(K)
    for d in range(K):
        targets = rg.permutation(n) + 1
        for h in range(n):
            t = int(targets[h])
            for i in range(2, T):
                tt = t if t < i else 1 + (t - 1) % (i - 1)
                row = S[d, h, i, :i]
                j = int(row.argmax()); row[j], row[tt] = row[tt], row[j]
                rest = np.array([c for c in range(i) if c != tt])
                row[rest] = row[rest][rg.permutation(len(rest))]
    return S


def unit(M):
    return M / np.sqrt((M ** 2).sum())


def family(S_, iu, mu, sd):
    G = np.stack([generators(x) for x in S_]); gn = [gnorms(g) for g in G]; C = coup_batch(G)
    K = len(S_)
    return (float(np.median([zmed_of(C[k], mu, sd, iu) for k in range(K)])),
            float(np.median([r1(C[k], gn[k], iu) for k in range(K)])),
            float(np.median([rho(G[k], gn[k]) for k in range(K)])))


def base(A):
    n, T, _ = A.shape
    iu = np.triu_indices(n, 1)
    G = generators(A); gn = gnorms(G); C = coup_batch(G[None])[0]
    cols, s, cm = sink_set(A)
    Smat, a, e, frac = shared_mode(G); Sn = unit(Smat)
    d = {"s": s, "cols": cols, "k": len(cols), "smass": col_mass(A, s), "set_mass": float(sum(cm[c] for c in cols)),
         "sharedE": frac, "cv_gn": float(gn.std() / gn.mean()), "r1_real": r1(C, gn, iu), "rho_real": rho(G, gn),
         "cosS": float(abs((Sn * multi_sink_generator(T, cols)).sum())),
         "cosS_single": float(abs((Sn * sink_generator(T, s)).sum()))}
    return d, G, gn, C, iu, cols


def stats_A(A, rg):
    d, G, gn, C, iu, cols = base(A)
    Sp = sur_plain_batch(A, rg, KPLAIN)
    Gp = np.stack([generators(x) for x in Sp]); gnp = [gnorms(g) for g in Gp]; Cp = coup_batch(Gp)
    z, mu, sd = zstats(C, list(Cp), iu)
    d.update(zmed_real=z, zmed_plain=float(np.median([zmed_of(Cp[k], mu, sd, iu) for k in range(KPLAIN)])),
             r1_plain=float(np.median([r1(Cp[k], gnp[k], iu) for k in range(KPLAIN)])),
             rho_plain=float(np.median([rho(Gp[k], gnp[k]) for k in range(KPLAIN)])))
    d["zmed_colfix"], d["r1_colfix"], d["rho_colfix"] = family(sur_colfix_batch(A, rg, KFAM, cols), iu, mu, sd)
    d["zmed_alt2"], d["r1_alt2"], d["rho_alt2"] = family(sur_alt2_batch(A, rg, KFAM), iu, mu, sd)
    return d


def stats_B(A, rg):
    d, G, gn, C, iu, cols = base(A)
    Sp = sur_plain_batch(A, rg, KFAM)
    Cp = coup_batch(np.stack([generators(x) for x in Sp]))
    z, mu, sd = zstats(C, list(Cp), iu)
    d["zmed_real"] = z
    d["zmed_wrapped"], d["r1_wrapped"], d["rho_wrapped"] = family(sur_wrapped_batch(A, rg, KFAM), iu, mu, sd)
    d["zmed_shift"], d["r1_shift"], d["rho_shift"] = family(sur_shift_batch(A, rg, KFAM), iu, mu, sd)
    d["zmed_alt2"], d["r1_alt2"], d["rho_alt2"] = family(sur_alt2_batch(A, rg, KFAM), iu, mu, sd)
    return d


MEAN_KEYS = ("smass", "set_mass", "sharedE")
KEEP_WINDOWS = ("smass", "sharedE", "r1_real", "zmed_real", "rho_real", "cv_gn", "cosS")


def aggregate(acc, L):
    rows = []
    for l in range(L):
        ds = acc[l]; agg = {"layer": l}
        for k in ds[0]:
            vals = [d[k] for d in ds]
            if k == "cols":
                strs = [str(v) for v in vals]; agg["cols_modal"] = max(set(strs), key=strs.count); agg["cols_modal_frac"] = round(strs.count(agg["cols_modal"]) / len(strs), 3)
            elif k == "s":
                agg["s_modal"] = int(max(set(vals), key=vals.count)); agg["s_nunique"] = len(set(vals))
            elif k == "k":
                agg["k_median"] = float(np.median(vals)); agg["k_max"] = int(max(vals))
            else:
                agg[k] = round(float(np.mean(vals)) if k in MEAN_KEYS else float(np.median(vals)), 4)
        agg["windows"] = {k: [round(float(d[k]), 4) for d in ds] for k in KEEP_WINDOWS if k in ds[0]}
        rows.append(agg)
    return rows


def run_protocol(model, tok, nwin, seq, fn, layers=None):
    wins = windows(tok, nwin, seq)
    rg = np.random.default_rng(0)
    L = model.config.num_hidden_layers
    use = list(range(L)) if layers is None else layers
    acc = {l: [] for l in use}
    for w, ids in enumerate(wins):
        out = model(**{k: v.to(DEV) for k, v in ids.items()})
        for l in use:
            acc[l].append(fn(out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64), rg))
        del out
        print(f"    window {w + 1}/{nwin} T={seq}", flush=True)
    rows = aggregate({i: acc[l] for i, l in enumerate(use)}, len(use))
    for r, l in zip(rows, use):
        r["layer"] = l
    return rows


def run_model(name, smoke=False):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, output_attentions=True, attn_implementation="eager",
                                                 dtype=getattr(torch, dt)).eval().to(DEV)
    print(name, "DEVICE", DEV, "DTYPE", dt, "layers", model.config.num_hidden_layers, "heads", model.config.num_attention_heads, flush=True)
    layers = [2, 11, 20] if smoke else None
    A = run_protocol(model, tok, 2 if smoke else A_NWIN, A_SEQ, stats_A, layers)
    B = run_protocol(model, tok, 1 if smoke else B_NWIN, B_SEQ, stats_B, layers)
    del model
    release_memory(DEV)
    return {"model": name, "dtype": dt, "device": DEV, "n_heads": None, "protocol_A": {"n_windows": 2 if smoke else A_NWIN, "seq": A_SEQ, "k_plain": KPLAIN, "k_fam": KFAM, "rows": A},
            "protocol_B": {"n_windows": 1 if smoke else B_NWIN, "seq": B_SEQ, "k_plain": KFAM, "k_fam": KFAM, "rows": B},
            "sink_set": {"thresh": THRESH, "kmax": KMAX}, "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    out_dir = "alignment_study/rerun_smoke" if SMOKE else OUT
    os.makedirs(out_dir, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name in (["Qwen/Qwen2.5-0.5B"] if SMOKE else (ONLY or MODELS)):
        short = name.split("/")[-1]
        path = f"{out_dir}/{short}.json"
        if name in EXCLUDE:
            print("EXCLUDED (run later under the same preregistration)", name, flush=True); continue
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        try:
            res = run_model(name, SMOKE)
            json.dump(res, open(path, "w"), indent=1)
            hs = sum(r["smass"] > 0.4 for r in res["protocol_A"]["rows"])
            print(f"MODEL_DONE {short} layers={len(res['protocol_A']['rows'])} high_sink={hs} dtype={res['dtype']} {res['runtime_s']}s", flush=True)
            if PURGE_LARGE and not SMOKE:
                from hubsfree.adapters import param_count
                if param_count(name) >= 3e9:
                    import shutil
                    from huggingface_hub.constants import HF_HUB_CACHE
                    d = os.path.join(HF_HUB_CACHE, "models--" + name.replace("/", "--"))
                    shutil.rmtree(d, ignore_errors=True); print("PURGED cache", d, flush=True)
        except Exception as e:
            import traceback
            print(f"MODEL_FAILED {short}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            print(traceback.format_exc()[-800:], flush=True)
    print("RERUN_DONE", flush=True)
