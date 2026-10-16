# Convention null for PREREGISTRATION24. Rotating every head's query/key space by a random orthogonal
# matrix (rotary-compatible: a random 2-D rotation in each rotary plane, any orthogonal matrix elsewhere)
# and its value/output space by another leaves the model's function unchanged. A conversion that depends
# only on the function should not care. For each group size, neighbouring heads are pooled
#   unaligned  as in Ainslie et al. (2023), and
#   aligned    after Jin et al.'s generalized Procrustes alignment within each group (recomputed in each basis),
# in the original basis and in three random bases. Reported: perplexity on the first 64 WikiText-2 windows,
# the function check of each random basis (max logit difference before pooling), and for the aligned
# conversion the max logit difference between each random basis's converted model and the original basis's.
#   python alignment_study/gqa_convention.py [--dev | --check] --registered=<freeze permalink>
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import zeroshot as Z
from scale_round_v2 import DEV
from hubsfree.adapters import release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV_RUN = "--dev" in sys.argv
CHECK = "--check" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not (DEV_RUN or CHECK) and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION24.md>; use --dev or --check")
if CHECK:
    MODELS, OUT = ["facebook/opt-125m", "JackFram/llama-160m"], "alignment_study/gqa_convention_check"
elif DEV_RUN:
    MODELS, OUT = ["gpt2-medium", "EleutherAI/pythia-410m"], "alignment_study/gqa_convention_dev"
else:
    MODELS, OUT = ["facebook/opt-1.3b", "EleutherAI/pythia-1.4b", "gpt2-large", "allenai/OLMo-1B-hf", "microsoft/phi-2"], "alignment_study/gqa_convention"
G_SIZES, SEEDS, N_PPL = (2, 4), (0, 1, 2), 64
N_CAL, T_CAL = (2, 512) if CHECK else (8, 1024)
torch.set_grad_enabled(False)


def random_rotations(arch, rng):
    """{(l, h): (Rk, Sv)} uniformly random within the function-preserving groups."""
    out = {}
    hd, r = arch.hd, arch.r
    for l in range(arch.L):
        for h in range(arch.n):
            Rk = np.zeros((hd, hd))
            if r:
                i = np.arange(r // 2); th = rng.uniform(0, 2 * np.pi, r // 2); c, s = np.cos(th), np.sin(th)
                Rk[i, i] = c; Rk[i, i + r // 2] = s; Rk[i + r // 2, i] = -s; Rk[i + r // 2, i + r // 2] = c
            if r < hd:
                q, rr = np.linalg.qr(rng.standard_normal((hd - r, hd - r))); Rk[r:, r:] = q * np.sign(np.diag(rr))
            q, rr = np.linalg.qr(rng.standard_normal((hd, hd))); Sv = q * np.sign(np.diag(rr))
            out[(l, h)] = (Rk, Sv)
    return out


def contiguous_alignment(arch, grams, g, R0=None):
    """Jin et al.'s GPA rotations for neighbouring groups, from Gram blocks of normalized caches in the
    basis rotated by R0 (X_a R0_a: blocks R0_a^T C_ab R0_b)."""
    rot = {}
    for l in range(arch.L):
        CK, CV = grams[l]
        for s in range(0, arch.n, g):
            G = list(range(s, s + g))
            tr = lambda Cb, k: [[(R0[(l, a)][k].T @ Cb[a, b] @ R0[(l, b)][k]) if R0 else Cb[a, b] for b in G] for a in G]
            keys = C.gpa_gram(tr(CK, 0), lambda Cm: C.rope_procrustes_C(Cm, arch.r))
            vals = C.gpa_gram(tr(CV, 1), C.procrustes_C)
            for h, rk, sv in zip(G, keys, vals):
                rot[(l, h)] = (rk, sv)
    return rot


def apply_all(arch, rot):
    for (l, h), (rk, sv) in rot.items():
        C.apply_rotation(arch, l, h, torch.tensor(rk), torch.tensor(sv))


def pool_contiguous(arch, g):
    for l in range(arch.L):
        for s in range(0, arch.n, g):
            C.pool_group(arch, l, list(range(s, s + g)))


def run_model(name, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="eager").eval().to(DEV)
    arch = C.Arch(model)
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    x0 = cal[:1, :256].to(DEV)
    try:
        model.set_attn_implementation("gqa_capture")
    except Exception:
        model.config._attn_implementation = "gqa_capture"
    C.CAP.clear()
    for w in range(cal.shape[0]):
        model(input_ids=cal[w:w + 1].to(DEV))
    try:
        model.set_attn_implementation("sdpa")
    except Exception:
        model.config._attn_implementation = "sdpa"
    grams = {}
    for l in range(arch.L):
        K = torch.stack([w[1] for w in C.CAP[l]]).float().to(DEV); V = torch.stack([w[2] for w in C.CAP[l]]).float().to(DEV)
        flat = lambda X: C.unit_rows(X.transpose(0, 1).reshape(arch.n, -1, X.shape[-1]))
        gram = lambda X: torch.einsum("anx,bny->abxy", X, X).cpu().double().numpy()
        grams[l] = (gram(flat(K)), gram(flat(V)))
        del K, V; C.CAP.pop(l)
    X = Z.wikitext2_windows(tok)[:N_PPL]
    ref = model(input_ids=x0).logits
    snap = C.snapshot(arch)
    res = {"model": name, "ppl_windows": N_PPL, "base_ppl": Z.perplexity(model, X, DEV), "groups": {}}
    log(f"{name}: base ppl {res['base_ppl']:.2f} on {N_PPL} windows")
    for g in G_SIZES:
        out = {"unaligned": {}, "aligned": {}, "function_check": {}, "aligned_vs_original_basis": {}}
        pool_contiguous(arch, g); out["unaligned"]["original"] = Z.perplexity(model, X, DEV); C.restore(arch, snap)
        apply_all(arch, contiguous_alignment(arch, grams, g)); pool_contiguous(arch, g)
        out["aligned"]["original"] = Z.perplexity(model, X, DEV); lg0 = model(input_ids=x0).logits; C.restore(arch, snap)
        for seed in SEEDS:
            R0 = random_rotations(arch, np.random.default_rng(1000 + seed))
            apply_all(arch, R0)
            out["function_check"][str(seed)] = float((model(input_ids=x0).logits - ref).abs().max())
            snap_r = C.snapshot(arch)
            pool_contiguous(arch, g); out["unaligned"][str(seed)] = Z.perplexity(model, X, DEV); C.restore(arch, snap_r)
            apply_all(arch, contiguous_alignment(arch, grams, g, R0)); pool_contiguous(arch, g)
            out["aligned"][str(seed)] = Z.perplexity(model, X, DEV)
            out["aligned_vs_original_basis"][str(seed)] = float((model(input_ids=x0).logits - lg0).abs().max())
            C.restore(arch, snap)
        res["groups"][str(g)] = out
        log(f"  g={g}: unaligned " + ", ".join(f"{k} {v:.1f}" for k, v in out["unaligned"].items())
            + " | aligned " + ", ".join(f"{k} {v:.2f}" for k, v in out["aligned"].items())
            + " | aligned-model difference across bases " + ", ".join(f"{v:.1e}" for v in out["aligned_vs_original_basis"].values()))
    res["max_logit_scale"] = float(ref.abs().max())
    res["runtime_s"] = round(time.time() - t0, 1)
    del model; release_memory(DEV)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name in MODELS:
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        res = run_model(name, log); res["registered"] = REGISTERED; res["dev"] = DEV_RUN; res["check"] = CHECK
        json.dump(res, open(path, "w"), indent=1)
        log(f"MODEL_DONE {name}")
    print("GQA_CONVENTION_DONE", flush=True)
