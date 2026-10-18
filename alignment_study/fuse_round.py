# PREREGISTRATION26: converting multi-head attention to grouped-query attention without training, by a
# closed-form fusion of each group's key and value heads (fuse_common.py) instead of aligning and averaging
# them. Groups are taken from round 24 (alignment_study/gqa/, development: gqa_dev/), so the conversion differs
# from round 24's arms only in how the heads of a group are combined:
#   fuse_u/contiguous   unweighted fusion of neighbouring heads (the objective of weight-space fits with an
#                       input-covariance metric, as in CAGA: keys and values fitted without regard to the
#                       queries that read them or the output slices that write them)
#   fuse/contiguous     query- and output-weighted fusion of neighbouring heads
#   fuse/jin            weighted fusion, Jin et al.'s groups
#   fuse/sim_align      weighted fusion, the groups round 24's simulated merge chose
#   fuse_u/sim_align    unweighted fusion, the same groups
# Calibration: the second moment of every layer's attention input on round 24's 8 C4 validation documents of
# 1,024 tokens. Evaluation as in round 24 (zeroshot.py; development: the first 500 examples of each task).
# Checks: groups of one head reproduce the model; fusing in random function-preserving bases (round 24's
# convention null) gives the same model as fusing in the trained basis; restoring the weights restores the
# unconverted perplexity.
#   python alignment_study/fuse_round.py [--dev] --registered=<freeze permalink> [--only=<model>]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import zeroshot as Z
from scale_round_v2 import DEV
from hubsfree.adapters import release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION26.md>; use --dev")
if DEV_RUN:
    MODELS, OUT, PARTS = ["gpt2-medium", "EleutherAI/pythia-410m"], "alignment_study/fuse_dev", "alignment_study/gqa_dev"
else:
    MODELS, OUT, PARTS = ["facebook/opt-1.3b", "EleutherAI/pythia-1.4b", "gpt2-large", "allenai/OLMo-1B-hf", "microsoft/phi-2"], "alignment_study/fuse", "alignment_study/gqa"
G_SIZES = (2, 4)
ARMS = {"fuse_u/contiguous": ("contiguous", False, False), "fuse/contiguous": ("contiguous", True, True),
        "fuse/jin": ("jin", True, True), "fuse/sim_align": ("sim_align", True, True), "fuse_u/sim_align": ("sim_align", False, False)}
N_CAL, T_CAL = 8, 1024
torch.set_grad_enabled(False)


def random_rotations(arch, rng):
    """Uniformly random function-preserving rotations per head (as in gqa_convention.py)."""
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


def run_model(name, log, partial_path):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    arch = C.Arch(model)
    r24 = json.load(open(f"{PARTS}/{name.split('/')[-1]}.json"))
    parts = {g: {grp: [r24["partitions"][str(l)][str(g)][grp] for l in range(arch.L)] for grp in ("contiguous", "jin", "sim_align")} for g in G_SIZES}
    log(f"{name}: kind {arch.kind} layers {arch.L} heads {arch.n} head dim {arch.hd} rotary dims {arch.r}")
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    x0 = cal[:1, :256].to(DEV)
    ref = model(input_ids=x0).logits
    S, ntok = F.second_moments(model, arch, cal, DEV)
    snap = C.snapshot(arch)
    t1 = time.time()
    F.fuse_model(arch, [[[h] for h in range(arch.n)] for _ in range(arch.L)], S)
    ones_diff = float((model(input_ids=x0).logits - ref).abs().max()); C.restore(arch, snap)
    log(f"  second moments from {ntok} tokens; groups of one reproduce the model: max logit difference {ones_diff:.2e} ({time.time() - t1:.0f}s)")
    # basis invariance: fuse (sim_align groups, g = 2) in the trained basis and in a random basis
    F.fuse_model(arch, parts[2]["sim_align"], S)
    lg_fused = model(input_ids=x0).logits; C.restore(arch, snap)
    R0 = random_rotations(arch, np.random.default_rng(1000))
    for (l, h), (rk, sv) in R0.items():
        C.apply_rotation(arch, l, h, torch.tensor(rk), torch.tensor(sv))
    rot_check = float((model(input_ids=x0).logits - ref).abs().max())
    S_rot, _ = F.second_moments(model, arch, cal, DEV)
    F.fuse_model(arch, parts[2]["sim_align"], S_rot)
    basis_diff = float((model(input_ids=x0).logits - lg_fused).abs().max()); C.restore(arch, snap)
    log(f"  random basis leaves the model unchanged to {rot_check:.2e}; fused in it, the converted model differs from the trained-basis fusion by {basis_diff:.2e}")
    tasks = Z.load_tasks(subset=500 if DEV_RUN else None)
    reqs = Z.build_requests(tok, tasks)
    X = Z.wikitext2_windows(tok)
    res = json.load(open(partial_path)) if os.path.exists(partial_path) else {"evals": {}}
    res.update({"model": name, "kind": arch.kind, "layers": arch.L, "heads": arch.n, "head_dim": arch.hd, "rotary_dims": arch.r,
                "calibration": {"source": "allenai/c4 en validation shard 0", "windows": N_CAL, "tokens": T_CAL},
                "groups_of_one_max_logit_diff": ones_diff, "random_basis_function_check": rot_check,
                "basis_invariance_max_logit_diff": basis_diff, "logit_scale": float(ref.abs().max()), "partitions_from": PARTS})

    def evaluate(key):
        if key in res["evals"]:
            log(f"  {key}: cached"); return
        t2 = time.time()
        r = Z.evaluate_tasks(model, tok, tasks, reqs, DEV); r["wikitext2_ppl"] = Z.perplexity(model, X, DEV)
        r["seconds"] = round(time.time() - t2, 1); res["evals"][key] = r
        json.dump(res, open(partial_path, "w"))
        log(f"  {key}: ppl {r['wikitext2_ppl']:.2f} avg {r['average']:.4f}")
    evaluate("base")
    for g in G_SIZES:
        for arm, (grp, qw, ow) in ARMS.items():
            key = f"g{g}/{arm}"
            if key not in res["evals"]:
                t3 = time.time()
                F.fuse_model(arch, parts[g][grp], S, qw, ow)
                log(f"  {key}: fused in {time.time() - t3:.0f}s")
            evaluate(key)
            C.restore(arch, snap)
    res["restore_check_ppl"] = Z.perplexity(model, X, DEV)
    res["runtime_s"] = round(time.time() - t0, 1)
    del model; release_memory(DEV)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    for name in MODELS:
        if only and name not in only:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        partial = path + ".partial"
        res = run_model(name, log, partial)
        res["registered"] = REGISTERED; res["dev"] = DEV_RUN
        json.dump(res, open(path, "w"), indent=1)
        if os.path.exists(partial):
            os.remove(partial)
        log(f"MODEL_DONE {name}")
    print("FUSE_ROUND_DONE", flush=True)
