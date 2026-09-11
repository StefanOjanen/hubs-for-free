# Target 4 (Dewage et al. 2026), step T2.3: the weight-level null battery for
# the frozen statistic. COMMITTED BEFORE ANY RESULT EXISTS; not to be run for
# real until audits/PREREGISTRATION4_DRAFT.md is publicly registered.
# `--dry-run` exercises the code on random matrices only and writes to /tmp.
#
# Statistic T_4 per matrix: number of Marchenko-Pastur outliers and their
# share of spectral energy (recipe of reproduce.py). Nulls per matrix
# (PREREGISTRATION4_DRAFT.md, Target 4): (a') Gaussian weights of the same
# shape and Frobenius norm; (b') row-norm-matched random weights (each row a
# random Gaussian direction scaled to the real row's norm); (c') within-matrix
# permutation of the real entries; (d') the same matrices of a
# config-initialized model of the same architecture, five seeds. Draws per
# random family: 200 (dry run: 3). Outcome per matrix type from the mean
# over layers: percentile of the real count and energy share in each null
# and shrinkage = null median / real.
#   python audits/dewage2026/battery.py [--dry-run] [--draws=N] [--layers=N]
import glob
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "audits/dewage2026")
from reproduce import mp_outliers   # same recipe as the reproduction

NAME = "mistralai/Mistral-7B-v0.1"
DRY = "--dry-run" in sys.argv
DRAWS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--draws=")), 3 if DRY else 200))
NLAY = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--layers=")), 2 if DRY else 32))
N_INIT = 1 if DRY else 5
OUT = "/tmp/dewage2026_battery_dryrun.json" if DRY else "audits/dewage2026/battery_result.json"
TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
rng = np.random.default_rng(0)


def gaussian_matched(W):
    G = rng.normal(size=W.shape); return G * (np.linalg.norm(W) / np.linalg.norm(G))


def row_norm_matched(W):
    G = rng.normal(size=W.shape); G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    return G * np.linalg.norm(W, axis=1, keepdims=True)


def permuted(W):
    return rng.permutation(W.ravel()).reshape(W.shape)


def real_matrices():
    if DRY:
        for l in range(NLAY):
            for t in TYPES:
                shape = (64, 256) if t in ("k_proj", "v_proj") else (256, 256)
                # planted structure: a few large directions plus heavy-tailed noise
                W = rng.standard_t(3, size=shape) * 0.02 + (rng.normal(size=(shape[0], 3)) @ rng.normal(size=(3, shape[1]))) * 0.05
                yield l, t, W
        return
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(NAME, allow_patterns=["*.safetensors", "*.json"])
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                if ".self_attn." in key and key.endswith(".weight") and any(p in key for p in TYPES):
                    l = int(key.split(".layers.")[1].split(".")[0]); t = key.split(".self_attn.")[1].split(".")[0]
                    if l < NLAY:
                        yield l, t, sf.get_tensor(key).float().numpy()


def untrained_matrices():
    if DRY:
        for seed in range(N_INIT):
            for l in range(NLAY):
                for t in TYPES:
                    shape = (64, 256) if t in ("k_proj", "v_proj") else (256, 256)
                    yield seed, l, t, rng.normal(size=shape) * 0.02
        return
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(NAME)
    for seed in range(N_INIT):
        torch.manual_seed(seed)
        with torch.device("cpu"):
            m = AutoModelForCausalLM.from_config(cfg)
        for l in range(NLAY):
            attn = m.model.layers[l].self_attn
            for t in TYPES:
                yield seed, l, t, getattr(attn, t).weight.detach().float().numpy()
        del m


t0 = time.time()
rows = []
for l, t, W in real_matrices():
    r = {"layer": l, "type": t, "real": mp_outliers(W)}
    for fam, f in (("a_gaussian", gaussian_matched), ("b_rownorm", row_norm_matched), ("c_permuted", permuted)):
        r[fam] = [mp_outliers(f(W)) for _ in range(DRAWS)]
    rows.append(r)
    print(f"  L{l:2d} {t:6s} real outliers {r['real']['outliers']:4d} energy {r['real']['energy_frac_outliers']:.3f} | medians: gaussian {np.median([x['outliers'] for x in r['a_gaussian']]):.0f}, rownorm {np.median([x['outliers'] for x in r['b_rownorm']]):.0f}, permuted {np.median([x['outliers'] for x in r['c_permuted']]):.0f} ({time.time()-t0:.0f}s)", flush=True)
unt = {}
for seed, l, t, W in untrained_matrices():
    unt.setdefault((l, t), []).append(mp_outliers(W))

res = {"target": "Dewage et al. 2026", "model": NAME, "layers": NLAY, "draws": DRAWS, "dry_run": DRY, "types": {}}
for t in TYPES:
    rs = [r for r in rows if r["type"] == t]
    real_c = float(np.mean([r["real"]["outliers"] for r in rs])); real_e = float(np.mean([r["real"]["energy_frac_outliers"] for r in rs]))
    entry = {"real_mean_outliers": real_c, "real_mean_energy_share": real_e, "nulls": {}}
    for fam in ("a_gaussian", "b_rownorm", "c_permuted"):
        cs = np.array([[x["outliers"] for x in r[fam]] for r in rs]).mean(0)          # per draw, mean over layers
        es = np.array([[x["energy_frac_outliers"] for x in r[fam]] for r in rs]).mean(0)
        entry["nulls"][fam] = {"outliers_median": float(np.median(cs)), "outliers_percentile_of_real": float((cs < real_c).mean() * 100),
                               "outliers_shrinkage": float(np.median(cs) / real_c) if real_c else None,
                               "energy_median": float(np.median(es)), "energy_percentile_of_real": float((es < real_e).mean() * 100),
                               "energy_shrinkage": float(np.median(es) / real_e) if real_e else None}
    uc = [np.mean([unt[(r["layer"], t)][s]["outliers"] for r in rs]) for s in range(N_INIT)]
    ue = [np.mean([unt[(r["layer"], t)][s]["energy_frac_outliers"] for r in rs]) for s in range(N_INIT)]
    entry["nulls"]["d_untrained"] = {"outliers_median": float(np.median(uc)), "outliers_shrinkage": float(np.median(uc) / real_c) if real_c else None,
                                     "energy_median": float(np.median(ue)), "n": N_INIT}
    res["types"][t] = entry
res["registered_expectation"] = ("survives (a') at the 99th percentile in every type; against (c') at least half of the outlier count is matched; "
                                 "against (b') the energy share shrinks by less than 50 percent; (d') reported as the initialization baseline")
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["types"], indent=1)[:1500]); print("BATTERY_DONE", OUT)
