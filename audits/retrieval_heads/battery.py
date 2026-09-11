# Target 5 (Retrieval Heads 2024), step T2.3: the null battery for the frozen
# statistic, computed online during the needle test so no attention rows
# need storing. COMMITTED BEFORE ANY RESULT EXISTS; not to be run for real
# until audits/PREREGISTRATION4_DRAFT.md is publicly registered. `--dry-run`
# uses the development model (Qwen2.5-0.5B, not the registered audit model)
# at 256 tokens with two instances and writes to /tmp.
#
# Statistic T_5: per-head retrieval score (share of needle tokens copied by
# argmax attention at the matching position) and the set of heads above 0.1.
# Nulls act on the (H, T) attention rows of each generated needle token and
# recompute the argmax hit: (a) random causal softmax rows of the same length;
# (b) per-row marginal-matched permutation of the real rows; (c)
# column-set-preserving permutation (sink set of the layer's map fixed, taken
# from the prompt's own attention: here the modal column of the row block,
# the first token); (d) a config-initialized model of the same architecture
# run on the same instances. Draws per random family: 200 (dry run: 3).
#   python audits/retrieval_heads/battery.py [--dry-run] [--draws=N]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "audits/retrieval_heads")
from hubsfree.adapters import pick_device, pick_dtype, release_memory
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

DRY = "--dry-run" in sys.argv
NAME = "Qwen/Qwen2.5-0.5B" if DRY else "Qwen/Qwen2.5-7B"
CTX = [256] if DRY else [1024, 2048]
NDEPTH = 1 if DRY else 5
DRAWS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--draws=")), 3 if DRY else 200))
N_INIT = 1 if DRY else 5
OUT = "/tmp/retrieval_battery_dryrun.json" if DRY else "audits/retrieval_heads/battery_result.json"
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)
import reproduce as R   # NEEDLE, QUESTION, MAX_NEW, haystacks, build (module-level run is guarded)


def hits_for_rows(rows, positions, rng, draws):
    """rows: (H, T) attention of one generated needle token; positions: needle
    positions matching the token. Returns per-head hit indicators for the real
    rows and for each null family (arrays of shape (H,) or (draws, H))."""
    H, T = rows.shape
    real = np.isin(rows.argmax(-1), positions)
    a = np.stack([np.isin(np.exp(rng.normal(size=(H, T))).argmax(-1), positions) for _ in range(draws)])
    b = np.stack([np.isin(rng.permuted(rows, axis=1).argmax(-1), positions) for _ in range(draws)])
    keep = 0                                                     # sink column of the prompt (first token) held fixed
    free = np.array([c for c in range(T) if c != keep])
    c_hits = []
    for _ in range(draws):
        S = rows.copy(); S[:, free] = rng.permuted(rows[:, free], axis=1); c_hits.append(np.isin(S.argmax(-1), positions))
    return real, a, b, np.stack(c_hits)


def run_needles(model, tok, rng, with_nulls):
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    score = np.zeros((L, H)); nulls = {k: np.zeros((DRAWS, L, H)) for k in ("a", "b", "c")}; n_inst = 0
    depths = np.linspace(0.05, 0.95, NDEPTH) if NDEPTH > 1 else np.array([0.5])
    for ctx in CTX:
        hays = R.haystacks(tok, ctx)
        for hay in hays:
            for depth in depths:
                prompt, pos, needle_ids = R.build(tok, hay, ctx, float(depth))
                at = {}
                for j, t in enumerate(needle_ids):
                    at.setdefault(t, []).append(pos + j)
                x = torch.tensor([prompt], device=DEV)
                out = model.generate(x, max_new_tokens=R.MAX_NEW, do_sample=False, output_attentions=True, return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
                gen = out.sequences[0, x.shape[1]:].tolist(); k = len(needle_ids)
                copied = np.zeros((L, H)); cop_null = {kk: np.zeros((DRAWS, L, H)) for kk in nulls}
                seen = {}
                for step, tid in enumerate(gen):
                    if tid not in at:
                        continue
                    for l in range(L):
                        rows = out.attentions[step][l][0, :, -1, :].float().cpu().numpy().astype(np.float64)
                        real, a, b, c = hits_for_rows(rows, at[tid], rng, DRAWS if with_nulls else 1)
                        key = (l, tid)
                        if key in seen:               # count each needle token once per head, as the reproduction does
                            continue
                        seen[key] = True
                        copied[l] += real
                        if with_nulls:
                            cop_null["a"][:, l] += a; cop_null["b"][:, l] += b; cop_null["c"][:, l] += c
                score += copied / k; n_inst += 1
                for kk in nulls:
                    nulls[kk] += cop_null[kk] / k
                del out
                print(f"  ctx {ctx} depth {depth:.2f} done ({time.time()-t0:.0f}s)", flush=True)
    return score / max(n_inst, 1), {kk: v / max(n_inst, 1) for kk, v in nulls.items()}, n_inst


t0 = time.time()
rng = np.random.default_rng(0)
tok = AutoTokenizer.from_pretrained(NAME)
dt = pick_dtype(NAME, DEV, "auto")
model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
score, nulls, n_inst = run_needles(model, tok, rng, True)
del model; release_memory(DEV)
cfg = AutoConfig.from_pretrained(NAME); unt = []
for seed in range(N_INIT):
    torch.manual_seed(seed)
    um = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval().to(DEV)
    s_u, _, _ = run_needles(um, tok, rng, False); unt.append(s_u); del um; release_memory(DEV)

flat = score.ravel(); real_frac = float((flat > 0.1).mean()); real_top = float(np.sort(flat)[::-1][:10].mean())
res = {"target": "Retrieval Heads 2024", "model": NAME, "contexts": CTX, "instances": n_inst, "draws": DRAWS, "dry_run": DRY,
       "real": {"frac_heads_above_0.1": real_frac, "mean_top10_score": real_top, "max_score": float(flat.max())}, "nulls": {}}
for kk, name in (("a", "a_random"), ("b", "b_marginal"), ("c", "c_colset")):
    fr = np.array([(nulls[kk][d].ravel() > 0.1).mean() for d in range(DRAWS)]); tp = np.array([np.sort(nulls[kk][d].ravel())[::-1][:10].mean() for d in range(DRAWS)])
    res["nulls"][name] = {"frac_above_0.1_median": float(np.median(fr)), "frac_percentile_of_real": float((fr < real_frac).mean() * 100),
                          "top10_median": float(np.median(tp)), "top10_percentile_of_real": float((tp < real_top).mean() * 100),
                          "top10_shrinkage": float(np.median(tp) / real_top) if real_top else None}
fu = [float((s.ravel() > 0.1).mean()) for s in unt]; tu = [float(np.sort(s.ravel())[::-1][:10].mean()) for s in unt]
res["nulls"]["d_untrained"] = {"frac_above_0.1_median": float(np.median(fu)), "top10_median": float(np.median(tu)), "top10_shrinkage": float(np.median(tu) / real_top) if real_top else None, "n": N_INIT}
res["registered_expectation"] = "survives (a), (b), (c), (d) at the 99th percentile with the score shrinking by less than 20 percent under every null"
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1)[:1600]); print("BATTERY_DONE", OUT)
