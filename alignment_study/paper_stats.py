# Statistics over models for the KV-fusion paper: for each pooled comparison, the per-model ratio of log-perplexity
# increases (ours over the baseline's, both over the unconverted model), the pooled ratio (sum over sum), the
# geometric mean, the count of models where ours is lower, a two-sided sign test, and a percentile bootstrap of the
# pooled ratio over models (10,000 resamples, seed 0). Reads the committed result files; writes
# alignment_study/paper_stats.json. Models are the unit; nothing here is a test over tokens or windows.
#   python alignment_study/paper_stats.py
import glob
import json
import math
import os
import numpy as np

rng = np.random.default_rng(0)
B = 10000


def comparison(name, rows, baseline_label):
    """rows: {model: (dlog_ours, dlog_baseline)} with dlog = ln(converted / unconverted)."""
    m = sorted(rows)
    ours = np.array([rows[k][0] for k in m]); base = np.array([rows[k][1] for k in m])
    ratio = ours / base
    pooled = float(ours.sum() / base.sum())
    idx = rng.integers(0, len(m), size=(B, len(m)))
    boot = ours[idx].sum(1) / base[idx].sum(1)
    lower = int((ours < base).sum()); n = len(m)
    p_sign = sum(math.comb(n, k) for k in range(n + 1) if abs(k - n / 2) >= abs(lower - n / 2)) / 2 ** n
    return {"baseline": baseline_label, "models": m, "per_model_ratio": {k: float(r) for k, r in zip(m, ratio)}, "pooled_ratio": pooled,
            "geometric_mean_ratio": float(np.exp(np.log(ratio).mean())), "lower_in": [lower, n], "sign_test_p": p_sign,
            "bootstrap_pooled_2.5_97.5": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


out = {}
# round 26: the fusion against round 24's best recipe (aligned pooling on the simulated-merge groups), g = 2 and g = 4
T = json.load(open("alignment_study/fuse_results.json"))["table"]
for g in (2, 4):
    rows = {k.rsplit(" ", 1)[0]: (v["fuse/sim_align"]["dlog"], v["sim_align"]["dlog"]) for k, v in T.items() if k.endswith(f" g{g}")}
    out[f"round26_fusion_vs_aligned_pooling_g{g}"] = comparison("r26", rows, "aligned mean pooling, simulated-merge groups (round 24's best)")
    rows = {k.rsplit(" ", 1)[0]: (v["fuse/sim_align"]["dlog"], v["fuse_u/sim_align"]["dlog"]) for k, v in T.items() if k.endswith(f" g{g}")}
    out[f"round26_weighted_vs_unweighted_fit_g{g}"] = comparison("r26", rows, "unweighted closed-form fit, same groups")
# round 27: the fusion against averaging in models that already share key-value heads (every cell)
T = json.load(open("alignment_study/fusekv_results.json"))["table"]
out["round27_fusion_vs_averaging"] = comparison("r27", {k: (v["fuse"]["dlog"], v["mean"]["dlog"]) for k, v in T.items()}, "averaging (mean pooling)")
out["round27_weighted_vs_unweighted_fit"] = comparison("r27", {k: (v["fuse"]["dlog"], v["fuse_u"]["dlog"]) for k, v in T.items()}, "unweighted closed-form fit")
# round 30: rebalanced against as-computed fused model, 4-bit rounding cost (ln(rounded / unrounded))
Q = json.load(open("alignment_study/balance_results.json"))["Q3"]
out["round30_rebalanced_vs_fused_4bit_cost"] = comparison("r30", {k: (Q["cost_rebalanced"][k], Q["cost_fused"][k]) for k in Q["cost_fused"]}, "fused model as computed")
for bits in (3,):
    rows = {}
    for f in glob.glob("alignment_study/balance/Q_*.json"):
        r = json.load(open(f)); k = os.path.basename(f)[2:-5]
        rows[k] = (math.log(r[f"balanced_w{bits}"] / r["balanced"]), math.log(r[f"fused_w{bits}"] / r["fused"]))
    out[f"round30_rebalanced_vs_fused_{bits}bit_cost"] = comparison("r30", rows, "fused model as computed")
# round 31: received-attention metric against the plain fusion (Qwen2.5-7B from the post hoc correction)
P = json.load(open("alignment_study/attnw_results.json"))["per_model"]
qc = json.load(open("alignment_study/attnw_correction/summary.json"))["qwen_corrected"]
rows = {k: (math.log((qc if k == "Qwen2.5-7B" else v)["attention"] / v["base"]), math.log((qc if k == "Qwen2.5-7B" else v)["plain"] / v["base"])) for k, v in P.items()}
out["round31_received_attention_vs_plain"] = comparison("r31", rows, "plain fusion (S)")
# round 32: the sensitivity metrics against the plain fusion, when its results exist
if os.path.exists("alignment_study/attnw2_results.json"):
    P = json.load(open("alignment_study/attnw2_results.json"))["per_model"]
    rows = {k: (math.log(v["sens"] / v["base"]), math.log(v["plain"] / v["base"])) for k, v in P.items()}
    out["round32_sensitivity_metrics_vs_plain"] = comparison("r32", rows, "plain fusion (S)")
    rows = {k: rows[k] for k in rows if P[k]["tier"] == "new"}
    out["round32_sensitivity_metrics_vs_plain_new_models"] = comparison("r32", rows, "plain fusion (S), models new to this work")
json.dump(out, open("alignment_study/paper_stats.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k}: pooled {v['pooled_ratio']:.3f} [{v['bootstrap_pooled_2.5_97.5'][0]:.3f}, {v['bootstrap_pooled_2.5_97.5'][1]:.3f}], "
          f"geometric mean {v['geometric_mean_ratio']:.3f}, lower in {v['lower_in'][0]} of {v['lower_in'][1]} (sign test p {v['sign_test_p']:.3f})")
print("PAPER_STATS_DONE")
