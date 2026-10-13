# POST HOC, NOT REGISTERED. Round 7's M2 (a layer's shared energy predicts its
# merge tolerance) failed with the change in NLL as outcome. Round 11 measured
# a cleaner outcome (KL) on new pairs. This asks whether M2's failure was
# noise: per merged layer of round 11, the layer's shared-energy fraction from
# the committed rerun (protocol A) against the median merge KL over the
# layer's pairs and against the KL of its cheapest pair. No threshold.
import json
import numpy as np
from scipy.stats import spearmanr

MODELS = {"Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B", "TinyLlama/TinyLlama_v1.1": "TinyLlama_v1.1",
          "mistralai/Mistral-7B-v0.1": "Mistral-7B-v0.1", "Qwen/Qwen2.5-7B": "Qwen2.5-7B"}
out = {"label": "post hoc, not registered", "per_model": {}}
pool_se, pool_med, pool_min, pool_layer = [], [], [], []
for name, short in MODELS.items():
    f = json.load(open(f"alignment_study/functional/{short}.json"))
    rr = {r["layer"]: r for r in json.load(open(f"alignment_study/rerun/{short}.json"))["protocol_A"]["rows"]}
    se, med, mn, ls = [], [], [], []
    for l, sw in f["sweep"].items():
        kl = [v["kl"] for v in sw.values()]
        se.append(rr[int(l)]["sharedE"]); med.append(float(np.median(kl))); mn.append(float(np.min(kl))); ls.append(int(l))
    out["per_model"][name] = {"n_layers": len(se), "rho_sharedE_median_kl": float(spearmanr(se, med)[0]), "rho_sharedE_min_kl": float(spearmanr(se, mn)[0])}
    pool_se += se; pool_med += med; pool_min += mn; pool_layer += ls
out["pooled"] = {"n_layers": len(pool_se), "rho_sharedE_median_kl": float(spearmanr(pool_se, pool_med)[0]), "rho_sharedE_min_kl": float(spearmanr(pool_se, pool_min)[0])}
json.dump(out, open("alignment_study/posthoc/sharedE_vs_merge_kl.json", "w"), indent=1)
print(json.dumps(out, indent=1))
