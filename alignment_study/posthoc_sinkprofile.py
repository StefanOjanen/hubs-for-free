# POST-HOC (labeled) split of the sink-profile round by layer type: the
# registered clauses are per model over all layers; this reports the same
# quantities separately for high-sink (sink mass > 0.4) and low-sink layers,
# pooled over models, to locate where the generative statement holds.
import glob
import json
import numpy as np
from scipy.stats import spearmanr

rows = []
for f in sorted(glob.glob("alignment_study/sinkprofile/*.json")):
    c = json.load(open(f))
    for r in c["rows"]:
        rows.append(dict(r, model=c["model"]))
out = {"note": "post hoc, not registered", "n_models": len({r["model"] for r in rows})}
for name, sel in (("high_sink", [r for r in rows if r["smass"] > 0.4]), ("low_sink", [r for r in rows if r["smass"] <= 0.4]), ("all", rows)):
    zr = [r["z_real"] for r in sel]; zp = [r["z_profile"] for r in sel]; zd = [r["z_dense"] for r in sel]
    out[name] = {"n_layers": len(sel), "spearman_z_profile": round(float(spearmanr(zr, zp)[0]), 3), "spearman_z_dense": round(float(spearmanr(zr, zd)[0]), 3),
                 "mae_z_profile": round(float(np.median(np.abs(np.array(zr) - np.array(zp)))), 2), "mae_z_dense": round(float(np.median(np.abs(np.array(zr) - np.array(zd)))), 2),
                 "median_profile_share": round(float(np.median([r["profile_share"] for r in sel])), 3)}
json.dump(out, open("alignment_study/posthoc_sinkprofile.json", "w"), indent=1)
print(json.dumps(out, indent=1))
