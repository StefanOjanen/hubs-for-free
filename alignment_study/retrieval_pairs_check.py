# CPU check from committed artifacts (no model): where the fusion's residual pairing puts the retrieval heads'
# key-value heads. Scores: audits/retrieval_heads/<model>_scores.npy (layers by query heads, Wu et al.'s protocol);
# a key-value head's score is the largest over the query heads it serves. Pairs: the reference matching of round 29
# (alignment_study/guard/<model>.json, reference_groups; the fusion's residual matching at half the heads).
#   python alignment_study/retrieval_pairs_check.py [--model=Mistral-7B-Instruct-v0.2]
import json
import sys
import numpy as np

model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "Mistral-7B-Instruct-v0.2")
scores = np.load(f"audits/retrieval_heads/{model}_scores.npy")
G = json.load(open(f"alignment_study/guard/{model}.json"))
groups = G["reference_groups"]; L, H = scores.shape
nkv = max(max(g) for layer in groups for g in layer) + 1; rep = H // nkv
kvs = scores.reshape(L, nkv, rep).max(2)                      # (layers, kv heads): largest score among the served query heads
partner = {}
for l, layer in enumerate(groups):
    for g in layer:
        for i in g:
            partner[(l, i)] = [j for j in g if j != i]
out = {"model": model, "layers": L, "kv_heads": nkv, "queries_per_kv": rep, "thresholds": {}}
for th in (0.3, 0.2, 0.1):
    heads = [(int(l), int(i)) for l, i in np.argwhere(kvs >= th)]
    rows = [{"layer": l, "kv": i, "score": round(float(kvs[l, i]), 3), "partners": partner[(l, i)],
             "partner_scores": [round(float(kvs[l, j]), 3) for j in partner[(l, i)]]} for l, i in heads]
    both = sum(1 for r in rows if any(s >= th for s in r["partner_scores"]))
    out["thresholds"][str(th)] = {"kv_heads": len(rows), "paired_with_a_head_below_threshold": len(rows) - both, "paired_with_another_above": both,
                                  "largest_partner_score": max((max(r["partner_scores"]) for r in rows), default=None), "rows": rows}
    print(f"score >= {th}: {len(rows)} key-value heads; {len(rows) - both} paired with a head below the threshold, {both} with another above; "
          f"largest partner score {out['thresholds'][str(th)]['largest_partner_score']}")
    if th == 0.3:
        for r in rows:
            print("  ", r)
json.dump(out, open(f"alignment_study/retrieval_pairs_{model}.json", "w"), indent=1)
print("RETRIEVAL_PAIRS_CHECK_DONE")
