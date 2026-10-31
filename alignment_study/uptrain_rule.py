# Applies the rule committed in PREREGISTRATION33_rule.md (e2c9852) to the 5M-token development run on Llama-160m
# (alignment_study/uptrain_dev/llama-160m_5M.json): F is the rebalanced fusion's retained lead over averaging at the
# end, lead(end) / lead(start) with lead = ln(ppl of averaging) - ln(ppl of the arm), rounded down to the nearest
# tenth and capped at 0.5; F = 0.1 if the lead at the end is not positive. Writes
# alignment_study/uptrain_dev/rule_result.json with every arm's lead at every checkpoint.
#   python alignment_study/uptrain_rule.py
import json
import math

SRC = "alignment_study/uptrain_dev/llama-160m_5M.json"

if __name__ == "__main__":
    r = json.load(open(SRC))
    arms = r["arms"]
    mean = {c["step"]: c["ppl"] for c in arms["mean"]["curve"]}
    leads = {a: {str(c["step"]): math.log(mean[c["step"]]) - math.log(c["ppl"]) for c in arms[a]["curve"] if c["step"] in mean}
             for a in arms if a != "mean"}
    fuse = arms["fuse"]["curve"]
    start, end = leads["fuse"][str(fuse[0]["step"])], leads["fuse"][str(fuse[-1]["step"])]
    retained = end / start if start > 0 else float("nan")
    F = 0.1 if end <= 0 else min(0.5, math.floor(10 * retained) / 10)
    res = {"rule": "PREREGISTRATION33_rule.md", "source": SRC, "tokens": r["tokens"], "lr": r["lr"], "leads_ln_mean_over_arm": leads,
           "fuse_lead_start": start, "fuse_lead_end": end, "fuse_retained_lead": retained, "F": F,
           "final_ppl": {a: arms[a]["curve"][-1]["ppl"] for a in arms}, "zeroshot_after": {a: arms[a].get("zeroshot_after", {}).get("average") for a in arms}}
    json.dump(res, open("alignment_study/uptrain_dev/rule_result.json", "w"), indent=1)
    for a in arms:
        print(f"{a:10s} final ppl {res['final_ppl'][a]:.2f}" + (f", lead over averaging {start if a == 'fuse' else leads.get(a, {}).get(str(fuse[0]['step']), float('nan')):.3f} at start, "
              f"{leads.get(a, {}).get(str(fuse[-1]['step']), float('nan')):.3f} at end" if a != "mean" else ""))
    print(f"fusion retained lead {retained:.3f}; F = {F}")
