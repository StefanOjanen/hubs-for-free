# Applies the rule committed in PREREGISTRATION32_rule.md (9c17ad1) to the development values in attnw_mix_dev/:
# the arm, other than plain, whose smallest relative improvement over plain across Llama-160m, TinyLlama and
# OpenLLaMA-3B is largest; none if that smallest improvement is not positive; ties within 0.1 percentage points go
# to the arm listed first. Writes attnw_mix_dev/rule_result.json.
#   python alignment_study/attnw_mix_rule.py
import json
import math

MODELS = ["llama-160m", "TinyLlama_v1.1", "open_llama_3b_v2"]
ORDER = ["round31", "mixv", "plainmixv", "sens", "sensk"]

if __name__ == "__main__":
    R = {m: json.load(open(f"alignment_study/attnw_mix_dev/{m}.json")) for m in MODELS}
    imp = {a: {m: 1 - R[m]["arms"][a]["ppl"] / R[m]["arms"]["plain"]["ppl"] for m in MODELS} for a in ORDER}
    worst = {a: min(v.values()) for a, v in imp.items()}
    best = max(worst.values())
    pick = next(a for a in ORDER if worst[a] >= best - 0.001)
    pooled = {a: sum(math.log(R[m]["arms"][a]["ppl"] / R[m]["base"]) for m in MODELS)
                 / sum(math.log(R[m]["arms"]["plain"]["ppl"] / R[m]["base"]) for m in MODELS) for a in ORDER}
    res = {"rule": "PREREGISTRATION32_rule.md", "improvement_over_plain": imp, "smallest_improvement": worst,
           "pick": pick if best > 0 else None, "pooled_ratio_dev": pooled,
           "ppl": {m: {a: R[m]["arms"][a]["ppl"] for a in ["plain"] + ORDER} for m in MODELS},
           "base": {m: R[m]["base"] for m in MODELS}}
    json.dump(res, open("alignment_study/attnw_mix_dev/rule_result.json", "w"), indent=1)
    for a in ORDER:
        print(f"{a:10s} smallest {100 * worst[a]:+6.2f}%  " + "  ".join(f"{m} {100 * imp[a][m]:+6.2f}%" for m in MODELS)
              + f"  pooled {pooled[a]:.3f}")
    print("pick:", res["pick"])
