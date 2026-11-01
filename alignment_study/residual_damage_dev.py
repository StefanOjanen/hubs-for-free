# Development (exploratory, labelled): does the fusion's own residual predict merge damage at 7B scale? Round 29
# converted each of five models many times, each conversion identical to the fusion's reference except in one of
# layers 0 to 4, where the matching was forced to contain one given pair, and recorded the perplexity relative to
# the reference (guard/*.json, guard_dev/Llama-2-7b-hf.json; the dev population holds the severe cases). This script
# recomputes, from the same calibration (128 C4 validation documents of 2,048 tokens), the pair costs of those
# layers (kvfuse.pair_costs: the closed-form residual of the key and value fits over the layer's energy) and gives
# each member its matching's cost increase over the reference matching and its forced pair's own cost; then the
# Spearman correlation of each with the damage, per model and pooled over members. Rounds 20 and 22 measured the
# same question at 1.4B with KL damage and a simulated merge; here the predictor is the residual the method uses.
#   python alignment_study/residual_damage_dev.py [--only=<model>]
import gc
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import kvfuse
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

FILES = {"mistralai/Mistral-7B-Instruct-v0.2": "alignment_study/guard/Mistral-7B-Instruct-v0.2.json",
         "openlm-research/open_llama_3b_v2": "alignment_study/guard/open_llama_3b_v2.json",
         "allenai/OLMo-1B-hf": "alignment_study/guard/OLMo-1B-hf.json",
         "Qwen/Qwen2.5-7B": "alignment_study/guard/Qwen2.5-7B.json",
         "NousResearch/Llama-2-7b-hf": "alignment_study/guard_dev/Llama-2-7b-hf.json"}
REVISIONS = {"NousResearch/Llama-2-7b-hf": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"}
N_CAL, T_CAL, BF16_ABOVE = 128, 2048, 4e9
OUT = "alignment_study/residual_damage"
torch.set_grad_enabled(False)


def spearman(x, y):
    rx, ry = (np.argsort(np.argsort(v, kind="stable")) for v in (x, y))
    # average ranks for ties
    def ranks(v):
        order = np.argsort(v); r = np.empty(len(v)); sv = np.array(v)[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r
    a, b = ranks(x), ranks(y)
    return float(np.corrcoef(a, b)[0, 1]) if len(x) > 2 else float("nan")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    summary = {}
    for name, path in FILES.items():
        if (only and name not in only) or not os.path.exists(path):
            continue
        out = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(out):
            summary[name] = json.load(open(out))["spearman"]; continue
        G = json.load(open(path))
        layers = sorted({m["layer"] for m in G["members"]})
        tok = AutoTokenizer.from_pretrained(name, revision=REVISIONS.get(name))
        model = AutoModelForCausalLM.from_pretrained(name, revision=REVISIONS.get(name), dtype=torch.bfloat16, attn_implementation="sdpa")
        if sum(p.numel() for p in model.parameters()) < BF16_ABOVE:
            model = model.float()
        model = model.eval().to(DEV); kv = K.KV(model)
        cal = C.windows_c4(tok, N_CAL, T_CAL)
        S, ntok = K.second_moments(model, kv, cal, DEV)
        costs = {l: kvfuse.pair_costs(kv, l, S[l].double()) for l in layers}
        del model, kv, S; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        ref_cost = {l: float(sum(costs[l][i, j] for i, j in G["reference_groups"][l])) for l in layers}
        rows = []
        for m in G["members"]:
            l = m["layer"]
            if m["label"].startswith("forced"):                     # "forced i-j": one pair forced into the matching
                i, j = (int(t) for t in m["label"].split(" ")[1].split("-")); forced, fcost = [i, j], float(costs[l][i, j])
            else:                                                   # "matching [[...]]" (4 heads: the pair fixes the matching); its new pairs
                new = m["new_pairs"]; forced, fcost = (new[0] if len(new) == 1 else None), float(sum(costs[l][a, b] for a, b in new) / len(new))
            rows.append({"layer": l, "forced_pair": forced, "forced_pair_cost": fcost,
                         "matching_cost": float(sum(costs[l][a, b] for a, b in m["groups"])), "reference_cost": ref_cost[l],
                         "damage": m["ppl_over_reference"], "check_max_ratio": m.get("max_ratio")})
            rows[-1]["cost_increase"] = rows[-1]["matching_cost"] - ref_cost[l]
        dmg = [r["damage"] for r in rows]
        sp = {"cost_increase": spearman([r["cost_increase"] for r in rows], dmg), "forced_pair_cost": spearman([r["forced_pair_cost"] for r in rows], dmg),
              "one_window_check": spearman([r["check_max_ratio"] for r in rows], dmg) if all(r["check_max_ratio"] is not None for r in rows) else None,
              "members": len(rows), "per_layer_cost_increase": {str(l): spearman([r["cost_increase"] for r in rows if r["layer"] == l], [r["damage"] for r in rows if r["layer"] == l]) for l in layers}}
        res = {"model": name, "source": path, "dev_population": bool(G.get("dev")), "calibration": [N_CAL, T_CAL], "tokens": ntok,
               "reference_cost": ref_cost, "members": rows, "spearman": sp, "seconds": round(time.time() - t0, 1)}
        json.dump(res, open(out, "w"), indent=1)
        summary[name] = sp
        log(f"{name}: {len(rows)} members; Spearman with damage: matching cost increase {sp['cost_increase']:.3f}, forced pair's cost {sp['forced_pair_cost']:.3f}, "
            f"one-window check {sp['one_window_check'] if sp['one_window_check'] is None else round(sp['one_window_check'], 3)}")
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    log("RESIDUAL_DAMAGE_DONE")
