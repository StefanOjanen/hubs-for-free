# STANDARD-STATISTICS ROUND (PREREGISTRATION12.md): the statistics the field
# uses, with function in the loop. Per model: the selection statistics on
# every layer (round-11 predictors plus Clark JS, map cosine, CHAI-style
# correlations, Voita confidence, entropy, sink mass, output norm), the
# single-head ablation outcome in the layers fixed below, and the multi-pair
# merge outcome for every criterion at budgets of 1 and 2 pairs per KV group
# in layers 1 to L-1. Pair-level merge outcomes for Part A come from the
# round-11 file of the same model (outcomes measured before this round's
# statistics were computed). One JSON per model; finished models are skipped.
#   python alignment_study/standard_round.py [--dev] [--only=name]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import functional_common as F
import standard_common as SC
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV_RUN = "--dev" in sys.argv
PLAN = {
    "Qwen/Qwen2.5-1.5B":         {"ablate": list(range(1, 28, 2)), "n_eval": 8, "pair_outcomes": "alignment_study/functional/Qwen2.5-1.5B.json"},
    "TinyLlama/TinyLlama_v1.1":  {"ablate": list(range(1, 22, 2)), "n_eval": 8, "pair_outcomes": "alignment_study/functional/TinyLlama_v1.1.json"},
    "mistralai/Mistral-7B-v0.1": {"ablate": [2, 10, 18, 26], "n_eval": 4, "pair_outcomes": "alignment_study/functional/Mistral-7B-v0.1.json"},
    "Qwen/Qwen2.5-7B":           {"ablate": [2, 9, 16, 23], "n_eval": 4, "pair_outcomes": "alignment_study/functional/Qwen2.5-7B.json"},
}
DEV_PLAN = {"Qwen/Qwen2.5-0.5B": {"ablate": list(range(24)), "n_eval": 8, "pair_outcomes": "alignment_study/functional_dev_calibration.json"}}
OUT = "alignment_study/standard_dev" if DEV_RUN else "alignment_study/standard"
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)


def run_model(name, plan):
    t0 = time.time()
    log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L, n, nkv, hd, group = F.geometry(model)
    log(f"{name} DEVICE {DEV} DTYPE {dt} layers {L} heads {n} kv {nkv} hd {hd} group {group}")
    M.N_SEL, M.T_SEL, M.N_EVAL, M.T_EVAL = F.N_SEL, F.T_SEL, plan["n_eval"], F.T_EVAL
    sel = M.selection_windows(tok); X = M.eval_windows(tok)
    stats = SC.selection_stats(model, sel, n, hd, group, L)
    log("selection statistics done")
    base = F.base_cache(model, X)
    multi = SC.multi_merge_all(model, X, base, stats, L, n, hd, group, log=log)
    abl = SC.ablation_sweep(model, X, base, plan["ablate"], n, hd, log=log)
    check_kl, _ = F.kl_eval(model, X, base)
    del model; release_memory(DEV)
    return {"model": name, "dtype": dt, "device": DEV, "layers": L, "heads": n, "kv_heads": nkv, "head_dim": hd, "group": group,
            "plan": plan, "eval_tokens": int(X.numel()), "restore_check_kl": float(np.mean(check_kl)),
            "stats": {str(l): stats[l] for l in range(L)}, "multi": multi, "ablation": {str(l): abl[l] for l in plan["ablate"]},
            "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name, plan in (DEV_PLAN if DEV_RUN else PLAN).items():
        if ONLY and name not in ONLY:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        try:
            res = run_model(name, plan)
            json.dump(res, open(path, "w"), indent=1)
            print(f"MODEL_DONE {name} {res['runtime_s']}s", flush=True)
        except Exception as e:
            import traceback
            print(f"MODEL_FAILED {name}: {type(e).__name__}: {str(e)[:300]}", flush=True); print(traceback.format_exc()[-1500:], flush=True)
    print("STANDARD_ROUND_DONE", flush=True)
