# PREREGISTRATION14: preregistration 12's head-importance and multi-pair-merge
# clauses at four times the context. Same instruments (standard_common.py),
# same clauses and thresholds; selection windows of 1,024 tokens cut from the
# concatenated WikiText-103 validation split (8 windows; validation texts are
# rarely that long on their own), evaluation windows of 1,024 tokens from the
# concatenated test split (4 windows). Models: Qwen2.5-1.5B and TinyLlama,
# the two fast models of rounds 11 and 12. One JSON per model.
#   python alignment_study/standard_long.py
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

T_LONG, N_SEL_LONG, N_EVAL_LONG = 1024, 8, 4
PLAN = {"Qwen/Qwen2.5-1.5B": {"ablate": [3, 9, 15, 21, 27]}, "TinyLlama/TinyLlama_v1.1": {"ablate": [3, 8, 13, 18, 21]}}
OUT = "alignment_study/standard_long"


def selection_windows_long(tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt")["input_ids"][0]
    n = ids.numel() // T_LONG
    picks = np.linspace(0, n - 1, N_SEL_LONG).round().astype(int)
    return [ids[p * T_LONG:(p + 1) * T_LONG][None] for p in picks]


def run_model(name, plan):
    t0 = time.time()
    log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L, n, nkv, hd, group = F.geometry(model)
    log(f"{name} DTYPE {dt} layers {L} heads {n} group {group} T {T_LONG}")
    M.N_EVAL, M.T_EVAL = N_EVAL_LONG, T_LONG
    sel = selection_windows_long(tok); X = M.eval_windows(tok)
    stats = SC.selection_stats(model, sel, n, hd, group, L)
    log("selection statistics done")
    base = F.base_cache(model, X)
    multi = SC.multi_merge_all(model, X, base, stats, L, n, hd, group, log=log)
    abl = SC.ablation_sweep(model, X, base, plan["ablate"], n, hd, log=log)
    check, _ = F.kl_eval(model, X, base)
    del model; release_memory(DEV)
    return {"model": name, "dtype": dt, "device": DEV, "layers": L, "heads": n, "group": group, "T": T_LONG,
            "plan": plan, "eval_tokens": int(X.numel()), "restore_check_kl": float(np.mean(check)),
            "stats": {str(l): stats[l] for l in range(L)}, "multi": multi, "ablation": {str(l): abl[l] for l in plan["ablate"]},
            "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name, plan in PLAN.items():
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
    print("STANDARD_LONG_DONE", flush=True)
