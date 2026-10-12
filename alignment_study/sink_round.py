# WHAT THE SINK DOES (PREREGISTRATION13.md). Per model: the KL of four edits
# of the first-token sink (sink_common.py), one layer at a time in the layers
# fixed below and in all layers at once, plus each layer's first-token
# attention mass on the evaluation windows. Models load with the "sinkint"
# attention, which equals eager attention when no edit is active; the
# equivalence is checked on one window against an eager load of the same
# model and recorded.
#   python alignment_study/sink_round.py [--dev] [--only=name]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import functional_common as F
import sink_common as K
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV_RUN = "--dev" in sys.argv
PLAN = {
    "Qwen/Qwen2.5-1.5B":         {"layers": list(range(28)), "n_eval": 8},
    "TinyLlama/TinyLlama_v1.1":  {"layers": list(range(22)), "n_eval": 8},
    "mistralai/Mistral-7B-v0.1": {"layers": list(range(0, 32, 4)), "n_eval": 4},
    "Qwen/Qwen2.5-7B":           {"layers": list(range(0, 28, 4)), "n_eval": 4},
}
DEV_PLAN = {"Qwen/Qwen2.5-0.5B": {"layers": list(range(24)), "n_eval": 8}}
OUT = "alignment_study/sink_dev" if DEV_RUN else "alignment_study/sink"
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)


def first_token_mass(model, X):
    out = model(input_ids=X[:1].to(DEV), output_attentions=True)
    return [float(a[0, :, F.ROW0:, 0].float().mean()) for a in out.attentions]


def run_model(name, plan):
    t0 = time.time()
    log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    M.N_EVAL, M.T_EVAL = plan["n_eval"], F.T_EVAL
    X = M.eval_windows(tok)
    eager = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    ref = eager(input_ids=X[:1].to(DEV)).logits.float().cpu()
    mass = first_token_mass(eager, X)
    del eager; release_memory(DEV)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sinkint", dtype=getattr(torch, dt)).eval().to(DEV)
    L = model.config.num_hidden_layers
    K.set_intervention([], None)
    equiv = float((model(input_ids=X[:1].to(DEV)).logits.float().cpu() - ref).abs().max())
    log(f"{name} DTYPE {dt} layers {L}; sinkint vs eager max abs logit difference {equiv:.2e}")
    base = F.base_cache(model, X)
    res = K.run_interventions(model, X, base, plan["layers"], list(range(L)), log=log)
    K.set_intervention([], None)
    check, _ = F.kl_eval(model, X, base)
    del model; release_memory(DEV)
    return {"model": name, "dtype": dt, "device": DEV, "layers": L, "plan": plan, "eval_tokens": int(X.numel()),
            "equivalence_max_abs_logit": equiv, "restore_check_kl": float(np.mean(check)),
            "first_token_mass_by_layer": mass, **res, "runtime_s": round(time.time() - t0, 1)}


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
    print("SINK_ROUND_DONE", flush=True)
