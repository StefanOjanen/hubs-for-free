# DYNAMICS ROUND (PREREGISTRATION5.md). Pythia-160m and Pythia-410m at ten
# training checkpoints, the registered protocol of rounds 2 and 3 (12
# WikiText windows, stride 40, T = 64, 12 plain / 8 sinkfix / 8 altsink-v2
# draws) and the registered statistics implementation
# (scale_round_v2.layer_stats). One JSON per (model, checkpoint) is written
# as soon as it completes, so an interrupted run resumes from the missing
# cells. Runs on the device selected by hubsfree.adapters (MPS float32 here).
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scale_round_v2 import layer_stats, NWIN, STRIDE, SEQ, KPLAIN, KFAM, DEV
from common import wikitext_windows
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
OUT = "alignment_study/dynamics"
torch.set_grad_enabled(False)


def run_cell(name, step):
    rev = f"step{step}"
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    dt = pick_dtype(name, DEV, "auto", revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=rev, output_attentions=True, attn_implementation="eager",
        dtype=getattr(torch, dt)).eval().to(DEV)
    L = model.config.num_hidden_layers
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    rg = np.random.default_rng(0)
    acc = {l: [] for l in range(L)}
    t0 = time.time()
    for ids in wins:
        out = model(**{k: v.to(DEV) for k, v in ids.items()})
        for l in range(L):
            A = out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)
            acc[l].append(layer_stats(A, rg))
        del out
    rows = []
    for l in range(L):
        ds = acc[l]
        agg = {"layer": l}
        for k in ds[0]:
            vals = [d[k] for d in ds]
            agg[k] = round(float(np.mean(vals)) if k in ("smass", "sharedE", "inv_dev")
                           else float(np.median(vals)), 6 if k == "inv_dev" else 4)
        rows.append(agg)
    del model
    release_memory(DEV)
    return {"model": name, "step": step, "revision": rev, "dtype": dt, "device": DEV,
            "n_windows": NWIN, "stride": STRIDE, "seq": SEQ, "k_plain": KPLAIN, "k_fam": KFAM,
            "runtime_s": round(time.time() - t0, 1), "rows": rows}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name in MODELS:
        short = name.split("/")[-1]
        for step in STEPS:
            path = f"{OUT}/{short}_step{step}.json"
            if os.path.exists(path):
                print("SKIP (exists)", path, flush=True)
                continue
            try:
                cell = run_cell(name, step)
                json.dump(cell, open(path, "w"), indent=1)
                hs = sum(r["smass"] > 0.4 for r in cell["rows"])
                print(f"CELL_DONE {short} step{step} layers={len(cell['rows'])} high_sink={hs} "
                      f"dtype={cell['dtype']} {cell['runtime_s']}s", flush=True)
            except Exception as e:
                print(f"CELL_FAILED {short} step{step}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    print("DYNAMICS_DONE", flush=True)
