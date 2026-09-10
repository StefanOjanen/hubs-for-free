# Platform fidelity check (2026-09-10). Rounds 1 and 2 ran float32 on CPU,
# round 3 ran bfloat16 on a Colab T4. Before any local GPU run, this script
# measures how the study's per-layer statistics move when the same windows
# are pushed through Apple MPS in float32 and in bfloat16, against the CPU
# float32 protocol. Protocol values as in PREREGISTRATION2/3: 12 WikiText
# windows, stride 40, T = 64, empirical sink column with 16 contributing
# rows. Output: platform_fidelity.json (cited in RUNLOG.md and in
# hubsfree/adapters.py).
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
from common import (wikitext_windows, generators, gnorms, coup, r1, shared_mode,
                    sink_column, col_mass, sink_generator)
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, NWIN, STRIDE, SEQ = "Qwen/Qwen2.5-1.5B", 12, 40, 64
KEYS = ("smass", "r1", "sharedE", "cosS")
tok = AutoTokenizer.from_pretrained(NAME)
wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)


def stats(A):
    n, T, _ = A.shape
    iu = np.triu_indices(n, 1)
    G = generators(A); gn = gnorms(G); C = coup(G); s = sink_column(A)
    Smat, a, e, frac = shared_mode(G)
    cosS = abs(np.sum(Smat / np.sqrt((Smat ** 2).sum()) * sink_generator(T, s)))
    return dict(s=s, smass=col_mass(A, s), r1=r1(C, gn, iu), sharedE=frac, cosS=cosS)


def run(device, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        NAME, output_attentions=True, attn_implementation="eager", dtype=dtype).to(device).eval()
    L = model.config.num_hidden_layers
    per = {l: [] for l in range(L)}; times = []
    with torch.no_grad():
        for ids in wins:
            ids = {k: v.to(device) for k, v in ids.items()}
            if device == "mps":
                torch.mps.synchronize()
            t = time.perf_counter(); out = model(**ids)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t)
            for l in range(L):
                per[l].append(stats(out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)))
    med = {l: {k: float(np.median([d[k] for d in per[l]])) for k in KEYS} for l in range(L)}
    sinks = {l: [int(d["s"]) for d in per[l]] for l in range(L)}
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return med, sinks, float(np.median(times[1:]))


ref, sref, tref = run("cpu", torch.float32)
L = len(ref); hs = [l for l in range(L) if ref[l]["smass"] > 0.4]
out = {"model": NAME, "n_windows": NWIN, "stride": STRIDE, "seq": SEQ, "n_layers": L,
       "high_sink_layers": hs, "reference": "cpu float32",
       "ms_per_window": {"cpu float32": round(tref * 1000, 1)}, "runs": {}}
print(f"{NAME}: {L} layers, {len(hs)} high-sink; cpu float32 {tref*1000:.0f} ms/window", flush=True)
for dev, dt in (("mps", torch.float32), ("mps", torch.bfloat16)):
    med, sinks, t = run(dev, dt)
    label = f"{dev} {str(dt)[6:]}"
    mx = {k: max(abs(med[l][k] - ref[l][k]) for l in range(L)) for k in KEYS}
    mxh = {k: max(abs(med[l][k] - ref[l][k]) for l in hs) for k in KEYS}
    mism = int(sum(a != b for l in range(L) for a, b in zip(sinks[l], sref[l])))
    out["ms_per_window"][label] = round(t * 1000, 1)
    out["runs"][label] = {"max_abs_dev_layer_median_all": {k: round(v, 4) for k, v in mx.items()},
                          "max_abs_dev_layer_median_high_sink": {k: round(v, 4) for k, v in mxh.items()},
                          "sink_column_mismatches": mism, "layer_windows": L * NWIN}
    print(label, f"{t*1000:.0f} ms/window;", "max dev all layers", {k: round(v, 4) for k, v in mx.items()},
          "; high-sink", {k: round(v, 4) for k, v in mxh.items()}, f"; sink mismatches {mism}/{L*NWIN}", flush=True)
json.dump(out, open("alignment_study/platform_fidelity.json", "w"), indent=1)
print("FIDELITY_DONE")
