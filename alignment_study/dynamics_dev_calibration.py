# Development calibration for PREREGISTRATION5 (dynamics), run BEFORE the
# preregistration was frozen and on data outside the registered set: Pythia
# models instantiated from their configs with random weights (torch seed 0),
# not any training checkpoint. Purpose: learn what the untrained baseline of
# the registered statistics looks like so the predictions test something
# that is not already implied by initialization. Output:
# dynamics_dev_calibration.json.
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scale_round_v2 import layer_stats, NWIN, STRIDE, SEQ, DEV
from common import wikitext_windows
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

KEYS = ("smass", "sharedE", "cosS", "r1_real", "zmed_real", "rho_real")
out = {"note": "random-init models from config, torch seed 0; not a registered checkpoint",
       "protocol": {"n_windows": NWIN, "stride": STRIDE, "seq": SEQ, "device": DEV}, "models": {}}
for name in ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]:
    cfg = AutoConfig.from_pretrained(name)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval().to(DEV)
    tok = AutoTokenizer.from_pretrained(name)
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    rg = np.random.default_rng(0)
    L = cfg.num_hidden_layers
    acc = {l: [] for l in range(L)}
    with torch.no_grad():
        for ids in wins:
            o = model(**{k: v.to(DEV) for k, v in ids.items()}, output_attentions=True)
            for l in range(L):
                acc[l].append(layer_stats(o.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64), rg))
    rows = []
    for l in range(L):
        d = acc[l]
        rows.append({"layer": l, **{k: round(float(np.mean([x[k] for x in d]) if k in ("smass", "sharedE")
                                                   else np.median([x[k] for x in d])), 4) for k in KEYS}})
    rng = {k: [min(r[k] for r in rows), max(r[k] for r in rows)] for k in KEYS}
    out["models"][name] = {"rows": rows, "range": rng}
    print(name, json.dumps({k: [round(v, 3) for v in rng[k]] for k in KEYS}), flush=True)
    del model
json.dump(out, open("alignment_study/dynamics_dev_calibration.json", "w"), indent=1)
print("CALIBRATION_DONE")
