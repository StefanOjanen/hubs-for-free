# POST-HOC (labeled, not registered): the sink-set version of the derived
# shared energy. PREREGISTRATION8's derivation used the ideal sink generator
# at the modal sink column; its residual was largest where the sink sits on
# a window-varying mid-sequence column (OLMo-2). Here the ideal generator is
# the normalized sum of ideal sink generators over the layer's sink set
# (hubsfree.stats.sink_columns: modal column plus columns whose layer-level
# mass reaches 0.10, at most three), and the derived per-head fraction is
# <G_h, S_set>^2 / ||G_h||^2. Same windows and protocol as derivation_dev.py.
# Output: derivation_sinkset_posthoc.json with, per model, the median gap
# (measured minus derived) for the single-column and sink-set versions.
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, gnorms, shared_mode, sink_column, col_mass
from hubsfree.stats import sink_columns, sink_set_generator
from hubsfree.adapters import pick_device, pick_dtype, release_memory
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr, spearmanr

MODELS = ["gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "TinyLlama/TinyLlama_v1.1",
          "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", "microsoft/Phi-3-mini-4k-instruct",
          "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B", "allenai/OLMo-2-1124-7B"]
NWIN, STRIDE, SEQ = 12, 40, 64
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)


def ideal(T, c):
    return sink_set_generator(T, [c])


def per_layer(A):
    n, T, _ = A.shape
    c = sink_column(A); cols = sink_columns(A)
    G = generators(A); g2 = gnorms(G) ** 2; V = G.reshape(n, -1)
    e_single = (V @ ideal(T, c).ravel()) ** 2; e_set = (V @ sink_set_generator(T, cols).ravel()) ** 2
    return {"smass": col_mass(A, c), "k": len(cols), "measured": float(shared_mode(G)[3]),
            "derived_single": float((e_single / g2).mean()), "derived_set": float((e_set / g2).mean())}


out = {"note": "post hoc, not registered", "models": {}}
for name in MODELS:
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L = model.config.num_hidden_layers
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    acc = {l: [] for l in range(L)}
    for ids in wins:
        o = model(**{k: v.to(DEV) for k, v in ids.items()})
        for l in range(L):
            acc[l].append(per_layer(o.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)))
        del o
    del model; release_memory(DEV)
    rows = [{"layer": l, **{k: float(np.mean([d[k] for d in acc[l]])) for k in ("smass", "k", "measured", "derived_single", "derived_set")}} for l in range(L)]
    hs = [r for r in rows if r["smass"] > 0.4]
    m = np.array([r["measured"] for r in hs]); ps = np.array([r["derived_single"] for r in hs]); pt = np.array([r["derived_set"] for r in hs])
    summ = {"n_high_sink": len(hs), "median_gap_single": float(np.median(m - ps)), "median_gap_set": float(np.median(m - pt)),
            "r2_single": float(pearsonr(ps, m)[0] ** 2) if len(hs) > 2 else None, "r2_set": float(pearsonr(pt, m)[0] ** 2) if len(hs) > 2 else None,
            "layers_with_k_above_1": int(sum(r["k"] > 1.0 for r in hs)), "frac_set_below_measured": float(np.mean(pt <= m)) if len(hs) else None}
    out["models"][name] = {"rows": rows, "summary": summ}
    print(f"{name.split('/')[-1]:24s} hs {len(hs):2d} | median gap single {summ['median_gap_single']:.3f} -> set {summ['median_gap_set']:.3f} | R2 single {summ['r2_single'] if summ['r2_single'] is None else round(summ['r2_single'],2)} -> set {summ['r2_set'] if summ['r2_set'] is None else round(summ['r2_set'],2)} | layers with set>1 col {summ['layers_with_k_above_1']}", flush=True)
    json.dump(out, open("alignment_study/derivation_sinkset_posthoc.json", "w"), indent=1)
allm, alls, allt = [], [], []
for v in out["models"].values():
    for r in v["rows"]:
        if r["smass"] > 0.4: allm.append(r["measured"]); alls.append(r["derived_single"]); allt.append(r["derived_set"])
out["pooled"] = {"n": len(allm), "r2_single": float(pearsonr(alls, allm)[0] ** 2), "r2_set": float(pearsonr(allt, allm)[0] ** 2),
                 "median_gap_single": float(np.median(np.array(allm) - np.array(alls))), "median_gap_set": float(np.median(np.array(allm) - np.array(allt))),
                 "spearman_single": float(spearmanr(alls, allm)[0]), "spearman_set": float(spearmanr(allt, allm)[0])}
json.dump(out, open("alignment_study/derivation_sinkset_posthoc.json", "w"), indent=1)
print("POOLED", json.dumps(out["pooled"])); print("SINKSET_DONE")
