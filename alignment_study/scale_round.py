# SCALE ROUND (PREREGISTRATION3.md, frozen and pushed before execution).
# Designed for a Colab GPU runtime:
#   !git clone -q https://github.com/StefanOjanen/hubs-for-free r
#   %cd r
#   !pip -q install -U transformers datasets accelerate
#   !python alignment_study/scale_round.py
# Forward passes in bfloat16 on GPU; all statistics in float64 on the
# attention probabilities. Per-model try/except: a model that cannot run
# on the available runtime is recorded untestable, per the prereg.
import gc
import json
import sys
import traceback
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
from common import (wikitext_windows, generators, gnorms, coup, r1, rho,
                    shared_mode, sink_column, col_mass, sink_generator,
                    sur_plain, sur_sinkfix, sur_altsink2, zstats, zmed_of)
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = ["Qwen/Qwen2.5-3B", "microsoft/Phi-3-mini-4k-instruct",
          "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B",
          "allenai/OLMo-2-1124-7B"]
NWIN, STRIDE, SEQ, KPLAIN, KFAM = 12, 40, 64, 12, 8
CUDA = torch.cuda.is_available()
torch.set_grad_enabled(False)
print("DEVICE", "cuda:" + torch.cuda.get_device_name(0) if CUDA else "cpu",
      flush=True)


def layer_stats(A, rg):
    n, T, _ = A.shape
    iu = np.triu_indices(n, 1)
    G = generators(A)
    gn = gnorms(G)
    C = coup(G)
    s = sink_column(A)
    Smat, a, e, frac = shared_mode(G)
    d = {"s": s, "smass": col_mass(A, s), "r1_real": r1(C, gn, iu),
         "rho_real": rho(G, gn), "sharedE": frac,
         "cv_gn": float(gn.std() / gn.mean()),
         "cosS": float(abs(np.sum(Smat / np.sqrt((Smat ** 2).sum())
                                  * sink_generator(T, s))))}
    plains, r1p, rhop = [], [], []
    for _ in range(KPLAIN):
        Gp = generators(sur_plain(A, rg))
        Cp = coup(Gp)
        plains.append(Cp)
        r1p.append(r1(Cp, gnorms(Gp), iu))
        rhop.append(rho(Gp, gnorms(Gp)))
    zr, mu, sd = zstats(C, plains, iu)
    d.update(zmed_real=zr, r1_plain=float(np.median(r1p)),
             rho_plain=float(np.median(rhop)),
             zmed_plain=float(np.median([zmed_of(Cp, mu, sd, iu)
                                         for Cp in plains])))
    zsf, r1sf, za2, r1a2, rhoa2 = [], [], [], [], []
    for _ in range(KFAM):
        Gf = generators(sur_sinkfix(A, rg, s=s))
        Cf = coup(Gf)
        zsf.append(zmed_of(Cf, mu, sd, iu))
        r1sf.append(r1(Cf, gnorms(Gf), iu))
        Ga = generators(sur_altsink2(A, rg))
        Ca = coup(Ga)
        za2.append(zmed_of(Ca, mu, sd, iu))
        r1a2.append(r1(Ca, gnorms(Ga), iu))
        rhoa2.append(rho(Ga, gnorms(Ga)))
    d.update(zmed_sinkfix=float(np.median(zsf)),
             r1_sinkfix=float(np.median(r1sf)),
             zmed_alt2=float(np.median(za2)),
             r1_alt2=float(np.median(r1a2)),
             rho_alt2=float(np.median(rhoa2)))
    return d


def run_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, output_attentions=True, attn_implementation="eager",
        dtype=torch.bfloat16 if CUDA else torch.float32,
        device_map="auto" if CUDA else None).eval()
    L = model.config.num_hidden_layers
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    rg = np.random.default_rng(0)
    acc = {l: [] for l in range(L)}
    for w, ids in enumerate(wins):
        ids = {k: v.to(model.device) for k, v in ids.items()}
        out = model(**ids)
        for l in range(L):
            A = (out.attentions[l].squeeze(0).float().cpu().numpy()
                 .astype(np.float64))
            acc[l].append(layer_stats(A, rg))
        del out
        print(name, "WIN", w, flush=True)
    rows = []
    for l in range(L):
        ds = acc[l]
        agg = {"layer": l}
        for k in ds[0]:
            vals = [d[k] for d in ds]
            agg[k] = round(float(np.mean(vals)) if k in ("smass", "sharedE")
                           else float(np.median(vals)), 4)
        rows.append(agg)
    del model
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
    return rows


res = {"models": {}, "S": {}}
for name in MODELS:
    try:
        rows = run_model(name)
        res["models"][name] = rows
        print("MODEL_RESULT_START " + name)
        print(json.dumps(rows))
        print("MODEL_RESULT_END", flush=True)
    except Exception:
        res["models"][name] = {"error": traceback.format_exc()[-400:]}
        print("MODEL_FAILED", name, flush=True)

def hs(rows):
    return [r_ for r_ in rows if r_["smass"] > 0.4]

pooled_hs = []
per_model = {}
for name, rows in res["models"].items():
    if isinstance(rows, dict):
        per_model[name] = {"testable": False}
        continue
    H = hs(rows)
    pooled_hs += H
    testable = len(H) >= 3
    def frac(cond):
        return float(np.mean([1.0 if cond(r_) else 0.0 for r_ in H])) if H else 0.0
    def ratios(r_):
        Rr = abs(r_["r1_sinkfix"] - r_["r1_real"]) / max(0.05, abs(r_["r1_plain"] - r_["r1_real"]))
        Rz = abs(r_["zmed_sinkfix"] - r_["zmed_real"]) / max(1.0, abs(r_["zmed_real"] - r_["zmed_plain"]))
        return Rr, Rz
    per_model[name] = {
        "testable": testable, "n_high_sink": len(H),
        "S1_frac": frac(lambda r_: r_["rho_real"] >= 2 * r_["rho_plain"] and r_["cosS"] > 0.7),
        "S2_frac": frac(lambda r_: max(ratios(r_)) < 0.4),
        "S3_frac": frac(lambda r_: r_["r1_alt2"] > 0.8 and r_["rho_alt2"] < 0.5 * r_["rho_real"]
                        and r_["zmed_alt2"] > r_["zmed_real"]),
    }
    for k in ("S1", "S2", "S3"):
        per_model[name][k + "_pass"] = bool(testable and per_model[name][k + "_frac"] >= 2 / 3)

test_models = [m for m, v in per_model.items() if v.get("testable")]
for k in ("S1", "S2", "S3"):
    passes = sum(1 for m in test_models if per_model[m][k + "_pass"])
    res["S"][k] = {"passes": passes, "testable": len(test_models),
                   "pass": bool(len(test_models) and passes >= (2 / 3) * len(test_models))}
if len(pooled_hs) >= 8:
    def rank(v):
        v = np.array(v); idx = np.argsort(v)
        rk = np.empty(len(v)); rk[idx] = np.arange(len(v))
        return rk
    sp = float(np.corrcoef(rank([r_["sharedE"] for r_ in pooled_hs]),
                           rank([r_["r1_real"] for r_ in pooled_hs]))[0, 1])
    res["S"]["S4"] = {"spearman": sp, "pass": bool(sp <= -0.5)}
else:
    res["S"]["S4"] = {"untestable": True}
res["S"]["per_model"] = per_model
json.dump(res, open("alignment_study/scale_round.json", "w"), indent=1)
print("RESULTS_JSON_START")
print(json.dumps(res["S"]))
print("RESULTS_JSON_END", flush=True)
