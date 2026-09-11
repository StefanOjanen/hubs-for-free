# SINK-PROFILE ROUND (PREREGISTRATION9.md, plan T4.3 generative model). For
# each layer and window of each model, rebuild every head from its real sink
# column alone (entries in the empirical sink column c and their skew mirror,
# shared part included) plus a random skew remainder with zero entries in
# that column and row, scaled to the remaining generator norm; compare the
# rebuilt layer's per-pair z (against the real layer's plain-surrogate
# reference) and r1 with the real values, layer by layer. The dense rebuild
# (a_h S plus a random remainder orthogonal to S) is computed alongside as
# the comparison the development runs used. Protocol: 12 WikiText windows,
# stride 40, T = 64, 8 plain draws, 6 rebuilds per window. One JSON per
# model as it completes.
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import wikitext_windows, generators, gnorms, coup, r1, shared_mode, sink_column, col_mass, sur_plain, zstats, zmed_of
from scale_round_v2 import DEV
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = ["gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "TinyLlama/TinyLlama_v1.1",
          "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", "microsoft/Phi-3-mini-4k-instruct",
          "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B", "allenai/OLMo-2-1124-7B"]
NWIN, STRIDE, SEQ, REPS, KPLAIN = 12, 40, 64, 6, 8
OUT = "alignment_study/sinkprofile"
torch.set_grad_enabled(False)


def unit(M):
    return M / np.sqrt((M ** 2).sum())


def random_skew_off_column(rg, T, c):
    M = rg.normal(0, 1, (T, T)); K = (M - M.T) / 2.0; K[:, c] = 0.0; K[c, :] = 0.0
    return unit(K)


def random_skew_perp(rg, S, T):
    M = rg.normal(0, 1, (T, T)); K = (M - M.T) / 2.0; K -= (K * S).sum() * S
    return unit(K)


def run_model(name):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, output_attentions=True, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L = model.config.num_hidden_layers
    wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
    rg = np.random.default_rng(0)
    acc = {l: {k: [] for k in ("smass", "r1_real", "z_real", "r1_dense", "z_dense", "r1_profile", "z_profile", "profile_share")} for l in range(L)}
    for ids in wins:
        out = model(**{k: v.to(DEV) for k, v in ids.items()})
        for l in range(L):
            A = out.attentions[l].squeeze(0).float().cpu().numpy().astype(np.float64)
            n, T, _ = A.shape; iu = np.triu_indices(n, 1); c = sink_column(A)
            G = generators(A); gn = gnorms(G); C = coup(G)
            plains = [coup(generators(sur_plain(A, rg))) for _ in range(KPLAIN)]
            zr, mu, sd = zstats(C, plains, iu)
            S = unit(shared_mode(G)[0]); a = np.array([(G[h] * S).sum() for h in range(n)])
            P = np.zeros_like(G); P[:, :, c] = G[:, :, c]; P[:, c, :] = G[:, c, :]
            rem = np.sqrt(np.clip(gn ** 2 - (P ** 2).sum((1, 2)), 0, None))
            acc[l]["smass"].append(col_mass(A, c)); acc[l]["r1_real"].append(r1(C, gn, iu)); acc[l]["z_real"].append(zr)
            acc[l]["profile_share"].append(float(np.mean((P ** 2).sum((1, 2)) / gn ** 2)))
            for _ in range(REPS):
                Gd = np.stack([a[h] * S + np.sqrt(max(gn[h] ** 2 - a[h] ** 2, 0)) * random_skew_perp(rg, S, T) for h in range(n)])
                Gp = np.stack([P[h] + rem[h] * random_skew_off_column(rg, T, c) for h in range(n)])
                for tag, Gt in (("dense", Gd), ("profile", Gp)):
                    Ct = coup(Gt); acc[l][f"r1_{tag}"].append(r1(Ct, gnorms(Gt), iu)); acc[l][f"z_{tag}"].append(zmed_of(Ct, mu, sd, iu))
        del out
    del model; release_memory(DEV)
    rows = []
    for l in range(L):
        d = acc[l]
        rows.append({"layer": l, "smass": round(float(np.mean(d["smass"])), 4), "profile_share": round(float(np.mean(d["profile_share"])), 4),
                     **{k: round(float(np.median(d[k])), 4) for k in ("r1_real", "z_real", "r1_dense", "z_dense", "r1_profile", "z_profile")}})
    return {"model": name, "dtype": dt, "device": DEV, "protocol": {"n_windows": NWIN, "stride": STRIDE, "seq": SEQ, "reps": REPS, "k_plain": KPLAIN},
            "rows": rows, "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name in MODELS:
        short = name.split("/")[-1]; path = f"{OUT}/{short}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        try:
            res = run_model(name); json.dump(res, open(path, "w"), indent=1)
            from scipy.stats import spearmanr
            zs = spearmanr([r["z_real"] for r in res["rows"]], [r["z_profile"] for r in res["rows"]])[0]
            print(f"MODEL_DONE {short} layers={len(res['rows'])} spearman_z_profile={zs:.3f} {res['runtime_s']}s", flush=True)
        except Exception as e:
            import traceback
            print(f"MODEL_FAILED {short}: {type(e).__name__}: {str(e)[:300]}", flush=True); print(traceback.format_exc()[-600:], flush=True)
    print("SINKPROFILE_ROUND_DONE", flush=True)
