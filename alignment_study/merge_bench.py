# PREREGISTRATION20: a benchmark of head-merging criteria against measured
# merge damage, on multi-head-attention decoders. The merge is the one used to
# convert multi-head to grouped-query attention (Ainslie et al. 2023): two
# heads' key projections (weights and bias) are replaced by their mean, and
# likewise their value projections; queries and outputs stay separate. Damage:
# KL of the next-token distribution on held-out windows (rounds 11 to 16's
# measure). Every pair of heads in every layer is merged on its own; then, per
# criterion, one and two disjoint pairs per layer in layers 1 to L-1 are merged
# at once (chosen greedily by the criterion; five random draws as baseline).
#
# Criteria per pair, on selection windows (rows >= 16 where rows are used):
#   corr_last  Pearson correlation of the last attention rows (CHAI, last-row reading)
#   corr_rows  per-row Pearson over the causal support, mean over rows (CHAI, every row)
#   js         Jensen-Shannon divergence between attention rows (Clark et al. 2019)
#   cos_map    cosine of the flattened maps
#   qk_wcos    cosine of the heads' concatenated query and key weights
#   kv_wcos    cosine of the heads' concatenated key and value weights
#   k_act, v_act  mean cosine of the heads' key (value) vectors over tokens
#   out_pert   the merge simulated on the selection windows: mean norm of the
#              change in the two heads' summed contribution to the residual stream
# Similarities are oriented so that larger means "predicted cheaper".
#   python alignment_study/merge_bench.py [--dev] --registered=<freeze permalink>
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, "audits/readings")
import functional_common as F
import merge_dev_calibration as M
from readings_common import js_matrix, row_pearson
from scale_round_v2 import DEV
from hubsfree.adapters import release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM, AttentionInterface

DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION20.md>; use --dev")
MODELS = ["gpt2"] if DEV_RUN else ["gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
OUT = "alignment_study/merge_bench_dev" if DEV_RUN else "alignment_study/merge_bench"
N_SEL, T_SEL, N_EVAL, T_EVAL, ROW0, N_RANDOM = 8, 256, 4, 512, 16, 5
CRITERIA = ("out_pert", "corr_last", "corr_rows", "js", "cos_map", "qk_wcos", "kv_wcos", "k_act", "v_act")
SIMILARITY = {"corr_last", "corr_rows", "cos_map", "qk_wcos", "kv_wcos", "k_act", "v_act"}
torch.set_grad_enabled(False)
CAP = {}


def capture_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Eager attention that records post-rotary queries, keys, values and weights (own causal mask)."""
    q, k, v = query.float(), key.float(), value.float()
    T = q.shape[-2]
    s = (q @ k.transpose(-1, -2)) * (scaling if scaling is not None else q.shape[-1] ** -0.5)
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    w = torch.softmax(s, -1)
    CAP[module.layer_idx] = (q[0], k[0], v[0], w[0])
    return (w @ v).to(value.dtype).transpose(1, 2).contiguous(), w


AttentionInterface.register("capture", capture_attention)


def arch(model):
    if hasattr(model, "transformer"):
        return "gpt2", model.transformer.h, lambda b: b.attn.c_attn, lambda b: b.attn.c_proj
    return "neox", model.gpt_neox.layers, lambda b: b.attention.query_key_value, lambda b: b.attention.dense


def kv_index(kind, h, hd, d):
    """Indices of head h's key and value features in the fused projection output."""
    if kind == "gpt2":
        return np.arange(d + h * hd, d + (h + 1) * hd), np.arange(2 * d + h * hd, 2 * d + (h + 1) * hd), np.arange(h * hd, (h + 1) * hd)
    b = h * 3 * hd
    return np.arange(b + hd, b + 2 * hd), np.arange(b + 2 * hd, b + 3 * hd), np.arange(b, b + hd)


def proj_rows(kind, mod):
    """Fused-projection weight with output features as rows, and its bias."""
    W = mod.weight.T if kind == "gpt2" else mod.weight
    return W, mod.bias


def out_slices(kind, mod, n, hd):
    W = mod.weight.detach().float()
    return [(W[h * hd:(h + 1) * hd, :] if kind == "gpt2" else W[:, h * hd:(h + 1) * hd].T) for h in range(n)]    # (hd, d): o_h -> residual


def merge(kind, mod, i, j, hd, d):
    W, b = proj_rows(kind, mod); saved = []
    for idx_i, idx_j in zip(kv_index(kind, i, hd, d)[:2], kv_index(kind, j, hd, d)[:2]):
        ii, jj = torch.tensor(idx_i, device=W.device), torch.tensor(idx_j, device=W.device)
        saved.append((ii, W[ii].clone(), b[ii].clone() if b is not None else None, jj, W[jj].clone(), b[jj].clone() if b is not None else None))
        m = (W[ii] + W[jj]) / 2; W[ii] = m; W[jj] = m
        if b is not None:
            mb = (b[ii] + b[jj]) / 2; b[ii] = mb; b[jj] = mb
    return saved


def unmerge(kind, mod, saved):
    W, b = proj_rows(kind, mod)
    for ii, wi, bi, jj, wj, bj in saved:
        W[ii] = wi; W[jj] = wj
        if b is not None:
            b[ii] = bi; b[jj] = bj


def criteria(model, sel, L, n, hd, d):
    kind, blocks, qkv, outp = arch(model)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    acc = {l: {p: {c: [] for c in CRITERIA} for p in pairs} for l in range(L)}
    Wo = {l: [w.to(DEV) for w in out_slices(kind, outp(blocks[l]), n, hd)] for l in range(L)}
    wq = {}
    for l in range(L):
        W, _ = proj_rows(kind, qkv(blocks[l])); W = W.detach().float()
        rows = [kv_index(kind, h, hd, d) for h in range(n)]
        wq[l] = {"qk": [torch.cat([W[r[2]].flatten(), W[r[0]].flatten()]) for r in rows], "kv": [torch.cat([W[r[0]].flatten(), W[r[1]].flatten()]) for r in rows]}
    for x in sel:
        CAP.clear(); model(input_ids=x.to(DEV))
        for l in range(L):
            Q, K, V, A = CAP[l]; T = A.shape[-1]
            rows_t = torch.arange(ROW0, T, device=DEV)
            sup = torch.arange(T, device=DEV)[None, :] <= rows_t[:, None]
            Cr, _ = row_pearson(A[:, ROW0:, :], sup)
            Cl = np.nan_to_num(np.corrcoef(A[:, -1, :].cpu().numpy()))
            Xf = A[:, ROW0:, :].reshape(n, -1); Xf = Xf / Xf.norm(dim=1, keepdim=True); Cm = (Xf @ Xf.T).cpu().numpy()
            J = js_matrix(A[:, ROW0:, :])
            Kn = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-12); Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            kc = torch.einsum("htd,gtd->hg", Kn, Kn).cpu().numpy() / T; vc = torch.einsum("htd,gtd->hg", Vn, Vn).cpu().numpy() / T
            mask = ~torch.ones(T, T, dtype=torch.bool, device=DEV).tril()
            O = A @ V                                                   # (n, T, hd)
            for (i, j) in pairs:
                Km, Vm = (K[i] + K[j]) / 2, (V[i] + V[j]) / 2
                delta = 0
                for h in (i, j):
                    s = (Q[h] @ Km.T) * hd ** -0.5
                    Ah = torch.softmax(s.masked_fill(mask, float("-inf")), -1)
                    delta = delta + (Ah @ Vm - O[h]) @ Wo[l][h]
                a = acc[l][(i, j)]
                a["out_pert"].append(float(delta.norm(dim=-1)[ROW0:].mean())); a["corr_last"].append(float(Cl[i, j])); a["corr_rows"].append(float(Cr[i, j]))
                a["js"].append(float(J[i, j])); a["cos_map"].append(float(Cm[i, j])); a["k_act"].append(float(kc[i, j])); a["v_act"].append(float(vc[i, j]))
                cos = lambda u, w: float((u @ w) / (u.norm() * w.norm()))
                a["qk_wcos"].append(cos(wq[l]["qk"][i], wq[l]["qk"][j])); a["kv_wcos"].append(cos(wq[l]["kv"][i], wq[l]["kv"][j]))
    return {l: {f"{i}-{j}": {c: float(np.mean(v)) for c, v in acc[l][(i, j)].items()} for (i, j) in pairs} for l in range(L)}


def predicted_damage(c, v):
    return -v if c in SIMILARITY else v


def choose(stats_l, crit, k, rng=None):
    """k disjoint pairs, lowest predicted damage first (random order when crit is None)."""
    keys = list(stats_l.keys())
    order = rng.permutation(len(keys)) if crit is None else np.argsort([predicted_damage(crit, stats_l[p][crit]) for p in keys])
    used, out = set(), []
    for o in order:
        i, j = map(int, keys[o].split("-"))
        if i in used or j in used:
            continue
        out.append((i, j)); used |= {i, j}
        if len(out) == k:
            break
    return out


def run_model(name, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    M.N_SEL, M.T_SEL, M.N_EVAL, M.T_EVAL = N_SEL, T_SEL, N_EVAL, T_EVAL
    sel, X = M.selection_windows(tok), M.eval_windows(tok)
    cap_model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="capture", dtype=torch.float32).eval().to(DEV)
    eager = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    c = eager.config; L = c.num_hidden_layers; n = c.num_attention_heads; d = c.hidden_size; hd = d // n
    x0 = X[:1].to(DEV)
    equiv = float((cap_model(input_ids=x0).logits - eager(input_ids=x0).logits).abs().max())
    log(f"{name} layers {L} heads {n}; capture vs eager max logit difference {equiv:.2e}")
    stats = criteria(cap_model, sel, L, n, hd, d); del cap_model; release_memory(DEV)
    log("criteria done")
    kind, blocks, qkv, _ = arch(eager)
    base = F.base_cache(eager, X)
    single = {}
    for l in range(L):
        single[str(l)] = {}
        for p in stats[l]:
            i, j = map(int, p.split("-"))
            sv = merge(kind, qkv(blocks[l]), i, j, hd, d)
            kls, dn = F.kl_eval(eager, X, base); single[str(l)][p] = {"kl": float(np.mean(kls)), "dnll": float(np.mean(dn))}
            unmerge(kind, qkv(blocks[l]), sv)
        log(f"  layer {l}: {len(stats[l])} merges")
    multi = {}
    for k in (1, 2):                                                # pairs per layer, layers 1 to L-1 (round 12's budgets)
        rng = np.random.default_rng(0); multi[str(k)] = {}
        for crit in list(CRITERIA) + [f"random{r}" for r in range(N_RANDOM)]:
            saved = []
            for l in range(1, L):
                for (i, j) in choose(stats[l], None if crit.startswith("random") else crit, k, rng):
                    saved.append((l, merge(kind, qkv(blocks[l]), i, j, hd, d)))
            kls, dn = F.kl_eval(eager, X, base); multi[str(k)][crit] = {"kl": float(np.mean(kls)), "dnll": float(np.mean(dn)), "pairs": len(saved)}
            for l, sv in reversed(saved):
                unmerge(kind, qkv(blocks[l]), sv)
    check = float(np.mean(F.kl_eval(eager, X, base)[0]))
    del eager; release_memory(DEV)
    return {"model": name, "layers": L, "heads": n, "head_dim": hd, "capture_equivalence_max_logit_diff": equiv, "restore_check_kl": check,
            "budgets_pairs_per_layer": [1, 2], "stats": {str(l): stats[l] for l in range(L)}, "single": single, "multi": multi,
            "eval_tokens": int(X.numel()), "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    for name in MODELS:
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        res = run_model(name, log); res["registered"] = REGISTERED
        json.dump(res, open(path, "w"), indent=1)
        log(f"MODEL_DONE {name}: one pair per layer " + " ".join(f"{c} {v['kl']:.4f}" for c, v in res["multi"]["1"].items()))
    print("MERGE_BENCH_DONE", flush=True)
