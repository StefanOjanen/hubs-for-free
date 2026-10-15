# PREREGISTRATION22: the merge benchmark of round 20 at larger scale and with a
# downstream task. Same merge (two heads' key and value projections replaced by
# their mean), same criteria, same KL damage, same selection and evaluation
# windows (merge_bench.py, frozen at 283e5b5); this file adds OPT (separate
# query, key and value projections, queries scaled before the attention call)
# and a downstream measure.
#
# Part S: every pair of heads in every layer of EleutherAI/pythia-1.4b,
# facebook/opt-1.3b and gpt2-large, merged on its own; the nine criteria; one
# and two pairs per layer in layers 1 to L-1 chosen greedily by each criterion.
# Part D: LAMBADA (OpenAI version, first 1,000 test examples) last-word
# accuracy, greedy over all continuation tokens, for the unmerged model and for
# the merges chosen at one and two pairs per layer by the output perturbation,
# the last-row correlation (CHAI-style), the Jensen-Shannon divergence, and
# three random draws; on round 20's three models and the three above.
#   python alignment_study/merge_bench2.py S|D [--dev] --registered=<freeze permalink>
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, "audits/readings")
_argv = sys.argv; sys.argv = [sys.argv[0], "--dev"]        # merge_bench's import guard
import merge_bench as MB
sys.argv = _argv
import functional_common as F
import merge_dev_calibration as M
from readings_common import js_matrix, row_pearson
from scale_round_v2 import DEV
from hubsfree.adapters import release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM, AttentionInterface

PART = sys.argv[1] if len(sys.argv) > 1 else ""
DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION22.md>; use --dev")
OUT = "alignment_study/merge_bench2_dev" if DEV_RUN else "alignment_study/merge_bench2"
S_MODELS = ["facebook/opt-125m"] if DEV_RUN else ["EleutherAI/pythia-1.4b", "facebook/opt-1.3b", "gpt2-large"]
D_MODELS = ["gpt2", "facebook/opt-125m"] if DEV_RUN else ["gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
                                                          "EleutherAI/pythia-1.4b", "facebook/opt-1.3b", "gpt2-large"]
D_CRIT = ("out_pert", "corr_last", "js")
N_LAMBADA, N_RANDOM_D = (200 if DEV_RUN else 1000), 3
CAP = {}
torch.set_grad_enabled(False)


def capture_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    q, k, v = query.float(), key.float(), value.float()
    sc = scaling if scaling is not None else q.shape[-1] ** -0.5
    T = q.shape[-2]
    s = (q @ k.transpose(-1, -2)) * sc
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    w = torch.softmax(s, -1)
    CAP[module.layer_idx] = (q[0], k[0], v[0], w[0], sc)
    return (w @ v).to(value.dtype).transpose(1, 2).contiguous(), w


AttentionInterface.register("capture2", capture_attention)


# ---------------- architectures ----------------
def arch(model):
    if hasattr(model, "transformer"):
        return "gpt2", model.transformer.h
    if hasattr(model, "gpt_neox"):
        return "neox", model.gpt_neox.layers
    return "opt", model.model.decoder.layers


def kv_targets(kind, block, h, hd, d):
    """[(weight with output features as rows, bias, row indices)] for head h's key and value."""
    if kind == "opt":
        a = block.self_attn; r = torch.arange(h * hd, (h + 1) * hd, device=a.k_proj.weight.device)
        return [(a.k_proj.weight, a.k_proj.bias, r), (a.v_proj.weight, a.v_proj.bias, r)]
    mod = block.attn.c_attn if kind == "gpt2" else block.attention.query_key_value
    W = mod.weight.T if kind == "gpt2" else mod.weight
    ki, vi, _ = MB.kv_index(kind, h, hd, d)
    return [(W, mod.bias, torch.tensor(ki, device=W.device)), (W, mod.bias, torch.tensor(vi, device=W.device))]


def q_rows(kind, block, h, hd, d):
    if kind == "opt":
        W = block.self_attn.q_proj.weight; return W[h * hd:(h + 1) * hd]
    mod = block.attn.c_attn if kind == "gpt2" else block.attention.query_key_value
    W = mod.weight.T if kind == "gpt2" else mod.weight
    return W[torch.tensor(MB.kv_index(kind, h, hd, d)[2], device=W.device)]


def out_slice(kind, block, h, hd):
    if kind == "opt":
        return block.self_attn.out_proj.weight.detach().float()[:, h * hd:(h + 1) * hd].T
    if kind == "gpt2":
        return block.attn.c_proj.weight.detach().float()[h * hd:(h + 1) * hd, :]
    return block.attention.dense.weight.detach().float()[:, h * hd:(h + 1) * hd].T


def merge(kind, block, i, j, hd, d):
    saved = []
    for (W, b, ii), (_, _, jj) in zip(kv_targets(kind, block, i, hd, d), kv_targets(kind, block, j, hd, d)):
        saved.append((W, b, ii, W[ii].clone(), b[ii].clone() if b is not None else None, jj, W[jj].clone(), b[jj].clone() if b is not None else None))
        m = (W[ii] + W[jj]) / 2; W[ii] = m; W[jj] = m
        if b is not None:
            mb = (b[ii] + b[jj]) / 2; b[ii] = mb; b[jj] = mb
    return saved


def unmerge(saved):
    for W, b, ii, wi, bi, jj, wj, bj in saved:
        W[ii] = wi; W[jj] = wj
        if b is not None:
            b[ii] = bi; b[jj] = bj


def geometry(model):
    c = model.config
    L = getattr(c, "num_hidden_layers", None) or c.n_layer; n = getattr(c, "num_attention_heads", None) or c.n_head
    d = getattr(c, "hidden_size", None) or c.n_embd
    return L, n, d, d // n


def criteria(model, sel, L, n, hd, d):
    kind, blocks = arch(model)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    acc = {l: {p: {c: [] for c in MB.CRITERIA} for p in pairs} for l in range(L)}
    Wo = {l: [out_slice(kind, blocks[l], h, hd).to(DEV) for h in range(n)] for l in range(L)}
    wv = {}
    for l in range(L):
        qk, kv = [], []
        for h in range(n):
            (Wk, _, ki), (Wv, _, vi) = kv_targets(kind, blocks[l], h, hd, d)
            qr = q_rows(kind, blocks[l], h, hd, d).detach().float().flatten()
            kr, vr = Wk[ki].detach().float().flatten(), Wv[vi].detach().float().flatten()
            qk.append(torch.cat([qr, kr])); kv.append(torch.cat([kr, vr]))
        wv[l] = (qk, kv)
    cos = lambda u, w: float((u @ w) / (u.norm() * w.norm()))
    for x in sel:
        CAP.clear(); model(input_ids=x.to(DEV))
        for l in range(L):
            Q, K, V, A, sc = CAP[l]; T = A.shape[-1]; R0 = MB.ROW0
            rows_t = torch.arange(R0, T, device=DEV); sup = torch.arange(T, device=DEV)[None, :] <= rows_t[:, None]
            Cr, _ = row_pearson(A[:, R0:, :], sup)
            Cl = np.nan_to_num(np.corrcoef(A[:, -1, :].cpu().numpy()))
            Xf = A[:, R0:, :].reshape(n, -1); Xf = Xf / Xf.norm(dim=1, keepdim=True); Cm = (Xf @ Xf.T).cpu().numpy()
            J = js_matrix(A[:, R0:, :])
            Kn = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-12); Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            kc = torch.einsum("htd,gtd->hg", Kn, Kn).cpu().numpy() / T; vc = torch.einsum("htd,gtd->hg", Vn, Vn).cpu().numpy() / T
            mask = ~torch.ones(T, T, dtype=torch.bool, device=DEV).tril()
            O = A @ V
            for (i, j) in pairs:
                Km, Vm = (K[i] + K[j]) / 2, (V[i] + V[j]) / 2
                delta = 0
                for h in (i, j):
                    Ah = torch.softmax(((Q[h] @ Km.T) * sc).masked_fill(mask, float("-inf")), -1)
                    delta = delta + (Ah @ Vm - O[h]) @ Wo[l][h]
                a = acc[l][(i, j)]
                a["out_pert"].append(float(delta.norm(dim=-1)[R0:].mean())); a["corr_last"].append(float(Cl[i, j])); a["corr_rows"].append(float(Cr[i, j]))
                a["js"].append(float(J[i, j])); a["cos_map"].append(float(Cm[i, j])); a["k_act"].append(float(kc[i, j])); a["v_act"].append(float(vc[i, j]))
                a["qk_wcos"].append(cos(wv[l][0][i], wv[l][0][j])); a["kv_wcos"].append(cos(wv[l][1][i], wv[l][1][j]))
    return {l: {f"{i}-{j}": {c: float(np.mean(v)) for c, v in acc[l][(i, j)].items()} for (i, j) in pairs} for l in range(L)}


def part_S(name, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    M.N_SEL, M.T_SEL, M.N_EVAL, M.T_EVAL = MB.N_SEL, MB.T_SEL, MB.N_EVAL, MB.T_EVAL
    sel, X = M.selection_windows(tok), M.eval_windows(tok)
    cap = AutoModelForCausalLM.from_pretrained(name, attn_implementation="capture2", dtype=torch.float32).eval().to(DEV)
    eager = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    L, n, d, hd = geometry(eager)
    equiv = float((cap(input_ids=X[:1].to(DEV)).logits - eager(input_ids=X[:1].to(DEV)).logits).abs().max())
    log(f"{name} layers {L} heads {n}; capture vs eager max logit difference {equiv:.2e}")
    stats = criteria(cap, sel, L, n, hd, d); del cap; release_memory(DEV)
    log("criteria done")
    kind, blocks = arch(eager); base = F.base_cache(eager, X); single = {}
    for l in range(L):
        single[str(l)] = {}
        for p in stats[l]:
            i, j = map(int, p.split("-")); sv = merge(kind, blocks[l], i, j, hd, d)
            kls, dn = F.kl_eval(eager, X, base); single[str(l)][p] = {"kl": float(np.mean(kls)), "dnll": float(np.mean(dn))}
            unmerge(sv)
        log(f"  layer {l}: {len(stats[l])} merges")
    multi = {}
    for k in (1, 2):
        rng = np.random.default_rng(0); multi[str(k)] = {}
        for crit in list(MB.CRITERIA) + [f"random{r}" for r in range(MB.N_RANDOM)]:
            saved = []
            for l in range(1, L):
                for (i, j) in MB.choose(stats[l], None if crit.startswith("random") else crit, k, rng):
                    saved.append(merge(kind, blocks[l], i, j, hd, d))
            kls, dn = F.kl_eval(eager, X, base); multi[str(k)][crit] = {"kl": float(np.mean(kls)), "dnll": float(np.mean(dn)), "pairs": len(saved)}
            for sv in reversed(saved):
                unmerge(sv)
    check = float(np.mean(F.kl_eval(eager, X, base)[0])); del eager; release_memory(DEV)
    return {"model": name, "layers": L, "heads": n, "head_dim": hd, "capture_equivalence_max_logit_diff": equiv, "restore_check_kl": check,
            "budgets_pairs_per_layer": [1, 2], "stats": {str(l): stats[l] for l in range(L)}, "single": single, "multi": multi,
            "eval_tokens": int(X.numel()), "runtime_s": round(time.time() - t0, 1)}


# ---------------- Part D: LAMBADA ----------------
def lambada(tok, n):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset")
    out = []
    for line in open(p):
        t = json.loads(line)["text"]; ctx, last = t.rsplit(" ", 1)
        c = tok(ctx)["input_ids"]; g = tok(" " + last, add_special_tokens=False)["input_ids"]
        out.append((c, g))
        if len(out) == n:
            break
    return out


def lambada_acc(model, data, pad_id, bs=16):
    hits = 0
    for b in range(0, len(data), bs):
        chunk = data[b:b + bs]; seqs = [c + g for c, g in chunk]; Lm = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), Lm), pad_id, dtype=torch.long); att = torch.zeros(len(seqs), Lm, dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
        pred = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits.argmax(-1).cpu()
        for r, (c, g) in enumerate(chunk):
            hits += int(all(int(pred[r, len(c) - 1 + t]) == g[t] for t in range(len(g))))
    return hits / len(data)


def part_D(name, stats_path, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data = lambada(tok, N_LAMBADA)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    L, n, d, hd = geometry(model); kind, blocks = arch(model)
    st = json.load(open(stats_path)); stats = {int(l): v for l, v in st["stats"].items()}
    res = {"model": name, "n_examples": len(data), "base": lambada_acc(model, data, pad), "multi": {}}
    for k in (1, 2):
        rng = np.random.default_rng(0); res["multi"][str(k)] = {}
        for crit in list(D_CRIT) + [f"random{r}" for r in range(N_RANDOM_D)]:
            saved = []
            for l in range(1, L):
                for (i, j) in MB.choose(stats[l], None if crit.startswith("random") else crit, k, rng):
                    saved.append(merge(kind, blocks[l], i, j, hd, d))
            res["multi"][str(k)][crit] = {"acc": lambada_acc(model, data, pad), "kl": st["multi"][str(k)][crit]["kl"] if crit in st["multi"][str(k)] else None}
            for sv in reversed(saved):
                unmerge(sv)
        log(f"  {name} budget {k}: " + " ".join(f"{c} {v['acc']:.3f}" for c, v in res["multi"][str(k)].items()))
    res["restore_check_acc"] = lambada_acc(model, data, pad)
    del model; release_memory(DEV)
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


STATS_FOR = {"gpt2-medium": "alignment_study/merge_bench/gpt2-medium.json", "EleutherAI/pythia-160m": "alignment_study/merge_bench/pythia-160m.json",
             "EleutherAI/pythia-410m": "alignment_study/merge_bench/pythia-410m.json"}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    if PART == "S":
        for name in S_MODELS:
            path = f"{OUT}/S_{name.split('/')[-1]}.json"
            if os.path.exists(path):
                print("SKIP (exists)", path, flush=True); continue
            res = part_S(name, log); res["registered"] = REGISTERED
            json.dump(res, open(path, "w"), indent=1)
            log(f"MODEL_DONE {name}: one pair per layer " + " ".join(f"{c} {v['kl']:.4f}" for c, v in res["multi"]["1"].items()))
    else:
        for name in D_MODELS:
            path = f"{OUT}/D_{name.split('/')[-1]}.json"
            if os.path.exists(path):
                print("SKIP (exists)", path, flush=True); continue
            sp = STATS_FOR.get(name, f"{OUT}/S_{name.split('/')[-1]}.json")
            if DEV_RUN and name == "gpt2":
                sp = "alignment_study/merge_bench_dev/gpt2.json"
            res = part_D(name, sp, log); res["registered"] = REGISTERED
            json.dump(res, open(path, "w"), indent=1)
            log(f"MODEL_DONE {name}: base {res['base']:.3f}")
    print("MERGE_BENCH2_DONE", PART, flush=True)
