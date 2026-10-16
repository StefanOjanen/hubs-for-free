# PREREGISTRATION24: converting multi-head attention to grouped-query attention
# without training. Each layer's heads are partitioned into groups of g (2 or
# 4, so half or a quarter of the key-value heads remain) and pooled by the mean
# (gqa_common.py). The arms differ in how the groups are chosen and in whether
# the heads of each group are first aligned by function-preserving rotations
# (Jin et al. 2025; arms ending in _align, and jin):
#   contiguous  neighbouring heads (Ainslie et al. 2023)
#   random0     a random partition
#   wse         weight-sharing error of QCQA (Joshi et al. 2024): squared distance of each head's key and
#               value weights to the group mean, minimized
#   act         activation similarity of AsymGQA (Chen et al. 2024): for keys and for values, the mean over
#               token vectors of the best cosine match in the other head, both directions; key plus value
#               similarity, maximized (one grouping serves keys and values; symmetric groups; no
#               accuracy-guided acceptance)
#   jin         Jin et al. (2025), cosine variant: value caches normalized per token, pairwise Procrustes
#               alignment, mean cosine after alignment, summed within groups and maximized; then
#               generalized Procrustes alignment within the chosen groups (keys with rotary-compatible
#               rotations, values freely), folded into the weights, before pooling
#   sim         the merge simulated on calibration tokens: mean norm of the change in the group's summed
#               contribution to the residual stream when its keys and values are pooled, minimized
#               (rounds 20 and 22's output perturbation, extended from pairs to groups)
#   sim_align   sim computed after jin's alignment of the candidate group; the chosen groups are aligned
#               and pooled as in jin
#   contiguous_align, random0_align  the contiguous and random partitions, aligned within groups as in jin
# Search, identical for every scored arm: minimum-cost (maximum-similarity) perfect matching of heads into
# pairs; for g = 4, the same matching of pairs into quadruples by the group score, then head swaps between
# groups while the total improves (at most two passes). Calibration: 8 C4 validation documents of 1,024
# tokens. Evaluation (zeroshot.py): WikiText-2 perplexity and seven zero-shot tasks (development runs:
# the first 500 examples of each task; --check, a code-path test: 2 calibration documents of 512 tokens,
# 200 examples per task and 8 perplexity windows).
#   python alignment_study/gqa_round.py [--dev | --check] --registered=<freeze permalink>
import json
import os
import sys
import time
import itertools
import numpy as np
import networkx as nx
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import zeroshot as Z
from scale_round_v2 import DEV
from hubsfree.adapters import release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV_RUN = "--dev" in sys.argv
CHECK = "--check" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not (DEV_RUN or CHECK) and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION24.md>; use --dev or --check")
if CHECK:
    MODELS, OUT = ["facebook/opt-125m", "JackFram/llama-160m", "microsoft/phi-1_5"], "alignment_study/gqa_check"
elif DEV_RUN:
    MODELS, OUT = ["gpt2-medium", "EleutherAI/pythia-410m"], "alignment_study/gqa_dev"
else:
    MODELS, OUT = ["facebook/opt-1.3b", "EleutherAI/pythia-1.4b", "gpt2-large", "allenai/OLMo-1B-hf", "microsoft/phi-2"], "alignment_study/gqa"
G_SIZES = (2, 4)
ARMS = ("contiguous", "random0", "wse", "act", "sim", "contiguous_align", "random0_align", "jin", "sim_align")
ALIGNED = {"contiguous_align", "random0_align", "jin", "sim_align"}
N_CAL, T_CAL, ROW0, MAX_PASSES = (2, 512, 16, 2) if CHECK else (8, 1024, 16, 2)
MAXIMIZE = {"act", "jin"}
TIMES = {}
torch.set_grad_enabled(False)


class timed:
    """Accumulate wall time per component (reported in the log)."""
    def __init__(self, k):
        self.k = k

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        TIMES[self.k] = TIMES.get(self.k, 0.0) + time.time() - self.t


# ---------------- per-layer data ----------------
def layer_data(arch, l):
    Q = torch.stack([w[0] for w in C.CAP[l]]).float().to(DEV)          # (windows, n, T, hd)
    K = torch.stack([w[1] for w in C.CAP[l]]).float().to(DEV)
    V = torch.stack([w[2] for w in C.CAP[l]]).float().to(DEV)
    sc = C.CAP[l][0][3]
    T = Q.shape[2]
    mask = ~torch.ones(T, T, dtype=torch.bool, device=DEV).tril()
    O = torch.empty_like(V)
    for h in range(arch.n):
        s = (Q[:, h] @ K[:, h].transpose(-1, -2)) * sc
        O[:, h] = torch.softmax(s.masked_fill(mask, float("-inf")), -1) @ V[:, h]
    Wo = torch.stack([arch.out_slice(l, h) for h in range(arch.n)]).to(DEV)    # (n, hd, d)
    flat = lambda X: X.transpose(0, 1).reshape(arch.n, -1, X.shape[-1])         # (n, windows*T, hd)
    Kn, Vn = C.unit_rows(flat(K)), C.unit_rows(flat(V))
    gram = lambda X: torch.einsum("anx,bny->abxy", X, X).cpu().double().numpy()  # (n, n, hd, hd): X_a^T X_b
    wflat = {w: torch.stack([arch.proj(l, h, w)[0][arch.proj(l, h, w)[2]].detach().flatten() for h in range(arch.n)]).cpu().double().numpy()
             for w in ("k", "v")}
    return {"Q": Q, "K": K, "V": V, "O": O, "Wo": Wo, "sc": sc, "mask": mask, "Kn": Kn, "Vn": Vn, "N": Kn.shape[1],
            "CK": gram(Kn), "CV": gram(Vn), "GW": {w: X @ X.T for w, X in wflat.items()}, "E": wflat["k"].shape[1]}


def sim_cost(D, G, rot=None):
    idx = torch.tensor(G, device=DEV)
    Qg, Kg, Vg = D["Q"][:, idx], D["K"][:, idx], D["V"][:, idx]
    if rot is not None:
        Rk = torch.stack([rot[h][0] for h in G])[None]; Sv = torch.stack([rot[h][1] for h in G])[None]
        Qg, Kg, Vg = Qg @ Rk, Kg @ Rk, Vg @ Sv
    Km, Vm = Kg.mean(1, keepdim=True), Vg.mean(1, keepdim=True)
    s = (Qg @ Km.transpose(-1, -2)) * D["sc"]
    On = torch.softmax(s.masked_fill(D["mask"], float("-inf")), -1) @ Vm
    if rot is not None:
        On = On @ Sv.transpose(-1, -2)
    delta = torch.einsum("bgtk,gkd->btd", On - D["O"][:, idx], D["Wo"][idx])
    return float(delta.norm(dim=-1)[:, ROW0:].mean())


def group_rotations(arch, D, G):
    """Jin et al.'s cosine-variant alignment of the heads in G: GPA on per-token normalized caches."""
    blocks = lambda Cb: [[Cb[a, b] for b in G] for a in G]
    keys = C.gpa_gram(blocks(D["CK"]), lambda Cm: C.rope_procrustes_C(Cm, arch.r))
    vals = C.gpa_gram(blocks(D["CV"]), C.procrustes_C)
    t = lambda R: torch.tensor(R, dtype=torch.float32, device=DEV)
    return {h: (t(rk), t(sv)) for h, rk, sv in zip(G, keys, vals)}


def chamfer_cos(X, Y, chunk=2048):
    """AsymGQA similarity of two activation matrices (rows normalized): mean best-match cosine, both ways."""
    a, b = torch.zeros((), device=X.device), torch.full((Y.shape[0],), -2.0, device=X.device)
    for i in range(0, X.shape[0], chunk):
        S = X[i:i + chunk] @ Y.T
        a = a + S.max(1).values.sum(); b = torch.maximum(b, S.max(0).values)
    return 0.5 * (float(a) / X.shape[0] + float(b.mean()))


def wse(D, G):
    """QCQA's weight-sharing error: per head, mean squared entry of (key weight - group mean), plus the
    same for values, summed over the group (from the Gram matrix of the flattened weights)."""
    ix = np.array(G)
    return float(sum(np.trace(Gw[np.ix_(ix, ix)]) - Gw[np.ix_(ix, ix)].sum() / len(G) for Gw in D["GW"].values()) / D["E"])


# ---------------- search ----------------
def matching(m, score, maximize):
    Gr = nx.Graph()
    for i in range(m):
        for j in range(i + 1, m):
            Gr.add_edge(i, j, weight=float(score[i][j]))
    if maximize:
        lo = min(d["weight"] for _, _, d in Gr.edges(data=True))
        for _, _, d in Gr.edges(data=True):
            d["weight"] = d["weight"] - lo + 1.0
        M = nx.max_weight_matching(Gr, maxcardinality=True)
    else:
        M = nx.min_weight_matching(Gr)
    out = sorted(tuple(sorted(e)) for e in M)
    assert len(out) * 2 == m
    return out


def partition(n, g, pair_score, group_score, maximize):
    """Pairs by matching; for g = 4, pairs of pairs by matching on the group score, then swaps."""
    better = (lambda a, b: a > b + 1e-12) if maximize else (lambda a, b: a < b - 1e-12)
    groups = matching(n, pair_score, maximize)
    if g == 2:
        return [list(p) for p in groups]
    P = len(groups)
    cache = {}

    def gs(G):
        k = tuple(sorted(G))
        if k not in cache:
            cache[k] = group_score(list(k))
        return cache[k]
    qs = [[gs(groups[a] + groups[b]) if a != b else 0.0 for b in range(P)] for a in range(P)]
    groups = [list(groups[a] + groups[b]) for a, b in matching(P, qs, maximize)]
    for _ in range(MAX_PASSES):
        improved = False
        for a, b in itertools.combinations(range(len(groups)), 2):
            for x in list(groups[a]):
                for y in list(groups[b]):
                    if x not in groups[a] or y not in groups[b]:
                        continue
                    na = [h for h in groups[a] if h != x] + [y]; nb = [h for h in groups[b] if h != y] + [x]
                    if better(gs(na) + gs(nb), gs(groups[a]) + gs(groups[b])):
                        groups[a], groups[b] = na, nb; improved = True
        if not improved:
            break
    return [sorted(G) for G in groups]


def layer_partitions(arch, l, D, rng_by_seed):
    """{g: {arm: (groups, rotations or None)}} and simulated-cost diagnostics for one layer."""
    n = arch.n
    pairs = list(itertools.combinations(range(n), 2))
    # pairwise scores
    Kn, Vn = D["Kn"], D["Vn"]
    act = np.zeros((n, n)); jin = np.zeros((n, n)); simc = np.zeros((n, n)); sima = np.zeros((n, n)); w = np.zeros((n, n))
    rot_pair = {}
    for i, j in pairs:
        with timed("act"):
            act[i, j] = act[j, i] = chamfer_cos(Kn[i], Kn[j]) + chamfer_cos(Vn[i], Vn[j])
        with timed("jin+wse"):
            R = C.procrustes_C(D["CV"][j, i])
            jin[i, j] = jin[j, i] = float(np.trace(R.T @ D["CV"][j, i]) / D["N"])
            w[i, j] = w[j, i] = wse(D, [i, j])
        with timed("sim pairs"):
            simc[i, j] = simc[j, i] = sim_cost(D, [i, j])
        with timed("align pairs"):
            rot_pair[(i, j)] = group_rotations(arch, D, [i, j])
        with timed("sim_align pairs"):
            sima[i, j] = sima[j, i] = sim_cost(D, [i, j], rot_pair[(i, j)])
    rot_cache = {}

    def rots(G):
        k = tuple(sorted(G))
        if len(k) == 2:
            return rot_pair[k]
        if k not in rot_cache:
            rot_cache[k] = group_rotations(arch, D, list(k))
        return rot_cache[k]
    sumpair = lambda M: (lambda G: float(sum(M[a, b] for a, b in itertools.combinations(G, 2))))
    spec = {"wse": (w, lambda G: wse(D, G)), "act": (act, sumpair(act)), "jin": (jin, sumpair(jin)),
            "sim": (simc, lambda G: sim_cost(D, G)), "sim_align": (sima, lambda G: sim_cost(D, G, rots(G)))}
    out = {}
    for g in G_SIZES:
        out[g] = {}
        out[g]["contiguous"] = ([list(range(s, s + g)) for s in range(0, n, g)], None)
        perm = rng_by_seed[0].permutation(n)
        out[g]["random0"] = ([sorted(int(h) for h in perm[s:s + g]) for s in range(0, n, g)], None)
        for arm, (pm, gsf) in spec.items():
            with timed(f"search {arm} g{g}"):
                groups = partition(n, g, pm, gsf, arm in MAXIMIZE)
            out[g][arm] = (groups, None)
        for arm in ("contiguous", "random0"):
            out[g][arm + "_align"] = (out[g][arm][0], None)
        for arm in ALIGNED:
            out[g][arm] = (out[g][arm][0], {h: rr for G in out[g][arm][0] for h, rr in rots(G).items()})
    # diagnostics: every arm's partition scored by the simulation (with and without alignment)
    with timed("diagnostics"):
        diag = {g: {arm: {"sim": float(sum(sim_cost(D, G) for G in groups)),
                          "sim_align": float(sum(sim_cost(D, G, rots(G)) for G in groups))}
                    for arm, (groups, _) in out[g].items()} for g in G_SIZES}
    return out, diag


# ---------------- conversion ----------------
def convert(arch, parts, g, arm):
    for l in range(arch.L):
        groups, rot = parts[l][g][arm]
        if rot is not None:
            for h in range(arch.n):
                C.apply_rotation(arch, l, h, rot[h][0], rot[h][1])
        for G in groups:
            C.pool_group(arch, l, G)


def set_attn(model, impl):
    try:
        model.set_attn_implementation(impl)
    except Exception:
        model.config._attn_implementation = impl
    assert model.config._attn_implementation == impl, model.config._attn_implementation


def run_model(name, log, partial_path):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="eager").eval().to(DEV)
    arch = C.Arch(model)
    log(f"{name}: kind {arch.kind} layers {arch.L} heads {arch.n} head dim {arch.hd} rotary dims {arch.r}")
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    x0 = cal[:1, :256].to(DEV)
    ref = model(input_ids=x0).logits
    set_attn(model, "gqa_capture"); C.CAP.clear()
    cap_diff = float((model(input_ids=x0).logits - ref).abs().max()); C.CAP.clear()
    for w in range(cal.shape[0]):
        model(input_ids=cal[w:w + 1].to(DEV))
    set_attn(model, "sdpa")
    sdpa_diff = float((model(input_ids=x0).logits - ref).abs().max())
    log(f"  capture vs eager {cap_diff:.2e}, sdpa vs eager {sdpa_diff:.2e}; calibration captured")
    snap = C.snapshot(arch)
    parts, diag = {}, {}
    rngs = [np.random.default_rng(0)]
    for l in range(arch.L):
        with timed("layer data"):
            D = layer_data(arch, l)
        parts[l], diag[l] = layer_partitions(arch, l, D, rngs)
        del D; C.CAP.pop(l, None)
        if l % 4 == 0 or l == arch.L - 1:
            log(f"  layer {l} partitions done; seconds so far: " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1])[:8]))
    release_memory(DEV)
    # function preservation of the alignment (jin's rotations at g = 4, before pooling)
    for l in range(arch.L):
        _, rot = parts[l][4]["jin"]
        for h in range(arch.n):
            C.apply_rotation(arch, l, h, rot[h][0], rot[h][1])
    align_diff = float((model(input_ids=x0).logits - ref).abs().max())
    C.restore(arch, snap)
    log(f"  alignment preserves the function: max logit difference {align_diff:.2e}")
    tasks = Z.load_tasks(subset=200 if CHECK else (500 if DEV_RUN else None))
    reqs = Z.build_requests(tok, tasks)
    X = Z.wikitext2_windows(tok)
    if CHECK:
        X = X[:8]
    res = json.load(open(partial_path)) if os.path.exists(partial_path) else {"evals": {}}
    new_parts = {str(l): {str(g): {a: v[0] for a, v in parts[l][g].items()} for g in G_SIZES} for l in range(arch.L)}
    if res.get("partitions") not in (None, new_parts):
        log("  partitions differ from the partial file's; its cached evaluations are discarded")
        res["evals"] = {k: v for k, v in res["evals"].items() if k == "base"}
    res.update({"model": name, "kind": arch.kind, "layers": arch.L, "heads": arch.n, "head_dim": arch.hd, "rotary_dims": arch.r,
                "capture_vs_eager": cap_diff, "sdpa_vs_eager": sdpa_diff, "alignment_max_logit_diff": align_diff,
                "calibration": {"source": "allenai/c4 en validation shard 0", "windows": N_CAL, "tokens": T_CAL},
                "partitions": new_parts,
                "sim_diagnostics": {str(l): {str(g): diag[l][g] for g in G_SIZES} for l in range(arch.L)}})

    def evaluate(key):
        if key in res["evals"]:
            log(f"  {key}: cached"); return
        t1 = time.time()
        r = Z.evaluate_tasks(model, tok, tasks, reqs, DEV); r["wikitext2_ppl"] = Z.perplexity(model, X, DEV)
        r["seconds"] = round(time.time() - t1, 1); res["evals"][key] = r
        json.dump(res, open(partial_path, "w"))
        log(f"  {key}: ppl {r['wikitext2_ppl']:.2f} avg {r['average']:.4f}")
    evaluate("base")
    arms = ARMS if not CHECK else ("contiguous", "contiguous_align", "jin", "sim", "sim_align")
    for g in G_SIZES:
        for arm in arms:
            convert(arch, parts, g, arm)
            evaluate(f"g{g}/{arm}")
            C.restore(arch, snap)
    r = Z.perplexity(model, X, DEV)
    res["restore_check_ppl"] = r
    res["runtime_s"] = round(time.time() - t0, 1)
    del model; release_memory(DEV)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    for name in MODELS:
        if only and name not in only:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        partial = path + ".partial"
        res = run_model(name, log, partial)
        res["registered"] = REGISTERED; res["dev"] = DEV_RUN; res["check"] = CHECK
        json.dump(res, open(path, "w"), indent=1)
        if os.path.exists(partial):
            os.remove(partial)
        log(f"MODEL_DONE {name}")
    print("GQA_ROUND_DONE", flush=True)
