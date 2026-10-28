# PREREGISTRATION28: the weighted fusion of rounds 26 and 27 against CAGA (Wang et al., CHASE, arXiv 2610.09476)
# on CAGA's own benchmark: Llama-2-7B converted without training from 32 to 16 key-value heads (pairs of heads),
# where the paper's Table 2 reports WikiText-2 perplexity 25.1 and a seven-task zero-shot average of 44.3 for
# CAGA (926 and 32.6 for direct averaging, 738 and 35.3 for Procrustes alignment; all on CAGA's head pairing).
# Constructions of each pair's shared key and value head:
#   mean   the key and value projections averaged (Ainslie et al. 2023)
#   caga   CAGA as the paper specifies it: keys fitted in the metric of the attention input's second moment with a
#          complex scalar per rotary plane absorbed into each query (the same fit as round 27's unweighted keys);
#          values from a rank-d_h SVD of the stacked value projections, the head factors absorbed into the outputs
#   fuse   the query- and output-weighted fusion (fuse_kv.py, frozen in round 27)
# Pairings: neighbouring heads (the standard conversion), and the minimum-cost perfect matching of each layer's
# heads under the weighted fusion's own residual (keys and values, each relative to the layer's total).
# A pairing that is not of neighbours is deployed by permuting the query heads and output columns, which
# preserves the function.
# Calibration as in CAGA: 128 C4 documents of 2,048 tokens (validation shard 0, the first documents that long).
# Evaluation: WikiText-2 perplexity over non-overlapping windows of 2,048 and 4,096 tokens (the paper does not
# state its length); CAGA's seven zero-shot tasks in lm-evaluation-harness's formats (BoolQ, RTE, HellaSwag,
# WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA; the first 1,000 examples of each; accuracy for BoolQ, RTE and
# WinoGrande, length-normalized accuracy for the others, the paper's convention); and a steering check: 64 MLP
# neuron write directions and 64 random directions pushed into the residual stream at the middle layer, dose 0.8
# times its median norm, on 16 C4 prompts; a direction's signature is the mean change in next-token probability;
# agreement with the unconverted model is the overlap of the 20 tokens each signature raises most, and the
# cosine of the signatures.
#   python alignment_study/llama2_round.py --registered=<freeze permalink of PREREGISTRATION28.md>
#   python alignment_study/llama2_round.py --dev=<model> [--ncal=32] [--tcal=2048] [--subset=200] [--pplwin=40]
#                                          [--lengths=1024,2048] [--arms=fuse/neighbour,...] [--tag=name]
import json
import os
import shutil
import sys
import tempfile
import time
import networkx as nx
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from hubsfree.adapters import device_label
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


REGISTERED, DEV_MODEL = arg("registered"), arg("dev")
if not REGISTERED and not DEV_MODEL:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION28.md>; use --dev=<model>")
ARMS = ("fuse/paired", "caga/paired", "fuse/neighbour", "caga/neighbour", "mean/neighbour")
if DEV_MODEL:
    NAME, REVISION, OUT = DEV_MODEL, None, "alignment_study/llama2_dev"
    N_CAL, T_CAL = int(arg("ncal", 32)), int(arg("tcal", 2048))
    SUBSET, N_PPL = int(arg("subset", 200)), int(arg("pplwin", 40))
    LENGTHS = tuple(int(t) for t in arg("lengths", "1024,2048").split(","))
    ARMS = tuple(arg("arms", ",".join(ARMS)).split(","))
    TAG = arg("tag", f"cal{N_CAL}x{T_CAL}")
else:
    NAME, REVISION, OUT = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d", "alignment_study/llama2"
    N_CAL, T_CAL, SUBSET, N_PPL, LENGTHS, TAG = 128, 2048, 1000, None, (2048, 4096), None
ZS7 = {"boolq": "acc", "rte": "acc", "hellaswag": "acc_norm", "winogrande": "acc", "arc_easy": "acc_norm",
       "arc_challenge": "acc_norm", "openbookqa": "acc_norm"}
N_DIR, N_PROMPT, PROMPT_LEN, DOSE, TOPK = 64, 16, 128, 0.8, 20
torch.set_grad_enabled(False)


# ---------------- data ----------------
def extra_tasks(subset):
    """BoolQ and RTE in lm-evaluation-harness's formats (super_glue boolq, glue rte); accuracy only."""
    T = {}
    df = pd.read_parquet(hf_hub_download("google/boolq", "data/validation-00000-of-00001.parquet", repo_type="dataset"))
    T["boolq"] = [([f"{r.passage}\nQuestion: {r.question}?\nAnswer:"] * 2, [" no", " yes"], int(bool(r.answer)), None) for r in df.itertuples()]
    df = pd.read_parquet(hf_hub_download("aps/super_glue", "rte/validation-00000-of-00001.parquet", repo_type="dataset"))
    T["rte"] = [([f"{r.premise}\nQuestion: {r.hypothesis} True or False?\nAnswer:"] * 2, [" True", " False"], int(r.label), None) for r in df.itertuples()]
    return {k: v[:subset] for k, v in T.items()}


def c4_prompts(tok, n, T, longest):
    """The first n documents of C4 validation shard 0 with at least T and fewer than `longest` tokens (so none is
    a calibration document), truncated to T."""
    import gzip
    p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    out = []
    with gzip.open(p, "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
            if T <= len(ids) < longest:
                out.append(torch.tensor(ids[:T])[None])
                if len(out) == n:
                    break
    return torch.cat(out)


# ---------------- pairing ----------------
def pair_costs(kv, l, S, parts=False):
    """Relative residual of the weighted fusion for every pair of KV heads of layer l: the key residual (query-
    energy weighted, per rotary plane) over the layer's total key energy plus the value residual (output weighted)
    over the layer's total value energy. Full rotary embedding and no biases (the llama layout)."""
    assert kv.r == kv.hd and not kv.has_bias(l)
    a, hd, half, n = kv.attn(l), kv.hd, kv.hd // 2, kv.nkv
    Q = [list(range(i * kv.rep, (i + 1) * kv.rep)) for i in range(n)]
    Wk, Wv, Wq = (F.host64(m.weight) for m in (a.k_proj, a.v_proj, a.q_proj))
    Wo = F.host64(a.o_proj.weight)
    GK, GV = Wk @ S @ Wk.T, Wv @ S @ Wv.T
    qe = ((Wq @ S) * Wq).sum(1).view(kv.n, hd)                                 # query energy of every row
    w = torch.stack([qe[Q[i]].sum(0)[:half] + qe[Q[i]].sum(0)[half:] for i in range(n)])     # (n, half) per plane
    x = lambda i: torch.arange(i * hd, i * hd + half)
    y = lambda i: torch.arange(i * hd + half, (i + 1) * hd)
    kdiag = torch.stack([GK[x(i), x(i)] + GK[y(i), y(i)] for i in range(n)])                # |b_i,p|^2, (n, half)
    OO = [sum(Wo[:, h * hd:(h + 1) * hd].T @ Wo[:, h * hd:(h + 1) * hd] for h in Q[i]) for i in range(n)]
    Oh = [F.psd_sqrt(o + 1e-12 * torch.eye(hd, dtype=o.dtype) * o.trace())[0] for o in OO]
    ev = torch.tensor([float((OO[i] * GV[i * hd:(i + 1) * hd, i * hd:(i + 1) * hd]).sum()) for i in range(n)])
    Ek, Ev = float((w * kdiag).sum()), float(ev.sum())
    Rk, Rv = torch.zeros(n, n, dtype=torch.float64), torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            re = GK[x(i), x(j)] + GK[y(i), y(j)]; im = GK[y(i), x(j)] - GK[x(i), y(j)]
            a11, a22 = w[i] * kdiag[i], w[j] * kdiag[j]
            off2 = w[i] * w[j] * (re ** 2 + im ** 2)
            lo = (a11 + a22) / 2 - torch.sqrt(((a11 - a22) / 2) ** 2 + off2)            # smaller eigenvalue per plane
            Mi = torch.cat([torch.cat([GV[i * hd:(i + 1) * hd, i * hd:(i + 1) * hd], GV[i * hd:(i + 1) * hd, j * hd:(j + 1) * hd]], 1),
                            torch.cat([GV[j * hd:(j + 1) * hd, i * hd:(i + 1) * hd], GV[j * hd:(j + 1) * hd, j * hd:(j + 1) * hd]], 1)])
            Ob = torch.block_diag(Oh[i], Oh[j])
            lv = torch.linalg.eigvalsh(Ob @ Mi @ Ob)[:hd].clamp_min(0).sum()           # energy the shared value head loses
            Rk[i, j] = Rk[j, i] = float(lo.clamp_min(0).sum()); Rv[i, j] = Rv[j, i] = float(lv)
    cost = Rk / Ek + Rv / Ev
    return (cost, Rk, Rv, Ek, Ev) if parts else cost


def min_cost_pairs(cost):
    g = nx.Graph()
    n = cost.shape[0]
    g.add_weighted_edges_from((i, j, float(cost[i, j])) for i in range(n) for j in range(i + 1, n))
    return sorted(sorted(e) for e in nx.min_weight_matching(g))


# ---------------- constructions ----------------
@torch.no_grad()
def convert_caga(kv, S, groups_by_layer):
    for l in range(kv.L):
        S64 = S[l].double()
        roots = F.layer_roots(S64, kv.d)
        a, hd = kv.attn(l), kv.hd
        for G in groups_by_layer[l]:
            G = list(G)
            Cv = torch.cat([F.host64(a.v_proj.weight[i * hd:(i + 1) * hd]) for i in G])
            O = {h: F.host64(a.o_proj.weight[:, h * hd:(h + 1) * hd]) for i in G for h in range(i * kv.rep, (i + 1) * kv.rep)}
            K.fuse_kv_group(kv, l, G, S64, roots, qw=False, ow=False)      # keys: the covariance-metric fit, one complex scalar per plane
            U, s, Vt = torch.linalg.svd(Cv, full_matrices=False)          # values: rank-d_h SVD of the stacked projections
            Vbar = (s[:hd, None] * Vt[:hd]).to(a.v_proj.weight.dtype).to(a.v_proj.weight.device)
            for k, i in enumerate(G):
                a.v_proj.weight[i * hd:(i + 1) * hd] = Vbar
                for h in range(i * kv.rep, (i + 1) * kv.rep):
                    a.o_proj.weight[:, h * hd:(h + 1) * hd] = (O[h] @ U[k * hd:(k + 1) * hd, :hd]).to(a.o_proj.weight.dtype).to(a.o_proj.weight.device)
        del S64, roots


def convert(kv, S, groups, construction):
    if construction == "caga":
        convert_caga(kv, S, groups)
    else:
        K.convert(kv, S, groups, construction)


# ---------------- steering ----------------
@torch.no_grad()
def signatures(model, kv, prompts, dirs, scale, layer):
    """Each direction pushed (times scale) into the residual stream at the output of `layer`, at every position:
    the mean over prompts of the change in next-token probability at the last position, (directions, vocab)."""
    body, head = Z._parts(model)
    state = {"v": None}

    def hook(m, i, o):
        if state["v"] is None:
            return o
        if isinstance(o, tuple):
            return (o[0] + state["v"].to(o[0].dtype),) + tuple(o[1:])
        return o + state["v"].to(o.dtype)
    hk = kv.blocks[layer].register_forward_hook(hook)
    x = prompts.to(DEV)
    probs = lambda: torch.softmax(head(body(input_ids=x, use_cache=False).last_hidden_state[:, -1]).float(), -1)
    try:
        ref = probs()
        sig = torch.empty(len(dirs), ref.shape[-1])
        for k, d in enumerate(dirs):
            state["v"] = (d * scale).to(DEV)
            sig[k] = (probs() - ref).mean(0).cpu()
            state["v"] = None
    finally:
        hk.remove()
    return sig


def agreement(sb, sc):
    tb, tc = sb.topk(TOPK, -1).indices, sc.topk(TOPK, -1).indices
    overlap = torch.tensor([len(set(u.tolist()) & set(v.tolist())) / TOPK for u, v in zip(tb, tc)])
    cos = torch.nn.functional.cosine_similarity(sb, sc, dim=-1)
    return overlap, cos


# ---------------- run ----------------
def run(log, path):
    partial = path + ".partial"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    dtype = torch.bfloat16 if (not DEV_MODEL or "3b" in NAME.lower() or "7b" in NAME.lower()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=dtype, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    log(f"{NAME}: {dtype}, layers {kv.L}, heads {kv.n}, KV heads {kv.nkv}, head dim {kv.hd}, rotary {kv.r}")
    res = json.load(open(partial)) if os.path.exists(partial) else {"evals": {}}
    res.update({"model": NAME, "revision": REVISION, "dtype": str(dtype), "layers": kv.L, "heads": kv.n, "kv_heads": kv.nkv,
                "head_dim": kv.hd, "calibration": {"source": "allenai/c4 en validation shard 0", "windows": N_CAL, "tokens": T_CAL},
                "lengths": list(LENGTHS), "subset": SUBSET, "ppl_windows": N_PPL, "zs7_metrics": ZS7})
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    S, ntok = K.second_moments(model, kv, cal, DEV)
    log(f"  second moments from {ntok} tokens")
    snapdir = tempfile.mkdtemp(prefix="llama2_snapshot_")              # attention weights on disk, not in memory
    for l in range(kv.L):
        torch.save([p.detach().cpu() for p in kv.params(l)], f"{snapdir}/{l}.pt")

    def restore():
        for l in range(kv.L):
            for p, s in zip(kv.params(l), torch.load(f"{snapdir}/{l}.pt")):
                p.copy_(s.to(p.device))
    x0 = cal[:1, :256].to(DEV)
    if "groups_of_one_max_logit_diff" not in res:
        ref = model(input_ids=x0).logits.float()
        K.convert(kv, S, [[[i] for i in range(kv.nkv)] for _ in range(kv.L)], "fuse")
        res["groups_of_one_max_logit_diff"] = float((model(input_ids=x0).logits.float() - ref).abs().max())
        res["logit_scale"] = float(ref.abs().max()); restore()
        log(f"  groups of one reproduce the model: max logit difference {res['groups_of_one_max_logit_diff']:.2e} of {res['logit_scale']:.1f}")
    # pairings
    neighbour = [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)]
    if "pairing" not in res:
        t1 = time.time(); paired, cn, cp = [], [], []
        for l in range(kv.L):
            cost = pair_costs(kv, l, S[l].double())
            pl = min_cost_pairs(cost); paired.append(pl)
            cn.append(float(sum(cost[i, j] for i, j in neighbour[l]))); cp.append(float(sum(cost[i, j] for i, j in pl)))
        res["pairing"] = {"paired": paired, "cost_neighbour": cn, "cost_paired": cp,
                          "neighbour_pairs_kept": int(sum(sum(1 for i, j in pl if j == i + 1 and i % 2 == 0) for pl in paired))}
        log(f"  pairing: summed relative residual {sum(cn):.3f} for neighbours, {sum(cp):.3f} for the matching; "
            f"{res['pairing']['neighbour_pairs_kept']} of {kv.L * kv.nkv // 2} pairs are neighbour pairs ({time.time() - t1:.0f}s)")
    groups = {"neighbour": neighbour, "paired": res["pairing"]["paired"]}
    # steering material, measured on the unconverted model
    layer = kv.L // 2
    prompts = c4_prompts(tok, N_PROMPT, PROMPT_LEN, T_CAL)
    g = torch.Generator().manual_seed(0)
    Wd = kv.blocks[layer].mlp.down_proj.weight.detach().float().cpu()            # (d, MLP width): column = a neuron's write
    cols = torch.randperm(Wd.shape[1], generator=g)[:N_DIR]
    dirs = torch.cat([Wd[:, cols].T, torch.randn(N_DIR, kv.d, generator=g)])
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    norms = []
    hk = kv.blocks[layer].register_forward_hook(lambda m, i, o: norms.append((o[0] if isinstance(o, tuple) else o)[:, 1:].float().norm(dim=-1).flatten().cpu()))
    model(input_ids=prompts.to(DEV), use_cache=False); hk.remove()
    scale = DOSE * float(torch.cat(norms).median())
    sig_base = signatures(model, kv, prompts, dirs, scale, layer)
    res["steering"] = {"layer": layer, "dose": scale, "neurons": cols.tolist(), "n_random": N_DIR, "prompts": N_PROMPT,
                       "prompt_tokens": PROMPT_LEN, "topk": TOPK}
    log(f"  steering: layer {layer}, {N_DIR} neuron and {N_DIR} random directions, dose {scale:.1f}")
    tasks = {**Z.load_tasks(subset=SUBSET), **extra_tasks(SUBSET)}
    tasks = {k: tasks[k] for k in ZS7}
    reqs = Z.build_requests(tok, tasks)
    win = {T: Z.wikitext2_windows(tok, T=T)[:N_PPL] for T in LENGTHS}

    def evaluate(key):
        if key in res["evals"]:
            log(f"  {key}: cached"); return
        t2 = time.time()
        r = Z.evaluate_tasks(model, tok, tasks, reqs, DEV)
        r["zs7"] = float(np.mean([r[t][m] for t, m in ZS7.items()]))
        r["zs7_acc"] = float(np.mean([r[t]["acc"] for t in ZS7]))
        for T in LENGTHS:
            r[f"ppl_{T}"] = Z.perplexity(model, win[T], DEV, bs=max(1, 4096 // T))
        ov, cs = agreement(sig_base, signatures(model, kv, prompts, dirs, scale, layer))
        r["steer"] = {"overlap_neuron": float(ov[:N_DIR].mean()), "overlap_random": float(ov[N_DIR:].mean()),
                      "cos_neuron": float(cs[:N_DIR].mean()), "cos_random": float(cs[N_DIR:].mean()),
                      "overlap": ov.tolist(), "cos": cs.tolist()}
        r["seconds"] = round(time.time() - t2, 1); res["evals"][key] = r
        json.dump(res, open(partial, "w"))
        log(f"  {key}: ppl " + ", ".join(f"{T}: {r[f'ppl_{T}']:.2f}" for T in LENGTHS) + f"; ZS-7 {r['zs7']:.4f}; "
            f"steering overlap neuron {r['steer']['overlap_neuron']:.3f} random {r['steer']['overlap_random']:.3f}")
    evaluate("base")
    for arm in ARMS:
        construction, pairing = arm.split("/")
        if arm not in res["evals"]:
            t3 = time.time(); convert(kv, S, groups[pairing], construction)
            log(f"  {arm}: converted in {time.time() - t3:.0f}s")
        evaluate(arm)
        restore()
    res["restore_check_ppl"] = Z.perplexity(model, win[LENGTHS[0]], DEV, bs=max(1, 4096 // LENGTHS[0]))
    res["runtime_s"] = round(time.time() - t0, 1)
    shutil.rmtree(snapdir, ignore_errors=True)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    path = f"{OUT}/{NAME.split('/')[-1]}{'_' + TAG if TAG else ''}.json"
    if os.path.exists(path):
        sys.exit(f"exists: {path}")
    res = run(log, path)
    res["registered"] = REGISTERED; res["dev"] = bool(DEV_MODEL)
    json.dump(res, open(path, "w"), indent=1)
    if os.path.exists(path + ".partial"):
        os.remove(path + ".partial")
    log("LLAMA2_ROUND_DONE")
