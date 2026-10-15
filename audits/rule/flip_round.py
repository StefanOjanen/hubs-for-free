# PREREGISTRATION23: does the weighting of a statistic decide whether the sink
# dominates it? Round 18 found that statistics weighting positions by attention
# mass were reproduced by the null that keeps the shared columns, and
# statistics counting positions above a threshold were not, a reading formed on
# development data. This round tests it prospectively by flipping the weighting
# of five audited statistics and predicting each flipped statistic's share s in
# advance. Original statistics are recomputed on the same maps and draws as
# references.
#
#   heavy  H2O (round 18, A1)          mass: share of accumulated attention on the top 10% of positions
#                                      count (new): share of above-average selections on the top 10% most-selected positions
#   persist Scissorhands (round 18, A2) count: persistence ratio minus pivotal share
#                                      mass (new): share of second-half attention mass on the first half's top 10%
#                                      positions by first-half mass, minus 0.1
#   hit    SnapKV (round 18, A3)       count: hit rate of above-threshold generation features
#                                      mass (new): share of generation attention mass on the voted positions
#   chai   CHAI (Target 3, last row)   mass: mean Pearson correlation between heads' last rows
#                                      count (new): mean Jaccard overlap between heads' top-5% positions of the last row
#   clark  Clark et al. (Target 1)     mass: Jensen-Shannon contrast, different minus same layer
#                                      count (new): Jaccard contrast of top-5 positions per row, same minus different layer
# Nulls: (a) random softmax rows; (c) each row permuted with the shared columns
# and the diagonal fixed (10 draws each); s = (median_c - median_a) / (real - median_a).
#   python audits/rule/flip_round.py dec|snap|chai|clark [--dev] --registered=<freeze permalink>
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "audits/rule")
sys.path.insert(0, "audits/readings")
_argv = sys.argv; sys.argv = [sys.argv[0], "X", "--dev"]
import rule_round as RR
sys.argv = _argv
from readings_common import permute_free, free_order, random_maps, js_matrix
from hubsfree.stats import sink_columns
from hubsfree.adapters import pick_device, pick_dtype, release_memory

PART = sys.argv[1] if len(sys.argv) > 1 else ""
DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of audits/PREREGISTRATION23.md>; use --dev")
DEV = pick_device("auto")
DRAWS = 3 if DEV_RUN else 10
RR.DRAWS = DRAWS                                                    # round 18's null generator reads its module's draw count
OUT = "audits/rule/flip_dev" if DEV_RUN else "audits/rule/flip"
torch.set_grad_enabled(False)


# ---------------- statistics ----------------
def heavy_mass(A):
    return RR.stat_heavy(A)


def heavy_count(A):
    n, T, _ = A.shape
    r = torch.arange(T, device=A.device)
    sel = A > (1.0 / (r + 1).float())[None, :, None]
    sel[:, 0, :] = False
    c = sel.float().sum(1)                                          # (n, T): selections per key
    k = max(1, int(np.ceil(0.1 * T)))
    return float((c.topk(k, dim=-1).values.sum(-1) / c.sum(-1).clamp_min(1)).mean())


def persist_count(A):
    return RR.stat_persist(A)


def persist_mass(A):
    n, T, _ = A.shape; t = T // 2
    m1 = A[:, 1:t, :t].sum(1); m2 = A[:, t:, :t].sum(1)              # (n, t)
    k = max(1, int(np.ceil(0.1 * t)))
    top = m1.topk(k, dim=-1).indices
    share = m2.gather(1, top).sum(-1) / m2.sum(-1).clamp_min(1e-12)
    return float((share - 0.1).mean())


def hit_mass(obs, gnr, Lp, k):
    votes = obs[..., :Lp].sum(1)
    sel = torch.zeros_like(votes, dtype=torch.bool); sel.scatter_(1, votes.topk(k, dim=-1).indices, True)
    g = gnr[..., :Lp]
    return float((g * sel[:, None, :]).sum() / g.sum().clamp_min(1e-12))


def corr_last(rows):                                                # rows (n, T)
    R = np.corrcoef(rows.cpu().numpy()); n = R.shape[0]
    return float(R[~np.eye(n, dtype=bool)].mean())


def jaccard_last(rows, frac=0.05):
    n, T = rows.shape; k = max(1, int(np.ceil(frac * T)))
    S = torch.zeros(n, T, device=rows.device); S.scatter_(1, rows.topk(k, dim=-1).indices, 1.0)
    inter = S @ S.T; J = inter / (2 * k - inter)
    return float(J[~torch.eye(n, dtype=torch.bool, device=rows.device)].mean())


def jaccard_matrix(A, k=5):
    """A (H, T, T): mean over rows of the Jaccard overlap of the heads' top-k positions."""
    H, T, _ = A.shape
    S = torch.zeros(H, T, T, device=A.device); S.scatter_(2, A.topk(k, dim=-1).indices, 1.0)
    inter = torch.einsum("hrt,grt->hg", S, S) / T
    return (inter / (2 * k - inter)).cpu().numpy().astype(np.float64)


def contrast(Msim, layer, similarity):
    n = len(layer); same = (layer[:, None] == layer[None, :]) & ~np.eye(n, dtype=bool); diff = layer[:, None] != layer[None, :]
    return float(Msim[same].mean() - Msim[diff].mean()) if similarity else float(Msim[diff].mean() - Msim[same].mean())


# ---------------- parts ----------------
def part_dec(name, log):
    """Heavy hitters and persistence, mass and count, on full causal maps (round 18's windows)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name); wins = RR.decoder_windows(tok, 2 if DEV_RUN else 8, 1024)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, pick_dtype(name, DEV, "auto"))).eval().to(DEV)
    mods = RR.attention_modules(model); L = len(mods); cap = {}
    hs = [m.register_forward_hook((lambda l: lambda mod, a, o: cap.__setitem__(l, o[1][0]))(l)) for l, m in enumerate(mods)]
    fn = {"heavy_mass": heavy_mass, "heavy_count": heavy_count, "persist_count": persist_count, "persist_mass": persist_mass}
    acc = {"real": {s: [] for s in fn}, "a_random": {s: [[] for _ in range(DRAWS)] for s in fn}, "c_colset": {s: [[] for _ in range(DRAWS)] for s in fn}}
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    for d, x in enumerate(wins):
        cap.clear(); model(input_ids=x.to(DEV))
        for l in range(L):
            A = cap[l].float()
            for s, f in fn.items():
                acc["real"][s].append(f(A))
            S = list(sink_columns(A.cpu().numpy().astype(np.float64)))
            k = {"a_random": 0, "c_colset": 0}
            for fam, Y in RR.null_maps(A, S, gen):
                if fam == "b_marginal":
                    continue
                for s, f in fn.items():
                    acc[fam][s][k[fam]].append(f(Y))
                k[fam] += 1
            cap[l] = None
        log(f"  window {d + 1}/{len(wins)}")
    for h in hs:
        h.remove()
    del model; release_memory(DEV)
    return {s: {"real": float(np.mean(acc["real"][s])), "a_random": [float(np.mean(v)) for v in acc["a_random"][s]],
                "c_colset": [float(np.mean(v)) for v in acc["c_colset"][s]]} for s in fn}


def part_snap(name, log):
    seqs = RR.snapkv_sequences(name, 2 if DEV_RUN else 6, 1920, 128, log)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, pick_dtype(name, DEV, "auto"))).eval().to(DEV)
    mods = RR.attention_modules(model); L = len(mods); cap = {}
    Lprompt, Lobs = 1920, 128; Lp = Lprompt - Lobs; k = max(1, int(0.1 * Lp))
    hs = [m.register_forward_hook((lambda l: lambda mod, a, o: cap.__setitem__(l, o[1][0, :, Lp:, :]))(l)) for l, m in enumerate(mods)]
    fn = {"hit_count": lambda o, g: RR.stat_hit(o, g, Lp, k), "hit_mass": lambda o, g: hit_mass(o, g, Lp, k)}
    acc = {"real": {s: [] for s in fn}, "a_random": {s: [[] for _ in range(DRAWS)] for s in fn}, "c_colset": {s: [[] for _ in range(DRAWS)] for s in fn}}
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    for d, x in enumerate(seqs):
        cap.clear(); model(input_ids=x.to(DEV)); T = x.shape[1]
        for l in range(L):
            Rw = cap[l].float(); n, R, _ = Rw.shape
            for s, f in fn.items():
                acc["real"][s].append(f(Rw[:, :Lobs], Rw[:, Lobs:]))
            rows = torch.arange(Lp, T, device=DEV)[:, None]; cols = torch.arange(T, device=DEV)[None, :]
            support = cols <= rows; free_b = support & (cols != rows)
            cm = Rw.mean((0, 1)); S = [int(c) for c in torch.nonzero(cm >= 0.10).flatten().tolist()] or [int(cm.argmax())]
            free_c = free_b.clone(); free_c[:, S] = False; dc = free_order(free_c, n)
            for kd in range(DRAWS):
                for fam, Y in (("a_random", random_maps(n, R, T, support, gen, DEV)), ("c_colset", permute_free(Rw, free_c, gen, dc))):
                    for s, f in fn.items():
                        acc[fam][s][kd].append(f(Y[:, :Lobs], Y[:, Lobs:]))
            cap[l] = None
        log(f"  sequence {d + 1}/{len(seqs)}")
    for h in hs:
        h.remove()
    del model; release_memory(DEV)
    return {s: {"real": float(np.mean(acc["real"][s])), "a_random": [float(np.mean(v)) for v in acc["a_random"][s]],
                "c_colset": [float(np.mean(v)) for v in acc["c_colset"][s]]} for s in fn}


def part_chai(name, log):
    """Last-row statistics on Target 3's documents: Pearson correlation and top-5% Jaccard overlap."""
    sys.path.insert(0, "audits/chai"); _a = sys.argv; sys.argv = [sys.argv[0]]
    from reproduce import c4_windows
    sys.argv = _a
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name); wins = c4_windows(tok, 4 if DEV_RUN else 32, 1024)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, pick_dtype(name, DEV, "auto"))).eval().to(DEV)
    mods = RR.attention_modules(model); L = len(mods); cap = {}
    hs = [m.register_forward_hook((lambda l: lambda mod, a, o: cap.__setitem__(l, o[1][0]))(l)) for l, m in enumerate(mods)]
    fn = {"corr_last": corr_last, "jaccard_last": jaccard_last}
    acc = {"real": {s: [] for s in fn}, "a_random": {s: [[] for _ in range(DRAWS)] for s in fn}, "c_colset": {s: [[] for _ in range(DRAWS)] for s in fn}}
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    for d, x in enumerate(wins):
        cap.clear(); model(input_ids=x.to(DEV))
        for l in range(L):
            A = cap[l].float(); n, T, _ = A.shape
            S = list(sink_columns(A.cpu().numpy().astype(np.float64)))
            last = A[:, -1:, :]                                     # (n, 1, T)
            for s, f in fn.items():
                acc["real"][s].append(f(last[:, 0]))
            support = torch.ones(1, T, dtype=torch.bool, device=DEV)
            free_c = support.clone(); free_c[:, S] = False; free_c[:, T - 1] = False; dc = free_order(free_c, n)
            for kd in range(DRAWS):
                for fam, Y in (("a_random", random_maps(n, 1, T, support, gen, DEV)), ("c_colset", permute_free(last, free_c, gen, dc))):
                    for s, f in fn.items():
                        acc[fam][s][kd].append(f(Y[:, 0]))
            cap[l] = None
        log(f"  document {d + 1}/{len(wins)}")
    for h in hs:
        h.remove()
    del model; release_memory(DEV)
    return {s: {"real": float(np.mean(acc["real"][s])), "a_random": [float(np.mean(v)) for v in acc["a_random"][s]],
                "c_colset": [float(np.mean(v)) for v in acc["c_colset"][s]]} for s in fn}


def part_clark(name, log):
    """Target 1's inputs: JS contrast and top-5 Jaccard contrast, layer labels from the model."""
    from transformers import AutoModel, AutoTokenizer
    from hubsfree.adapters import wikitext_texts
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    Lc, Hc = model.config.num_hidden_layers, model.config.num_attention_heads; layer = np.repeat(np.arange(Lc), Hc)
    texts = wikitext_texts(6 if DEV_RUN else 38)
    acc = {s: {"real": 0, "a_random": [0] * DRAWS, "c_colset": [0] * DRAWS} for s in ("js", "jac")}
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    for w, t in enumerate(texts):
        e = tok(t, return_tensors="pt", truncation=True, max_length=64)
        A = torch.cat([a[0].float() for a in model(**{k: v.to(DEV) for k, v in e.items()}, output_attentions=True).attentions], 0)
        T = A.shape[-1]; offd = ~torch.eye(T, dtype=torch.bool, device=DEV)
        free_c = offd.clone(); free_c[:, [0, T - 1]] = False; dc = free_order(free_c, A.shape[0])
        full = torch.ones(T, T, dtype=torch.bool, device=DEV)
        acc["js"]["real"] = acc["js"]["real"] + js_matrix(A); acc["jac"]["real"] = acc["jac"]["real"] + jaccard_matrix(A)
        for kd in range(DRAWS):
            for fam, Y in (("a_random", random_maps(A.shape[0], T, T, full, gen, DEV)), ("c_colset", permute_free(A, free_c, gen, dc))):
                acc["js"][fam][kd] = acc["js"][fam][kd] + js_matrix(Y); acc["jac"][fam][kd] = acc["jac"][fam][kd] + jaccard_matrix(Y)
    del model; release_memory(DEV)
    nW = len(texts)
    out = {}
    for s, sim in (("js", False), ("jac", True)):
        key = "clark_js" if s == "js" else "clark_jaccard"
        out[key] = {"real": contrast(acc[s]["real"] / nW, layer, sim),
                    "a_random": [contrast(m / nW, layer, sim) for m in acc[s]["a_random"]],
                    "c_colset": [contrast(m / nW, layer, sim) for m in acc[s]["c_colset"]]}
    return out


PLAN = {"dec": ("Qwen/Qwen2.5-0.5B", "facebook/opt-6.7b", part_dec), "snap": ("Qwen/Qwen2.5-0.5B", "mistralai/Mistral-7B-Instruct-v0.2", part_snap),
        "chai": ("Qwen/Qwen2.5-0.5B", "mistralai/Mistral-7B-v0.1", part_chai), "clark": ("google/bert_uncased_L-4_H-512_A-8", "bert-base-uncased", part_clark)}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    dev_name, conf_name, f = PLAN[PART]; name = dev_name if DEV_RUN else conf_name
    path = f"{OUT}/{PART}_{name.split('/')[-1]}.json"
    if os.path.exists(path):
        print("SKIP (exists)", path); sys.exit(0)
    res = f(name, log)
    for s, v in res.items():
        rnd = float(np.median(v["a_random"])); v["random_median"] = rnd; v["colset_median"] = float(np.median(v["c_colset"]))
        v["s"] = (v["colset_median"] - rnd) / (v["real"] - rnd) if abs(v["real"] - rnd) > 1e-12 else None
    out = {"part": PART, "model": name, "registered": REGISTERED, "dev": DEV_RUN, "draws": DRAWS, "device": DEV, "stats": res, "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(path, "w"), indent=1)
    log("RESULT " + PART + " " + name + " " + json.dumps({s: {"real": round(v["real"], 4), "s": None if v["s"] is None else round(v["s"], 3)} for s, v in res.items()}))
    print("FLIP_DONE", PART, flush=True)
