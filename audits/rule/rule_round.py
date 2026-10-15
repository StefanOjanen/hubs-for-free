# PREREGISTRATION18: a rule for which published attention claims survive
# null models, tested prospectively on six claims nobody has audited with
# nulls. Rule: let s be the share of a claimed effect that a null keeping each
# row's values and the layer's shared columns (sink set, or [CLS]/[SEP])
# reproduces, measured from the random-null baseline,
#     s = (median_colset - median_random) / (real - median_random).
# Claims that aggregate attention over many positions or heads into a
# distribution-level summary: s >= 0.5. Claims that a head's attention lands
# on a token picked out by content or relative position: s <= 0.2.
#
# Statistics, as the sources define them (audits/PREREGISTRATION18.md):
#   A1 heavy   H2O (Zhang et al. 2023): accumulated attention per key position
#              (column sums over queries); share held by the top 10 percent of
#              positions, mean over heads.
#   A2 persist Scissorhands (Liu et al. 2023): pivotal = weight above the row
#              average 1/(r+1); t = T/2; persistence ratio P = |S_late & S_early|
#              / |S_late| over keys < t; f = |S_early| / t; statistic P - f.
#   A3 hitrate SnapKV (Li et al. 2024): votes = sum over the last 128 prompt
#              rows; selected = top 10 percent of prefix positions per head;
#              important = generated-row weights above 1/L_prefix; hit rate =
#              important and selected / important.
#   L1 induct  Olsson et al. (2022): prefix-matching score on twice-repeated
#              random tokens; mean of the 5 best heads.
#   L2 offset  Clark et al. (2019, 3.1): mean attention to the previous token
#              of the 4 best heads and to the next token of the 5 best heads.
#   L3 syntax  Clark et al. (2019, 4.2): word-level maps; per dependency
#              relation, the best head's accuracy over both directions; mean
#              over relations (UD English EWT).
# Nulls per map: (a) random softmax rows over the support; (b) each row
# permuted over its support with the diagonal fixed; (c) the same with the
# shared columns fixed; (d) untrained initializations of the architecture.
#   python audits/rule/rule_round.py <claim> [--dev] --registered=<freeze permalink>
import json
import os
import random
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "audits/readings")
from readings_common import permute_free, free_order, random_maps
from hubsfree.stats import sink_columns
from hubsfree.adapters import pick_device, pick_dtype, release_memory

CLAIM = sys.argv[1] if len(sys.argv) > 1 else ""
DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of audits/PREREGISTRATION18.md>; use --dev")
DEV = pick_device("auto")
DRAWS = 3 if DEV_RUN else 10
N_INIT = 1 if DEV_RUN else 3
OUT = "audits/rule/dev" if DEV_RUN else "audits/rule"
FAMS = ("a_random", "b_marginal", "c_colset")
torch.set_grad_enabled(False)


# ---------------- statistics on one layer's maps (n, T, T) ----------------
def stat_heavy(A):
    acc = A.sum(1)                                                  # (n, T): accumulated attention per key
    k = max(1, int(np.ceil(0.1 * A.shape[-1])))
    return float((acc.topk(k, dim=-1).values.sum(-1) / acc.sum(-1)).mean())


def stat_persist(A):
    n, T, _ = A.shape; t = T // 2
    r = torch.arange(T, device=A.device)
    piv = A > (1.0 / (r + 1).float())[None, :, None]               # (n, T, T): pivotal for row r
    keys_early = torch.arange(T, device=A.device) < t
    early = piv[:, 1:t, :].any(1) & keys_early                     # S_{0->t}
    late = piv[:, t:, :].any(1) & keys_early                       # S_{t->l} restricted to keys < t
    P = (late & early).sum(-1).float() / late.sum(-1).clamp_min(1).float()
    f = early.sum(-1).float() / t
    return float((P - f).mean())


# ---------------- decoder maps: capture, nulls ----------------
def attention_modules(model):
    mods = [m for nme, m in model.named_modules() if nme.split(".")[-1] in ("self_attn", "self_attention", "attn", "attention") and "Attention" in type(m).__name__]
    assert len(mods) == model.config.num_hidden_layers
    return mods


def null_maps(A, S, gen, causal=True, keep_cols=None):
    """Yield (family, map) for DRAWS draws of each random family."""
    n, R, T = A.shape
    rows = torch.arange(R, device=A.device)[:, None]; cols = torch.arange(T, device=A.device)[None, :]
    support = (cols <= rows) if causal else torch.ones(R, T, dtype=torch.bool, device=A.device)
    free_b = support & (cols != rows)
    free_c = free_b.clone(); free_c[:, list(S if keep_cols is None else keep_cols)] = False
    db, dc = free_order(free_b, n), free_order(free_c, n)
    for _ in range(DRAWS):
        yield "a_random", random_maps(n, R, T, support, gen, A.device)
        yield "b_marginal", permute_free(A, free_b, gen, db)
        yield "c_colset", permute_free(A, free_c, gen, dc)


def decoder_windows(tok, n, T, source="wikitext"):
    from datasets import load_dataset
    if source == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt")["input_ids"][0]
        m = ids.numel() // T; picks = np.linspace(0, m - 1, n).round().astype(int)
        return [ids[p * T:(p + 1) * T][None] for p in picks]
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True); out = []
    for ex in ds:
        e = tok(ex["text"], return_tensors="pt", truncation=True, max_length=T)["input_ids"]
        if e.shape[1] >= T:
            out.append(e)
        if len(out) == n:
            return out


def run_A12(name, wins, trained, gen, log):
    """A1 and A2 on full causal maps; real and nulls per layer and document."""
    from transformers import AutoConfig, AutoModelForCausalLM
    dt = pick_dtype(name, DEV, "auto")
    if trained:
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    else:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), attn_implementation="eager", dtype=torch.bfloat16).eval().to(DEV)
    mods = attention_modules(model); L = len(mods); cap = {}
    hs = [m.register_forward_hook((lambda l: lambda mod, a, o: cap.__setitem__(l, o[1][0]))(l)) for l, m in enumerate(mods)]
    res = {"real": {"heavy": [], "persist": []}}
    if trained:
        for f in FAMS:
            res[f] = {"heavy": [[] for _ in range(DRAWS)], "persist": [[] for _ in range(DRAWS)]}
    for d, x in enumerate(wins):
        cap.clear(); model(input_ids=x.to(DEV))
        for l in range(L):
            A = cap[l].float()
            res["real"]["heavy"].append(stat_heavy(A)); res["real"]["persist"].append(stat_persist(A))
            if trained:
                S = list(sink_columns(A.cpu().numpy().astype(np.float64)))
                k = {f: 0 for f in FAMS}
                for fam, Y in null_maps(A, S, gen):
                    res[fam]["heavy"][k[fam]].append(stat_heavy(Y)); res[fam]["persist"][k[fam]].append(stat_persist(Y)); k[fam] += 1
            cap[l] = None
        log(f"  document {d + 1}/{len(wins)}")
    for h in hs:
        h.remove()
    del model; release_memory(DEV)
    out = {"real": {s: float(np.mean(v)) for s, v in res["real"].items()}, "real_by_layer": {s: np.array(v).reshape(len(wins), L).mean(0).tolist() for s, v in res["real"].items()}}
    if trained:
        for f in FAMS:
            out[f] = {s: [float(np.mean(v)) for v in res[f][s]] for s in ("heavy", "persist")}
    return out, L


# ---------------- A3: SnapKV hit rate ----------------
def stat_hit(rows_obs, rows_gen, Lp, k):
    """rows_obs (n, Lobs, T), rows_gen (n, G, T) restricted to prefix keys [:Lp]."""
    votes = rows_obs[..., :Lp].sum(1)                               # (n, Lp)
    sel = torch.zeros_like(votes, dtype=torch.bool)
    sel.scatter_(1, votes.topk(k, dim=-1).indices, True)
    imp = rows_gen[..., :Lp] > (1.0 / Lp)
    return float((imp & sel[:, None, :]).sum() / imp.sum().clamp_min(1))


def run_A3(name, seqs, Lprompt, Lobs, trained, gen, log):
    from transformers import AutoConfig, AutoModelForCausalLM
    dt = pick_dtype(name, DEV, "auto")
    if trained:
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    else:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), attn_implementation="eager", dtype=torch.bfloat16).eval().to(DEV)
    mods = attention_modules(model); L = len(mods); cap = {}
    Lp = Lprompt - Lobs; k = max(1, int(0.1 * Lp))
    hs = [m.register_forward_hook((lambda l: lambda mod, a, o: cap.__setitem__(l, o[1][0, :, Lprompt - Lobs:, :]))(l)) for l, m in enumerate(mods)]
    res = {"real": []}
    if trained:
        for f in FAMS:
            res[f] = [[] for _ in range(DRAWS)]
    for d, x in enumerate(seqs):
        cap.clear(); model(input_ids=x.to(DEV))
        T = x.shape[1]
        for l in range(L):
            Rw = cap[l].float()                                     # rows Lp .. T-1 (observation window, then generation)
            obs, gnr = Rw[:, :Lobs], Rw[:, Lobs:]
            res["real"].append(stat_hit(obs, gnr, Lp, k))
            if trained:
                # nulls act on these rows over their own causal support; the sink set is taken from the rows' column mass
                n, R, _ = Rw.shape
                rows = torch.arange(Lp, T, device=Rw.device)[:, None]; cols = torch.arange(T, device=Rw.device)[None, :]
                support = cols <= rows; free_b = support & (cols != rows)
                cm = Rw.mean((0, 1)); S = [int(c) for c in torch.nonzero(cm >= 0.10).flatten().tolist()] or [int(cm.argmax())]
                free_c = free_b.clone(); free_c[:, S] = False
                db, dc = free_order(free_b, n), free_order(free_c, n)
                for kd in range(DRAWS):
                    for fam, Y in (("a_random", random_maps(n, R, T, support, gen, Rw.device)), ("b_marginal", permute_free(Rw, free_b, gen, db)),
                                   ("c_colset", permute_free(Rw, free_c, gen, dc))):
                        res[fam][kd].append(stat_hit(Y[:, :Lobs], Y[:, Lobs:], Lp, k))
            cap[l] = None
        log(f"  sequence {d + 1}/{len(seqs)}")
    for h in hs:
        h.remove()
    del model; release_memory(DEV)
    out = {"real": float(np.mean(res["real"]))}
    if trained:
        for f in FAMS:
            out[f] = [float(np.mean(v)) for v in res[f]]
    return out, L


def snapkv_sequences(name, n, Lprompt, G, log):
    """Prompts: the first Lprompt tokens of C4 validation documents; continuation:
    the instruct model's own greedy generation of G tokens."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    prompts = decoder_windows(tok, n, Lprompt, source="c4")
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dt)).eval().to(DEV)
    seqs = []
    for i, p in enumerate(prompts):
        g = model.generate(input_ids=p.to(DEV), attention_mask=torch.ones_like(p).to(DEV), max_new_tokens=G, min_new_tokens=G, do_sample=False)
        seqs.append(g[:, :Lprompt + G].cpu()); log(f"  generated {i + 1}/{n}")
    del model; release_memory(DEV)
    return seqs


# ---------------- located claims ----------------
def run_L1(name, n_seq, Lrep, trained, gen, log, seed=0):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    rg = np.random.default_rng(seed)
    V = tok.vocab_size
    seqs = [torch.tensor(np.tile(rg.integers(1000, V - 1000, Lrep), 2))[None] for _ in range(n_seq)]
    if trained:
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    else:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), attn_implementation="eager").eval().to(DEV)
    mods = attention_modules(model); L = len(mods); cap = {}
    hs = [m.register_forward_hook((lambda l: lambda mod, a, o: cap.__setitem__(l, o[1][0]))(l)) for l, m in enumerate(mods)]
    q = torch.arange(Lrep, 2 * Lrep, device=DEV); tgt = q - Lrep + 1          # second repeat attends to the token after the earlier occurrence

    def score(A):                                                   # (n, T, T) -> per-head prefix-matching score
        return A[:, q, tgt].mean(-1)

    per_head = {"real": []}
    if trained:
        for f in FAMS:
            per_head[f] = [[] for _ in range(DRAWS)]
    for x in seqs:
        cap.clear(); model(input_ids=x.to(DEV))
        real, nulls = [], {f: [[] for _ in range(DRAWS)] for f in FAMS}
        for l in range(L):
            A = cap[l].float(); real.append(score(A))
            if trained:
                S = list(sink_columns(A.cpu().numpy().astype(np.float64)))
                kk = {f: 0 for f in FAMS}
                for fam, Y in null_maps(A, S, gen):
                    nulls[fam][kk[fam]].append(score(Y)); kk[fam] += 1
        per_head["real"].append(torch.cat(real).cpu().numpy())
        if trained:
            for f in FAMS:
                for kd in range(DRAWS):
                    per_head[f][kd].append(torch.cat(nulls[f][kd]).cpu().numpy())
    for h in hs:
        h.remove()
    del model; release_memory(DEV)
    top5 = lambda arrs: float(np.sort(np.mean(arrs, 0))[-5:].mean())              # mean of the 5 best heads, scores averaged over sequences
    out = {"real": top5(per_head["real"])}
    if trained:
        for f in FAMS:
            out[f] = [top5(per_head[f][kd]) for kd in range(DRAWS)]
    return out, L


def clark_segments(tok, n):
    sys.path.insert(0, "audits/readings")
    _argv = sys.argv; sys.argv = [sys.argv[0], "--check"]
    import clark_source_format as K
    sys.argv = _argv
    return K.wikipedia_segments(tok, n)


def run_L2(name, nseg, trained, gen, log):
    """Clark et al. 3.1 on bert in the source's input format: per head, mean attention to the previous and the next token."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    segs = clark_segments(tok, nseg)
    model = (AutoModel.from_pretrained(name, attn_implementation="eager", dtype=torch.float32) if trained
             else AutoModel.from_config(AutoConfig.from_pretrained(name), attn_implementation="eager")).eval().to(DEV)

    def offs(A):                                                    # (n, T, T) -> (prev, next) per head, over non-special rows
        T = A.shape[-1]; r = torch.arange(1, T - 1, device=A.device)
        return A[:, r, r - 1].mean(-1), A[:, r, r + 1].mean(-1)

    acc = {"real": []}
    if trained:
        for f in FAMS:
            acc[f] = [[] for _ in range(DRAWS)]
    for ids, tt, keep in segs:
        out = model(input_ids=ids.to(DEV), token_type_ids=tt.to(DEV), output_attentions=True)
        A = torch.cat([a[0].float() for a in out.attentions], 0)
        p, nx = offs(A); acc["real"].append(torch.stack([p, nx]).cpu().numpy())
        if trained:
            kk = {f: 0 for f in FAMS}
            for fam, Y in null_maps(A, None, gen, causal=False, keep_cols=keep):
                p, nx = offs(Y); acc[fam][kk[fam]].append(torch.stack([p, nx]).cpu().numpy()); kk[fam] += 1
    del model; release_memory(DEV)

    def summ(arrs):                                                 # mean over segments; top-4 previous and top-5 next heads
        m = np.mean(arrs, 0)
        return float(np.concatenate([np.sort(m[0])[-4:], np.sort(m[1])[-5:]]).mean())

    out = {"real": summ(acc["real"]), "heads_over_half": {"previous": int((np.mean(acc["real"], 0)[0] > 0.5).sum()), "next": int((np.mean(acc["real"], 0)[1] > 0.5).sum())}}
    if trained:
        for f in FAMS:
            out[f] = [summ(acc[f][kd]) for kd in range(DRAWS)]
    return out, 12


UD_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.14/en_ewt-ud-test.conllu"
RELATIONS = ("obj", "det", "nsubj", "amod", "advmod", "aux", "nmod:poss", "compound", "aux:pass", "ccomp", "mark", "compound:prt")


def ud_sentences(path, max_n):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur)
            cur = []
            if len(sents) >= max_n:
                break
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        cur.append((c[1], int(c[6]), c[7]))                        # form, head (1-based, 0 = root), relation
    return sents


def run_L3(name, path, nsent, trained, gen, log):
    """Clark et al. 4.2: word-level maps (sum over a word's tokens as keys, mean as queries), [CLS]/[SEP] ignored;
    per relation, the best head's accuracy over both directions; mean over relations."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = (AutoModel.from_pretrained(name, attn_implementation="eager", dtype=torch.float32) if trained
             else AutoModel.from_config(AutoConfig.from_pretrained(name), attn_implementation="eager")).eval().to(DEV)
    sents = [s for s in ud_sentences(path, 10 ** 6) if 4 <= len(s) <= 40][:nsent]
    counts = {"real": {}}
    if trained:
        for f in FAMS:
            counts[f] = [dict() for _ in range(DRAWS)]

    def tally(store, W, deps):                                      # W (n, w, w) word maps with self excluded
        pred = W.argmax(-1).cpu().numpy()                           # (n, w)
        for (i, h, rel) in deps:
            for direction, (a, b) in (("d2h", (i, h)), ("h2d", (h, i))):
                key = (rel, direction); c = store.setdefault(key, [np.zeros(pred.shape[0]), 0])
                c[0] += (pred[:, a] == b); c[1] += 1

    for s in sents:
        words = [w for w, _, _ in s]
        enc = tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        wid = enc.word_ids()
        if any(w is None for w in wid[1:-1]) or max(w for w in wid if w is not None) != len(words) - 1:
            continue
        out = model(**{k: v.to(DEV) for k, v in enc.items()}, output_attentions=True)
        A = torch.cat([a[0].float() for a in out.attentions], 0)    # (n, T, T)
        T = A.shape[-1]; nw = len(words)
        M = torch.zeros(T, nw, device=DEV)
        for t, w in enumerate(wid):
            if w is not None:
                M[t, w] = 1.0
        cnt = M.sum(0)

        def to_words(Y):
            Wd = (M.T @ Y @ M) / cnt[:, None]                       # mean over a word's query tokens, sum over its key tokens
            Wd = Wd.clone(); idx = torch.arange(nw, device=DEV); Wd[:, idx, idx] = -1.0
            return Wd

        deps = [(i, h - 1, rel) for i, (_, h, rel) in enumerate(s) if h > 0 and rel in RELATIONS]
        tally(counts["real"], to_words(A), deps)
        if trained:
            kk = {f: 0 for f in FAMS}
            keep = (0, T - 1)
            for fam, Y in null_maps(A, None, gen, causal=False, keep_cols=keep):
                tally(counts[fam][kk[fam]], to_words(Y), deps); kk[fam] += 1
    del model; release_memory(DEV)

    def summ(store):
        best = {}
        for (rel, direction), (hits, tot) in store.items():
            best[rel] = max(best.get(rel, 0.0), float((hits / tot).max()))
        return float(np.mean([best[r] for r in RELATIONS if r in best])), best

    real, best = summ(counts["real"])
    out = {"real": real, "best_by_relation": best}
    if trained:
        for f in FAMS:
            out[f] = [summ(counts[f][kd])[0] for kd in range(DRAWS)]
    return out, 12


# ---------------- driver ----------------
PLAN = {
    "A1A2": {"dev": ["Qwen/Qwen2.5-0.5B"], "conf": ["facebook/opt-6.7b"], "n": 2 if DEV_RUN else 8, "T": 1024},
    "A3": {"dev": ["Qwen/Qwen2.5-0.5B"], "conf": ["mistralai/Mistral-7B-Instruct-v0.2"], "n": 2 if DEV_RUN else 6, "Lprompt": 1920, "Lobs": 128, "G": 128},
    "L1": {"dev": ["gpt2"], "conf": ["gpt2-medium", "EleutherAI/pythia-410m"], "n": 4 if DEV_RUN else 16, "Lrep": 128},
    "L2": {"dev": ["bert-base-uncased"], "conf": ["bert-base-cased"], "n": 20 if DEV_RUN else 200},
    "L3": {"dev": ["bert-base-uncased"], "conf": ["bert-base-cased"], "n": 40 if DEV_RUN else 500},
}

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    plan = PLAN[CLAIM]; gen = torch.Generator(device=DEV); gen.manual_seed(0)
    for name in plan["dev" if DEV_RUN else "conf"]:
        path = f"{OUT}/{CLAIM}_{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        log(f"{CLAIM} {name}")
        if CLAIM == "A1A2":
            from transformers import AutoTokenizer
            wins = decoder_windows(AutoTokenizer.from_pretrained(name), plan["n"], plan["T"])
            res, L = run_A12(name, wins, True, gen, log)
            unt = []
            for s in range(N_INIT):
                torch.manual_seed(s); unt.append(run_A12(name, wins, False, gen, lambda s: None)[0]["real"])
            res["untrained"] = unt
        elif CLAIM == "A3":
            seqs = snapkv_sequences(name, plan["n"], plan["Lprompt"], plan["G"], log)
            res, L = run_A3(name, seqs, plan["Lprompt"], plan["Lobs"], True, gen, log)
            unt = []
            for s in range(N_INIT):
                torch.manual_seed(s); unt.append(run_A3(name, seqs, plan["Lprompt"], plan["Lobs"], False, gen, lambda s: None)[0]["real"])
            res["untrained"] = unt
        elif CLAIM == "L1":
            res, L = run_L1(name, plan["n"], plan["Lrep"], True, gen, log)
            unt = []
            for s in range(N_INIT):
                torch.manual_seed(s); unt.append(run_L1(name, plan["n"], plan["Lrep"], False, gen, lambda s: None)[0]["real"])
            res["untrained"] = unt
        elif CLAIM == "L2":
            res, L = run_L2(name, plan["n"], True, gen, log)
            unt = []
            for s in range(N_INIT):
                torch.manual_seed(s); unt.append(run_L2(name, plan["n"], False, gen, lambda s: None)[0]["real"])
            res["untrained"] = unt
        elif CLAIM == "L3":
            ud = "audits/rule/data/en_ewt-ud-test.conllu"
            if not os.path.exists(ud):
                os.makedirs("audits/rule/data", exist_ok=True)
                import urllib.request
                urllib.request.urlretrieve(UD_URL, ud)
            res, L = run_L3(name, ud, plan["n"], True, gen, log)
            unt = []
            for s in range(N_INIT):
                torch.manual_seed(s); unt.append(run_L3(name, ud, plan["n"], False, gen, lambda s: None)[0]["real"])
            res["untrained"] = unt
        res.update({"claim": CLAIM, "model": name, "registered": REGISTERED, "dev": DEV_RUN, "draws": DRAWS, "n_untrained": N_INIT,
                    "device": DEV, "layers": L, "plan": {k: v for k, v in plan.items() if k not in ("dev", "conf")}, "runtime_s": round(time.time() - t0, 1)})
        json.dump(res, open(path, "w"), indent=1)
        log(f"RESULT {CLAIM} {name}: real {res['real'] if not isinstance(res['real'], dict) else res['real']}")
    print("RULE_DONE", CLAIM, flush=True)
