import subprocess, sys, json, math, time
try:
    import transformers, datasets  # noqa
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets"])
import numpy as np, torch
from itertools import combinations, product
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)

MODEL, N, NDEPTH, SEQ, LAYER = "Qwen/Qwen2.5-0.5B", 150, 100, 64, 21
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
        output_attentions=True, attn_implementation="eager",
        dtype=torch.float32).eval()
H, L = model.config.num_attention_heads, model.config.num_hidden_layers
print("MODEL LOADED", H, "heads", L, "layers", flush=True)
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
texts = [t for t in ds["text"] if len(t.split()) > 40][:N]
print("TEXTS", len(texts), flush=True)

def stats(A):
    G = 0.5 * (A - A.transpose(-1, -2))
    gn = torch.linalg.norm(G, dim=(1, 2))
    ipr = (A ** 2).sum(-1)
    dg = torch.diagonal(A, dim1=1, dim2=2)
    return G, gn, (ipr - dg ** 2).sum(-1), A[:, :, 0].mean(-1), dg.mean(-1), ipr.mean(-1)

def coup(G):
    n = G.shape[0]; C = torch.zeros(n, n)
    for i, j in combinations(range(n), 2):
        C[i, j] = C[j, i] = torch.linalg.norm(G[i] @ G[j] - G[j] @ G[i])
    return C

res = {}; am = []; cs = []; iderr = 0.0
lwL = []; lwF = []; lwD = []; lwI = []
sink_by_head = np.zeros(H)
t0 = time.time()
for w, text in enumerate(texts):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ)
    out = model(**ids)
    A = out.attentions[LAYER - 1].squeeze(0)
    G, gn, rhs, sink, dm, ipr = stats(A)
    iderr = max(iderr, float((2 * gn ** 2 - rhs).abs().max()))
    am.append(int(gn.argmax()))
    cs.append(float(np.corrcoef(gn.numpy(), sink.numpy())[0, 1]))
    sink_by_head += sink.numpy()
    if w < NDEPTH:
        for l in range(L):
            Al = out.attentions[l].squeeze(0)
            Gl, g2, _, _, d2, i2 = stats(Al)
            lwL.append(l); lwF.append(float(coup(Gl)[np.triu_indices(H, 1)].mean()))
            lwD.append(float(d2.mean())); lwI.append(float(i2.mean()))
    if w % 25 == 0:
        print("WIN", w, int(time.time() - t0), "s", flush=True)
cnt = np.bincount(am, minlength=H)
res["P1"] = {"identity_max_err": iderr, "corr_gnorm_sink_mean": float(np.mean(cs)),
             "sink_head": int(sink_by_head.argmax())}
res["P2"] = {"hub_head": int(cnt.argmax()), "stability": float(cnt.max() / cnt.sum()),
             "distribution": cnt.tolist()}
res["P1"]["hub_is_sink_head"] = bool(res["P2"]["hub_head"] == res["P1"]["sink_head"])
lw = np.array(lwF); ll = np.array(lwL); lD = np.array(lwD); lI = np.array(lwI)
X = np.stack([lD, lI, np.ones(len(lD))], 1)
beta, _, _, _ = np.linalg.lstsq(X, lw, rcond=None)
pred = X @ beta; ss = ((lw - lw.mean()) ** 2).sum()
res["P5"] = {"r2_locality": float(1 - ((lw - pred) ** 2).sum() / ss),
             "meanF_by_layer": [float(lw[ll == l].mean()) for l in range(L)]}
print("P1P2P5 DONE", flush=True)

rg = np.random.default_rng(0); rows = []
def sur(A):
    S = A.clone()
    for h in range(A.shape[0]):
        for i in range(1, A.shape[1]):
            S[h, i, :i] = A[h, i, torch.from_numpy(rg.permutation(i))]
    return S
for w in range(10):
    ids = tok(texts[w], return_tensors="pt", truncation=True, max_length=SEQ)
    A = model(**ids).attentions[LAYER - 1].squeeze(0)
    G, gn, *_ = stats(A); C = coup(G); ev = torch.linalg.eigvalsh(C)
    tr = np.triu_indices(H, 1)
    r1 = float(np.corrcoef(C.numpy()[tr], torch.outer(gn, gn).numpy()[tr])[0, 1])
    real = dict(gap=float(ev[-1] / ev[-2].abs()), r1=r1)
    sg = []; sr = []
    for _ in range(8):
        Gs, gs, *_ = stats(sur(A)); Cs = coup(Gs); evs = torch.linalg.eigvalsh(Cs)
        sg.append(float(evs[-1] / evs[-2].abs()))
        sr.append(float(np.corrcoef(Cs.numpy()[tr], torch.outer(gs, gs).numpy()[tr])[0, 1]))
    rows.append((real, sg, sr, bool(min(sg) <= real["gap"] <= max(sg))))
res["P3"] = {"median_real_gap": float(np.median([r[0]["gap"] for r in rows])),
             "median_sur_gap": float(np.median([g for r in rows for g in r[1]])),
             "median_real_r1": float(np.median([r[0]["r1"] for r in rows])),
             "median_sur_r1": float(np.median([g for r in rows for g in r[2]])),
             "frac_inside": float(np.mean([r[3] for r in rows]))}
print("P3 DONE", flush=True)

dh = model.config.hidden_size // H
att = model.model.layers[LAYER - 1].self_attn
Wsave = att.o_proj.weight.data.clone()
def evloss(n=40):
    tot = 0.0; ntk = 0
    for t in texts[:n]:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ)
        o = model(**ids, labels=ids["input_ids"])
        tot += float(o.loss) * ids["input_ids"].numel(); ntk += ids["input_ids"].numel()
    return tot / ntk
base = evloss(); hub = res["P2"]["hub_head"]
def abl(h):
    att.o_proj.weight.data[:, h * dh:(h + 1) * dh] = 0
    v = evloss(); att.o_proj.weight.data = Wsave.clone(); return v
dhub = abl(hub) - base
rnds = [int(h) for h in np.random.default_rng(1).choice(
    [h for h in range(H) if h != hub], 5, replace=False)]
drnd = [abl(h) - base for h in rnds]
res["P6"] = {"base_loss": base, "hub_head": hub, "hub_delta": dhub,
             "random_heads": rnds, "random_deltas": drnd,
             "hub_z": float((dhub - np.mean(drnd)) / (np.std(drnd) + 1e-12))}
print("P6 DONE", flush=True)

def sc(A):
    G = (0.5 * (A - A.transpose(-1, -2))).numpy()
    Mv = G.reshape(G.shape[0], -1)
    U, S, Vh = np.linalg.svd(Mv, full_matrices=False)
    B = Vh.reshape(G.shape)
    n = G.shape[0]; f = np.zeros((n, n, n))
    for i, j, k in product(range(n), repeat=3):
        f[i, j, k] = np.sum(B[k] * (B[i] @ B[j] - B[j] @ B[i]))
    Xa = np.array(list(product(range(n), repeat=3)), float)
    y = f.ravel(); kp = np.abs(y) > 1e-4
    return Xa[kp], y[kp]
def rand_tree(rg2, depth=0, maxd=4):
    if depth >= maxd or rg2.random() < 0.28:
        if rg2.random() < 0.7:
            return ["aff", int(rg2.integers(3)), rg2.normal(0, 1), rg2.normal(0, 2)]
        return ["const", rg2.normal() * rg2.choice([0.1, 1, 10])]
    if rg2.random() < 0.45:
        return [str(rg2.choice(["sin", "cos", "exp"])), rand_tree(rg2, depth + 1, maxd)]
    return [str(rg2.choice(["+", "-", "*", "/"])), rand_tree(rg2, depth + 1, maxd), rand_tree(rg2, depth + 1, maxd)]
def consts(t, acc):
    if t[0] == "aff": acc += [(t, 2), (t, 3)]
    elif t[0] == "const": acc.append((t, 1))
    elif t[0] in ("sin", "cos", "exp"): consts(t[1], acc)
    elif t[0] in "+-*/": consts(t[1], acc); consts(t[2], acc)
    return acc
def evt(t, Xa):
    op = t[0]
    if op == "aff": return t[2] * Xa[:, t[1]] + t[3]
    if op == "const": return np.full(Xa.shape[0], t[1])
    if op == "exp": return np.exp(np.clip(evt(t[1], Xa), -20, 20))
    if op in ("sin", "cos"): return getattr(np, op)(evt(t[1], Xa))
    a, b = evt(t[1], Xa), evt(t[2], Xa)
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    return a / np.where(np.abs(b) < 1e-9, 1e-9, b)
def r2v(g, yc, ssq):
    if not np.all(np.isfinite(g)): return -1
    gc = g - g.mean(); den = (gc ** 2).sum()
    return -1 if den < 1e-12 else float((gc @ yc) ** 2 / (den * ssq))
def search(Xa, y, seed, n_rand=8000, n_climb=600):
    rg2 = np.random.default_rng(seed); yc = y - y.mean(); ssq = (yc ** 2).sum()
    bt, best = None, -1
    for _ in range(n_rand):
        t = rand_tree(rg2); s = r2v(evt(t, Xa), yc, ssq)
        if s > best: best, bt = s, t
    cc = consts(bt, [])
    for _ in range(n_climb):
        node, idx = cc[rg2.integers(len(cc))]
        old = node[idx]; node[idx] = old + rg2.normal(0, abs(old) * 0.3 + 0.1)
        s = r2v(evt(bt, Xa), yc, ssq)
        if s > best: best = s
        else: node[idx] = old
    return best
ids = tok(texts[0], return_tensors="pt", truncation=True, max_length=SEQ)
A = model(**ids).attentions[LAYER - 1].squeeze(0)
Xr, yr = sc(A)
T = A.shape[1]
lo = np.random.default_rng(9).normal(size=(H, T, T))
lo = np.where(np.tril(np.ones((T, T), bool)), lo, -1e9)
An = np.exp(lo - lo.max(-1, keepdims=True)); An /= An.sum(-1, keepdims=True)
Xn, yn = sc(torch.tensor(An, dtype=torch.float32))
res["P4"] = {"n_real": int(len(yr)),
             "real_best_r2": [round(search(Xr, yr, s), 4) for s in range(2)],
             "noise_best_r2": [round(search(Xn, yn, s), 4) for s in range(2)]}
json.dump(res, open("qwen_results.json", "w"), indent=1)
print("RESULTS_JSON_START"); print(json.dumps(res)); print("RESULTS_JSON_END", flush=True)
