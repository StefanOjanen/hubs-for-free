# Shared machinery for the tier-1/tier-2 hardening runs.
# All statistics operate on float64 numpy attention tensors A of shape
# (n_heads, T, T), causal (lower-triangular row-stochastic).
import numpy as np
import torch


def load_model(name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, output_attentions=True, attn_implementation="eager",
        dtype=torch.float32).eval()
    return tok, model


def wikitext_windows(n, stride, seq, tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="validation")
    pool = [t for t in ds["text"] if len(t.split()) > 40]
    texts = [pool[i * stride] for i in range(n) if i * stride < len(pool)]
    return [tok(t, return_tensors="pt", truncation=True, max_length=seq)
            for t in texts]


def humaneval_windows(n, stride, seq, tok):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    pool = [r["prompt"] + r["canonical_solution"] for r in ds]
    texts = [pool[(i * stride) % len(pool)] for i in range(n)]
    return [tok(t, return_tensors="pt", truncation=True, max_length=seq)
            for t in texts]


def attn_layer(out, l):
    return out.attentions[l].squeeze(0).numpy().astype(np.float64)


def generators(A):
    return (A - A.transpose(0, 2, 1)) / 2.0


def gnorms(G):
    return np.sqrt((G ** 2).sum((1, 2)))


def coup(G):
    P = np.einsum('aij,bjk->abik', G, G)
    K = P - P.transpose(1, 0, 2, 3)
    return np.sqrt((K ** 2).sum((2, 3)))


def r1(C, gn, iu):
    x, y = C[iu], np.outer(gn, gn)[iu]
    if x.std() < 1e-15 or y.std() < 1e-15:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def rho(G, gn):
    n = G.shape[0]
    V = G.reshape(n, -1)
    M = (V @ V.T) / np.outer(gn, gn)
    return float(np.linalg.eigvalsh(M)[-1] / n)


def shared_mode(G):
    """Top PC of vec(G_h): returns S (T,T), coefficients a, residual norms e,
    mean shared-energy fraction."""
    n = G.shape[0]
    V = G.reshape(n, -1)
    U, sv, Vh = np.linalg.svd(V, full_matrices=False)
    S = Vh[0]
    a = V @ S
    E = V - np.outer(a, S)
    e = np.sqrt((E ** 2).sum(1))
    gn = np.sqrt((V ** 2).sum(1))
    frac = float((a ** 2 / gn ** 2).mean())
    return S.reshape(G.shape[1], G.shape[2]), a, e, frac


def sink_column(A, min_rows=16):
    """Empirical sink column: modal argmax over heads of column mass.
    Columns with fewer than min_rows contributing rows are excluded, which
    removes the near-final-column artifact (a column receivable by only one
    or two rows can look concentrated from a single recency entry)."""
    n, T, _ = A.shape
    cmax = T - 1 - min_rows
    best = []
    for h in range(n):
        cm = np.zeros(max(cmax, 1))
        for c in range(max(cmax, 1)):
            rows = np.arange(max(1, c + 1), T)
            cm[c] = A[h, rows, c].mean()
        best.append(int(cm.argmax()))
    vals, counts = np.unique(best, return_counts=True)
    return int(vals[counts.argmax()])


def sink_generator(T, c):
    """Ideal causal sink generator at column c, unit Frobenius norm:
    A_sink rows i > c put all mass on column c (rows i <= c on the
    diagonal), G = (A - A^T)/2 normalized."""
    A = np.zeros((T, T))
    for i in range(T):
        if i > c:
            A[i, c] = 1.0
        else:
            A[i, i] = 1.0
    G = (A - A.T) / 2.0
    return G / np.sqrt((G ** 2).sum())


def col_mass(A, c):
    """Mean attention mass on column c, rows below c only, excluding row 0."""
    n, T, _ = A.shape
    rows = np.arange(max(1, c + 1), T)
    return float(A[:, rows, c].mean())


# --- surrogate families: per-head, per-row permutations of causal entries ---

def sur_plain(A, rg):
    S_ = A.copy()
    n, T, _ = A.shape
    for h in range(n):
        for i in range(1, T):
            S_[h, i, :i] = S_[h, i, rg.permutation(i)]
    return S_


def sur_sinkfix(A, rg, s=0):
    """Preserve column s (where it exists among causal off-diagonal entries)
    and the diagonal; permute the remaining causal off-diagonal entries."""
    S_ = A.copy()
    n, T, _ = A.shape
    for h in range(n):
        for i in range(1, T):
            cols = [c for c in range(i) if c != s]
            if len(cols) > 1:
                vals = S_[h, i, cols]
                S_[h, i, cols] = vals[rg.permutation(len(cols))]
    return S_


def sur_altsink2(A, rg):
    """Distinct target column per head: t_h = h + 1. Swap each row's causal
    off-diagonal max into t_h (when t_h < i), then permute the other
    off-diagonal entries. Rows with t_h >= i are plain-permuted."""
    S_ = A.copy()
    n, T, _ = A.shape
    for h in range(n):
        t = h + 1
        for i in range(1, T):
            row = S_[h, i, :i].copy()
            if t < i:
                jmax = int(row.argmax())
                row[jmax], row[t] = row[t], row[jmax]
                rest = [c for c in range(i) if c != t]
                vals = row[rest]
                row[rest] = vals[rg.permutation(len(rest))]
            else:
                row = row[rg.permutation(i)]
            S_[h, i, :i] = row
    return S_


def check_invariants(A, S_, preserve_cols=()):
    """Max abs deviations: row sums, per-head gnorm, sorted rows, preserved
    columns. All must be < 1e-9."""
    devs = {}
    devs["rowsum"] = float(np.abs(A.sum(-1) - S_.sum(-1)).max())
    devs["gnorm"] = float(np.abs(gnorms(generators(A)) - gnorms(generators(S_))).max())
    devs["sorted"] = float(np.abs(np.sort(A, -1) - np.sort(S_, -1)).max())
    for c in preserve_cols:
        n, T, _ = A.shape
        rows = np.arange(c + 1, T)
        if len(rows):
            devs[f"col{c}"] = float(np.abs(A[:, rows, c] - S_[:, rows, c]).max())
    devs["diag"] = float(np.abs(np.diagonal(A, axis1=1, axis2=2)
                                - np.diagonal(S_, axis1=1, axis2=2)).max())
    return devs


def zstats(C_real, C_plain_draws, iu):
    """Per-pair z of C_real against the plain-surrogate ensemble; returns
    median z over pairs."""
    stack = np.stack([C[iu] for C in C_plain_draws])
    mu, sd = stack.mean(0), stack.std(0) + 1e-12
    return float(np.median((C_real[iu] - mu) / sd)), mu, sd


def zmed_of(C, mu, sd, iu):
    return float(np.median((C[iu] - mu) / sd))
