# Instruments for converting multi-head attention to grouped-query attention
# (PREREGISTRATION24). A conversion partitions each layer's heads into groups
# of size g and replaces every head's key projection (weights and bias) by the
# group mean, and likewise its value projection (Ainslie et al. 2023, mean
# pooling); queries and output projections stay per head. The converted model
# is evaluated as is: it computes exactly what a grouped-query model with the
# pooled key and value heads computes.
#
# Optional alignment (Jin et al. 2025): before pooling, each head's keys and
# queries are rotated by the same orthogonal matrix and its values by another,
# with the inverse folded into the output projection, so the unmerged model is
# unchanged. Key rotations commute with the rotary embedding: on rotary
# dimensions they are 2-D rotations in each rotary plane (i, i + r/2), on the
# remaining dimensions any orthogonal matrix.
#
# Architectures: gpt2 (fused Conv1D c_attn), neox (fused, per-head interleaved
# query_key_value; partial rotary), opt (separate projections, query scaled
# before the attention call), llama/olmo (separate projections, full rotary,
# no bias) and phi (separate projections with bias, partial rotary).
import numpy as np
import torch
from transformers import AttentionInterface

CAP = {}
CAPTURE_LAYERS = None          # None: every layer


def capture_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Eager causal attention that records post-rotary queries, keys and values (float16, on the host)."""
    q, k, v = query.float(), key.float(), value.float()
    sc = scaling if scaling is not None else q.shape[-1] ** -0.5
    T = q.shape[-2]
    s = (q @ k.transpose(-1, -2)) * sc
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    w = torch.softmax(s, -1)
    if CAPTURE_LAYERS is None or module.layer_idx in CAPTURE_LAYERS:
        CAP.setdefault(module.layer_idx, []).append((q[0].half().cpu(), k[0].half().cpu(), v[0].half().cpu(), float(sc)))
    return (w @ v).to(value.dtype).transpose(1, 2).contiguous(), w


AttentionInterface.register("gqa_capture", capture_attention)


class Arch:
    """Row views of each head's query, key and value projections and of its output slice."""

    def __init__(self, model):
        c = model.config
        self.model = model
        if hasattr(model, "transformer"):
            self.kind, self.blocks = "gpt2", model.transformer.h
        elif hasattr(model, "gpt_neox"):
            self.kind, self.blocks = "neox", model.gpt_neox.layers
        elif c.model_type == "opt":
            self.kind, self.blocks = "opt", model.model.decoder.layers
        elif c.model_type in ("llama", "olmo"):
            self.kind, self.blocks = "llama", model.model.layers
        elif c.model_type == "phi":
            self.kind, self.blocks = "phi", model.model.layers
        else:
            raise ValueError(c.model_type)
        self.L = len(self.blocks)
        self.n = getattr(c, "num_attention_heads", None) or c.n_head
        self.d = getattr(c, "hidden_size", None) or c.n_embd
        self.hd = self.d // self.n
        nkv = getattr(c, "num_key_value_heads", None)
        assert nkv in (None, self.n), "multi-head models only"
        if self.kind in ("gpt2", "opt"):
            self.r = 0
        elif self.kind == "llama":
            self.r = self.hd
        else:
            rp = getattr(c, "rope_parameters", None) or {}
            frac = rp.get("partial_rotary_factor", None) if isinstance(rp, dict) else None
            if frac is None:
                frac = getattr(c, "rotary_pct", None) or getattr(c, "partial_rotary_factor", None)
            self.r = int(self.hd * frac)

    # ---- projections as (weight with output features as rows, bias, row indices) ----
    def _fused(self, l):
        b = self.blocks[l]
        mod = b.attn.c_attn if self.kind == "gpt2" else b.attention.query_key_value
        W = mod.weight.T if self.kind == "gpt2" else mod.weight
        return W, mod.bias

    def proj(self, l, h, which):
        """which in 'qkv'."""
        hd, d = self.hd, self.d
        if self.kind in ("gpt2", "neox"):
            W, b = self._fused(l)
            if self.kind == "gpt2":
                off = {"q": 0, "k": d, "v": 2 * d}[which] + h * hd
            else:
                off = h * 3 * hd + {"q": 0, "k": hd, "v": 2 * hd}[which]
            return W, b, torch.arange(off, off + hd, device=W.device)
        a = self.blocks[l].self_attn
        mod = {"q": a.q_proj, "k": a.k_proj, "v": a.v_proj}[which]
        return mod.weight, mod.bias, torch.arange(h * hd, (h + 1) * hd, device=mod.weight.device)

    def out_module(self, l):
        b = self.blocks[l]
        if self.kind == "gpt2":
            return b.attn.c_proj
        if self.kind == "neox":
            return b.attention.dense
        if self.kind == "opt":
            return b.self_attn.out_proj
        if self.kind == "llama":
            return b.self_attn.o_proj
        return b.self_attn.dense

    def out_slice(self, l, h):
        """(hd, d) matrix mapping head h's output to the residual stream (row-vector convention)."""
        W = self.out_module(l).weight.detach().float()
        cols = slice(h * self.hd, (h + 1) * self.hd)
        return W[cols, :] if self.kind == "gpt2" else W[:, cols].T

    def attn_params(self, l):
        """Every parameter tensor a conversion may change in layer l."""
        out = []
        if self.kind in ("gpt2", "neox"):
            W, b = self._fused(l); m = self.blocks[l].attn.c_attn if self.kind == "gpt2" else self.blocks[l].attention.query_key_value
            out += [m.weight] + ([m.bias] if m.bias is not None else [])
        else:
            a = self.blocks[l].self_attn
            for m in (a.q_proj, a.k_proj, a.v_proj):
                out += [m.weight] + ([m.bias] if m.bias is not None else [])
        o = self.out_module(l)
        out += [o.weight]
        return out


def snapshot(arch):
    return {l: [p.detach().clone() for p in arch.attn_params(l)] for l in range(arch.L)}


@torch.no_grad()
def restore(arch, snap):
    for l in range(arch.L):
        for p, s in zip(arch.attn_params(l), snap[l]):
            p.copy_(s)


# ---------------- rotations ----------------
def procrustes(X, Y):
    """Orthogonal R minimizing ||X R - Y|| (rows are samples)."""
    U, _, Vt = torch.linalg.svd(X.T @ Y)
    return U @ Vt


def rope_procrustes(X, Y, r):
    """Orthogonal R minimizing ||X R - Y|| among matrices that commute with a rotary embedding on the
    first r dimensions (rotate-half pairing: planes (i, i + r/2), rotations only), any orthogonal
    matrix on the rest."""
    k = X.shape[1]
    R = torch.zeros(k, k, dtype=X.dtype, device=X.device)
    h = r // 2
    if r:
        i = torch.arange(h, device=X.device)
        x1, x2, y1, y2 = X[:, i], X[:, i + h], Y[:, i], Y[:, i + h]
        A = (x1 * y1 + x2 * y2).sum(0); B = (x1 * y2 - x2 * y1).sum(0)
        th = torch.atan2(B, A); c, s = torch.cos(th), torch.sin(th)
        R[i, i] = c; R[i, i + h] = s; R[i + h, i] = -s; R[i + h, i + h] = c
    if r < k:
        R[r:, r:] = procrustes(X[:, r:], Y[:, r:])
    return R


def gpa(Xs, solver, iters=10):
    """Generalized Procrustes: rotations aligning every matrix in Xs to their common mean."""
    Rs = [torch.eye(Xs[0].shape[1], dtype=Xs[0].dtype, device=Xs[0].device) for _ in Xs]
    M = Xs[0]
    for _ in range(iters):
        Rs = [solver(X, M) for X in Xs]
        M_new = torch.stack([X @ R for X, R in zip(Xs, Rs)]).mean(0)
        if (M_new - M).norm() <= 1e-6 * M.norm():
            M = M_new; break
        M = M_new
    return Rs


def unit_rows(X):
    return X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# The same solvers from cross-products C = X^T Y alone (float64 host arrays): GPA then runs on the
# (hd x hd) Gram blocks X_a^T X_b of the group's caches, never touching the token dimension.
def procrustes_C(Cm):
    U, _, Vt = np.linalg.svd(Cm)
    return U @ Vt


def rope_procrustes_C(Cm, r):
    k = Cm.shape[0]
    R = np.zeros((k, k))
    h = r // 2
    if r:
        i = np.arange(h)
        A = Cm[i, i] + Cm[i + h, i + h]; B = Cm[i, i + h] - Cm[i + h, i]
        th = np.arctan2(B, A); c, s = np.cos(th), np.sin(th)
        R[i, i] = c; R[i, i + h] = s; R[i + h, i] = -s; R[i + h, i + h] = c
    if r < k:
        R[r:, r:] = procrustes_C(Cm[r:, r:])
    return R


def gpa_gram(Cb, solver, iters=10, tol=1e-9):
    """gpa() computed from Gram blocks Cb[a][b] = X_a^T X_b: start from the first member as target,
    rotate every member onto the current mean, recompute the mean, repeat."""
    g = len(Cb)
    XtM = [Cb[a][0] for a in range(g)]
    R = None
    for _ in range(iters):
        Rn = [solver(XtM[a]) for a in range(g)]
        XtM = [sum(Cb[a][b] @ Rn[b] for b in range(g)) / g for a in range(g)]
        if R is not None and max(np.abs(Rn[a] - R[a]).max() for a in range(g)) < tol:
            R = Rn; break
        R = Rn
    return R


@torch.no_grad()
def apply_rotation(arch, l, h, Rk, Sv):
    """Rotate head h's queries and keys by Rk and its values by Sv; fold Sv^T into the output slice.
    Function-preserving: q.k and the head's residual contribution are unchanged."""
    for which, R in (("q", Rk), ("k", Rk), ("v", Sv)):
        W, b, idx = arch.proj(l, h, which)
        R = R.to(W.device, W.dtype)
        W[idx] = R.T @ W[idx]
        if b is not None:
            b[idx] = R.T @ b[idx]
    o = arch.out_module(l).weight; cols = torch.arange(h * arch.hd, (h + 1) * arch.hd, device=o.device)
    S = Sv.to(o.device, o.dtype)
    if arch.kind == "gpt2":
        o[cols] = S.T @ o[cols]
    else:
        o[:, cols] = o[:, cols] @ S


@torch.no_grad()
def pool_group(arch, l, group):
    """Mean-pool the key and value projections (weights and biases) of the heads in group."""
    for which in ("k", "v"):
        rows = [arch.proj(l, h, which) for h in group]
        W, b = rows[0][0], rows[0][1]
        m = torch.stack([W[idx] for _, _, idx in rows]).mean(0)
        mb = torch.stack([b[idx] for _, _, idx in rows]).mean(0) if b is not None else None
        for _, _, idx in rows:
            W[idx] = m
            if b is not None:
                b[idx] = mb


def windows_c4(tok, n, T):
    """The first n documents of C4 validation shard 0 with at least T tokens, truncated to T."""
    import gzip, json
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    out = []
    with gzip.open(p, "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
            if len(ids) >= T:
                out.append(torch.tensor(ids[:T])[None])
                if len(out) == n:
                    break
    return torch.cat(out)
