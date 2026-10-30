# Exploratory follow-up to round 28 (not preregistered; run after the registered run and reported as exploratory).
# Round 28's reimplementation of CAGA's construction gave perplexity 323 on the pairs matched by the weighted
# fusion's residual, against 25.1 published for CAGA on its own pairing, which the paper defines through another
# component of its method (GSA-II supports) and two similarity terms without formulas. This asks whether CAGA's
# construction does much better on pairs chosen for it: the minimum-cost perfect matching under CAGA's own
# residual (its covariance-metric key fit, a complex scalar per rotary plane, plus the Frobenius residual of the
# rank-d_h SVD of the stacked value projections, each relative to the layer's total). The weighted fusion is run on
# the same pairs for comparison. Llama-2-7B as in round 28 (same revision, calibration and WikiText-2 windows);
# perplexity only, at 2,048 and 4,096 tokens.
#   python alignment_study/llama2_caga_followup.py
import json
import os
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

argv0 = sys.argv
sys.argv = [argv0[0], "--dev=NousResearch/Llama-2-7b-hf"]            # import the frozen round's functions unchanged
import importlib.util
spec = importlib.util.spec_from_file_location("R", "alignment_study/llama2_round.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
sys.argv = argv0
NAME, REVISION = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
OUT = "alignment_study/llama2_followup.json"
torch.set_grad_enabled(False)


def caga_pair_costs(kv, l, S):
    """CAGA's own residual for every pair: unweighted key fit in the S metric (per rotary plane, rank one over
    complex numbers) over the layer's total key energy, plus the stacked-value SVD residual over the layer's
    total value energy (Frobenius, no metric)."""
    a, hd, half, n = kv.attn(l), kv.hd, kv.hd // 2, kv.nkv
    Wk, Wv = F.host64(a.k_proj.weight), F.host64(a.v_proj.weight)
    GK, GW = Wk @ S @ Wk.T, Wv @ Wv.T
    x = lambda i: torch.arange(i * hd, i * hd + half)
    y = lambda i: torch.arange(i * hd + half, (i + 1) * hd)
    kd = torch.stack([GK[x(i), x(i)] + GK[y(i), y(i)] for i in range(n)])
    Ek = float(kd.sum()); Ev = float(sum(GW[i * hd:(i + 1) * hd, i * hd:(i + 1) * hd].trace() for i in range(n)))
    cost = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            re = GK[x(i), x(j)] + GK[y(i), y(j)]; im = GK[y(i), x(j)] - GK[x(i), y(j)]
            lo = (kd[i] + kd[j]) / 2 - torch.sqrt(((kd[i] - kd[j]) / 2) ** 2 + re ** 2 + im ** 2)
            idx = torch.cat([torch.arange(i * hd, (i + 1) * hd), torch.arange(j * hd, (j + 1) * hd)])
            lv = torch.linalg.eigvalsh(GW[idx][:, idx])[:hd].clamp_min(0).sum()
            cost[i, j] = cost[j, i] = float(lo.clamp_min(0).sum()) / Ek + float(lv) / Ev
    return cost


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    cal = C.windows_c4(tok, 128, 2048)
    S, ntok = K.second_moments(model, kv, cal, DEV)
    log(f"second moments from {ntok} tokens")
    import tempfile as _tf
    sdir = _tf.mkdtemp(prefix="llama2_followup_S_")                    # second moments on disk, one layer in memory at a time

    class OnDisk(dict):
        def __getitem__(self, l):
            return torch.load(f"{sdir}/{l}.pt")
    for l in list(S):
        torch.save(S[l], f"{sdir}/{l}.pt")
    S = OnDisk({l: None for l in range(kv.L)})
    del cal
    groups = [R.min_cost_pairs(caga_pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    r28 = json.load(open("alignment_study/llama2/Llama-2-7b-hf.json"))
    same = sum(sum(1 for p in groups[l] if p in r28["pairing"]["paired"][l]) for l in range(kv.L))
    log(f"CAGA-residual matching chosen; {same} of {kv.L * kv.nkv // 2} pairs also in the fusion-residual matching")
    win = {T: Z.wikitext2_windows(tok, T=T) for T in (2048, 4096)}
    import shutil, tempfile
    snapdir = tempfile.mkdtemp(prefix="llama2_followup_snapshot_")   # attention weights on disk, not in memory
    for l in range(kv.L):
        torch.save([p.detach().cpu() for p in kv.params(l)], f"{snapdir}/{l}.pt")

    def restore():
        for l in range(kv.L):
            for p, s_ in zip(kv.params(l), torch.load(f"{snapdir}/{l}.pt")):
                p.copy_(s_.to(p.device))
    res = {"model": NAME, "revision": REVISION, "exploratory": True, "pairs_shared_with_round28_matching": same, "groups": groups, "evals": {}}
    for construction in ("caga", "fuse"):
        t = time.time(); R.convert(kv, S, groups, construction)
        r = {f"ppl_{T}": Z.perplexity(model, win[T], DEV, bs=max(1, 4096 // T)) for T in (2048, 4096)}
        res["evals"][f"{construction}/caga_matched"] = r
        log(f"{construction} on pairs matched by CAGA's residual: ppl 2048 {r['ppl_2048']:.2f}, 4096 {r['ppl_4096']:.2f} ({time.time() - t:.0f}s)")
        restore()
        json.dump(res, open(OUT, "w"), indent=1)
    shutil.rmtree(snapdir, ignore_errors=True); shutil.rmtree(sdir, ignore_errors=True)
    log("FOLLOWUP_DONE")
