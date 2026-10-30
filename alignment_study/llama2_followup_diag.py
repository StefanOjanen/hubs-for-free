# Diagnostic for the exploratory follow-up to round 28 (not preregistered). On pairs matched by CAGA's own
# residual, both CAGA's construction and the weighted fusion reached WikiText-2 perplexity above 9,000, while the
# fusion gave 16.8 on its own matched pairs and 19.7 on neighbouring pairs. This locates the damage: for every
# layer, the weighted fusion residual (kvfuse.pair_costs) summed over the layer's pairs under each pairing, and the
# perplexity when only that layer is converted (weighted fusion), on CAGA's pairs and on round 28's.
#   python alignment_study/llama2_followup_diag.py
import json
import os
import shutil
import sys
import tempfile
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, REVISION = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
OUT = "alignment_study/llama2_followup_diag.json"
N_WIN = 8
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pairings = {"caga_matched": json.load(open("alignment_study/llama2_followup.json"))["groups"],
                "fuse_matched": json.load(open("alignment_study/llama2/Llama-2-7b-hf.json"))["pairing"]["paired"]}
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    S, ntok = K.second_moments(model, kv, C.windows_c4(tok, 128, 2048), DEV)
    log(f"second moments from {ntok} tokens")
    win = Z.wikitext2_windows(tok, T=2048)[:N_WIN]
    base = Z.perplexity(model, win, DEV, bs=2)
    snapdir = tempfile.mkdtemp(prefix="llama2_diag_snapshot_")
    for l in range(kv.L):
        torch.save([p.detach().cpu() for p in kv.params(l)], f"{snapdir}/{l}.pt")
    res = {"model": NAME, "revision": REVISION, "exploratory": True, "windows": N_WIN, "base_ppl": base, "layers": []}
    for l in range(kv.L):
        cost = kvfuse.pair_costs(kv, l, S[l].double())
        row = {"layer": l}
        for name, groups in pairings.items():
            row[f"residual_{name}"] = float(sum(cost[i, j] for i, j in groups[l]))
            S64 = S[l].double(); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]   # this layer only
            for G in groups[l]:
                K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)
            row[f"ppl_only_this_layer_{name}"] = Z.perplexity(model, win, DEV, bs=2)
            for p, s_ in zip(kv.params(l), torch.load(f"{snapdir}/{l}.pt")):
                p.copy_(s_.to(p.device))
        res["layers"].append(row)
        log(f"layer {l:2d}: residual caga {row['residual_caga_matched']:.3f} fuse {row['residual_fuse_matched']:.3f}; "
            f"ppl converting only this layer: caga pairs {row['ppl_only_this_layer_caga_matched']:.2f}, fuse pairs {row['ppl_only_this_layer_fuse_matched']:.2f}")
        json.dump(res, open(OUT, "w"), indent=1)
    shutil.rmtree(snapdir, ignore_errors=True)
    log("DIAG_DONE")
