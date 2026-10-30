# Bisection for the exploratory follow-up to round 28 (not preregistered). On Llama-2-7B converted to 16 key-value
# heads, the weighted fusion reaches WikiText-2 perplexity 16.8 on pairs matched by its own residual (round 28) and
# 12,142 on pairs matched by CAGA's residual (llama2_followup_rerun.json), although the two pairings are within about
# 2 percent of each other in the fusion's residual and within 1 percent in single-layer perplexity in every layer
# (llama2_followup_diag.json). This locates the difference:
#   cumulative     layers 0..k converted, for every k, under each pairing
#   swaps          every layer on one pairing except layer k on the other, both ways
#   profile        the residual stream after every layer, every layer converted under each pairing, against the
#                  unconverted model: relative change at the first position and at the others, and the largest
#                  activation at the first position (where Llama-2 keeps its massive activations)
# Each layer is converted exactly as fuse_kv.convert converts it (second moments of the unconverted model); the
# unconverted weights are read back from the checkpoint files, cast as from_pretrained casts them (checked).
#   python alignment_study/llama2_followup_bisect.py
import json
import shutil
import sys
import tempfile
import time
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME, REVISION = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
OUT = "alignment_study/llama2_followup_bisect.json"
N_WIN, T_PROFILE = 4, 1024
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
torch.set_grad_enabled(False)


def original_reader():
    """The unconverted attention weights of layer l, read from the checkpoint files."""
    index = json.load(open(hf_hub_download(NAME, "model.safetensors.index.json", revision=REVISION)))["weight_map"]
    files = {}

    def get(l):
        out = []
        for p in PROJ:
            key = f"model.layers.{l}.self_attn.{p}.weight"
            if index[key] not in files:
                files[index[key]] = safe_open(hf_hub_download(NAME, index[key], revision=REVISION), framework="pt", device="cpu")
            out.append(files[index[key]].get_tensor(key).to(torch.bfloat16))
        return out
    return get


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    pairings = {"fuse_matched": json.load(open("alignment_study/llama2/Llama-2-7b-hf.json"))["pairing"]["paired"],
                "caga_matched": json.load(open("alignment_study/llama2_followup.json"))["groups"]}
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    orig = original_reader()
    get_layer = lambda l: [p.detach().cpu().clone() for p in kv.params(l)]

    def set_layer(l, ts):
        for p, t in zip(kv.params(l), ts):
            p.copy_(t.to(p.device))
    for l in (0, kv.L - 1):
        assert all(torch.equal(a, b) for a, b in zip(get_layer(l), orig(l))), f"checkpoint read differs from the loaded model in layer {l}"
    S, ntok = K.second_moments(model, kv, C.windows_c4(tok, 128, 2048), DEV)
    log(f"second moments from {ntok} tokens")

    def fuse_layer(l, groups):
        """fuse_kv.convert's body for one layer, method 'fuse'."""
        S64 = S[l].double(); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
        for G in groups[l]:
            K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)

    win = Z.wikitext2_windows(tok, T=2048)
    xprof, win = win[:1, :T_PROFILE], win[:N_WIN]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)
    body, _ = Z._parts(model)

    def hidden():
        hs = {}

        def mk(l):
            def hook(m, i, o):
                hs[l] = (o[0] if isinstance(o, tuple) else o)[0].to(torch.bfloat16).cpu()
            return hook
        hooks = [kv.blocks[l].register_forward_hook(mk(l)) for l in range(kv.L)]
        body(input_ids=xprof.to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
        return hs
    H0 = hidden()

    def profile():
        H, rows = hidden(), []
        for l in range(kv.L):
            a, b = H[l].float(), H0[l].float()
            rows.append({"layer": l, "rel_change_first": float((a[0] - b[0]).norm() / b[0].norm()),
                         "rel_change_rest": float((a[1:] - b[1:]).norm() / b[1:].norm()),
                         "max_abs_first": float(a[0].abs().max()), "max_abs_first_unconverted": float(b[0].abs().max()),
                         "median_norm_rest": float(a[1:].norm(dim=1).median()), "median_norm_rest_unconverted": float(b[1:].norm(dim=1).median())})
        return rows

    res = {"model": NAME, "revision": REVISION, "exploratory": True, "windows": N_WIN, "window_tokens": 2048,
           "profile_tokens": T_PROFILE, "pairings_from": {"fuse_matched": "alignment_study/llama2/Llama-2-7b-hf.json",
                                                          "caga_matched": "alignment_study/llama2_followup.json"},
           "base_ppl": ppl(), "cumulative": {}, "profile": {}}
    dump = lambda: json.dump(res, open(OUT, "w"), indent=1)
    log(f"unconverted: ppl {res['base_ppl']:.3f} on {N_WIN} windows")

    # cumulative conversion; round 28's pairs are stashed on disk, CAGA's in memory, for the swaps
    stash = tempfile.mkdtemp(prefix="llama2_bisect_")
    caga = {}
    for name in ("fuse_matched", "caga_matched"):
        res["cumulative"][name] = []
        for k in range(kv.L):
            t = time.time(); fuse_layer(k, pairings[name])
            if name == "fuse_matched":
                torch.save(get_layer(k), f"{stash}/{k}.pt")
            else:
                caga[k] = get_layer(k)
            res["cumulative"][name].append(ppl()); dump()
            log(f"{name}: layers 0-{k} converted, ppl {res['cumulative'][name][-1]:.3f} ({time.time() - t:.0f}s)")
        res["profile"][f"all_{name}"] = profile(); dump()
        for l in range(kv.L):
            set_layer(l, orig(l))
        res[f"restored_after_{name}"] = ppl(); dump()
        log(f"restored: ppl {res[f'restored_after_{name}']:.3f}")
    del S

    # swaps: every layer on CAGA's pairs except layer k on round 28's, then the reverse
    for l in range(kv.L):
        set_layer(l, caga[l])
    res["all_caga_check"] = ppl()
    res["caga_except_k_on_fuse_pairs"] = []
    for k in range(kv.L):
        set_layer(k, torch.load(f"{stash}/{k}.pt"))
        res["caga_except_k_on_fuse_pairs"].append(ppl()); dump()
        set_layer(k, caga[k])
        log(f"CAGA's pairs except layer {k} on round 28's: ppl {res['caga_except_k_on_fuse_pairs'][-1]:.3f}")
    for l in range(kv.L):
        set_layer(l, torch.load(f"{stash}/{l}.pt"))
    res["all_fuse_check"] = ppl()
    res["fuse_except_k_on_caga_pairs"] = []
    for k in range(kv.L):
        set_layer(k, caga[k])
        res["fuse_except_k_on_caga_pairs"].append(ppl()); dump()
        set_layer(k, torch.load(f"{stash}/{k}.pt"))
        log(f"round 28's pairs except layer {k} on CAGA's: ppl {res['fuse_except_k_on_caga_pairs'][-1]:.3f}")
    shutil.rmtree(stash, ignore_errors=True)
    dump()
    log("BISECT_DONE")
