# Which unit fires, and why (exploratory follow-up to round 28, not preregistered). The mechanism check
# (llama2_followup_mechanism.py) found that merging key-value heads 8 and 18 of Llama-2-7B's layer 1, once layer 0 is
# converted, gives ordinary tokens the large activation in residual dimension 2533 that the model otherwise keeps for
# the first position (perplexity 6,008). Dimension 2533 is written by layer 1's MLP; Yu et al. (2024, "The super
# weight in large language models") locate one weight, down_proj[2533, 7890] of layer 1, that produces it. This
# records, on one window, the activation of the MLP neurons that write dimension 2533 most strongly, the
# contribution of every attention head of layer 1 to those neurons' gate and up inputs, and the attention of query
# heads 8 and 18 (to the first position, to themselves, to newline tokens), unconverted and converted.
#   python alignment_study/llama2_followup_neuron.py
import json
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
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

NAME, REVISION = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
OUT = "alignment_study/llama2_followup_neuron.json"
DIM, PAIR, T = 2533, [8, 18], 2048
torch.set_grad_enabled(False)


class Stop(Exception):
    pass


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    caga = json.load(open("alignment_study/llama2_followup.json"))["groups"]
    fuse = json.load(open("alignment_study/llama2/Llama-2-7b-hf.json"))["pairing"]["paired"]
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)

    S, n = {0: None, 1: None}, 0                       # second moments of layers 0 and 1, as in the mechanism check
    def mk(l):
        def pre(mod, inp):
            x = inp[0].detach().reshape(-1, kv.d).float()
            m = (x.T @ x).float().cpu()
            S[l] = m if S[l] is None else S[l] + m
        return pre
    def stop(mod, i, o):
        raise Stop
    hs = [kv.attn(l).k_proj.register_forward_pre_hook(mk(l)) for l in (0, 1)] + [kv.blocks[1].register_forward_hook(stop)]
    cal = C.windows_c4(tok, 128, 2048)
    for w in range(cal.shape[0]):
        try:
            model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        except Stop:
            pass
        n += cal.shape[1]
    for h in hs:
        h.remove()
    S = {l: S[l] / n for l in S}
    snap = {l: [p.detach().cpu().clone() for p in kv.params(l)] for l in (0, 1)}
    fit = {}
    def fuse_groups(l, groups):
        if l not in fit:
            S64 = S[l].double(); roots = F.layer_roots(S64, kv.d); fit[l] = (S64, roots, roots[0] @ roots[0])
        S64, roots, Lsq = fit[l]
        for G in groups:
            K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)
    def restore(l):
        for p, s_ in zip(kv.params(l), snap[l]):
            p.copy_(s_.to(p.device))

    blk = kv.blocks[1]
    mlp, a1, hd = blk.mlp, kv.attn(1), kv.hd
    wd = mlp.down_proj.weight[DIM].float().cpu()
    top_neurons = wd.abs().topk(3).indices.tolist()
    log(f"layer 1 MLP neurons writing dimension {DIM} most strongly: {top_neurons} (weights {[round(float(wd[j]), 2) for j in top_neurons]})")
    x = Z.wikitext2_windows(tok, T=T)[:1]
    toks = tok.convert_ids_to_tokens(x[0].tolist())
    newline = torch.tensor([t == "<0x0A>" for t in toks])

    def record():
        cap = {}
        hk = [blk.input_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("xa", o[0].float())),
              a1.o_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("z", i[0][0].float())),
              blk.post_attention_layernorm.register_forward_hook(lambda m, i, o: cap.update(hmid=i[0][0].float(), xm=o[0].float())),
              kv.blocks[0].register_forward_hook(lambda m, i, o: cap.__setitem__("hin", (o[0] if isinstance(o, tuple) else o)[0].float()))]
        model.model(input_ids=x.to(DEV), use_cache=False)
        for h in hk:
            h.remove()
        g = mlp.gate_proj.weight[top_neurons].float(); u = mlp.up_proj.weight[top_neurons].float()
        act = torch.nn.functional.silu(cap["xm"] @ g.T) * (cap["xm"] @ u.T)                      # (T, 3)
        # contributions to the first neuron's gate input: g . (w_ln * part) / rms(h_mid), parts = residual and heads
        w_ln = blk.post_attention_layernorm.weight.float()
        eps = blk.post_attention_layernorm.variance_epsilon
        rms = (cap["hmid"].pow(2).mean(1, keepdim=True) + eps).sqrt()
        gw = g[0] * w_ln
        Wo = a1.o_proj.weight.float()
        heads = torch.stack([(cap["z"][:, h * hd:(h + 1) * hd] @ (Wo[:, h * hd:(h + 1) * hd].T @ gw)) / rms[:, 0] for h in range(kv.n)], 1)  # (T, n)
        resid = (cap["hin"] @ gw) / rms[:, 0]
        # attention of query heads 8 and 18 (their own key-value heads: the merged pair shares one)
        pos = torch.arange(T, device=DEV)[None]
        cos, sin = model.model.rotary_emb(cap["xa"][None], position_ids=pos)
        q = a1.q_proj(cap["xa"][None].to(a1.q_proj.weight.dtype)).view(1, T, kv.n, hd).transpose(1, 2)
        k = a1.k_proj(cap["xa"][None].to(a1.k_proj.weight.dtype)).view(1, T, kv.nkv, hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        att = {}
        mask = torch.ones(T, T, dtype=torch.bool, device=DEV).tril()
        for h in PAIR:
            s = (q[0, h].float() @ k[0, h].float().T) / hd ** 0.5
            p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1).cpu()
            att[h] = {"to_first": float(p[1:, 0].mean()), "to_self": float(torch.diagonal(p)[1:].mean()),
                      "to_newlines": float(p[1:][:, newline].sum(1).mean())}
        a = act[:, 0].abs().cpu()
        rest = torch.arange(T) > 1
        return {"neuron_abs_median_rest": float(a[rest].median()), "neuron_abs_p90_rest": float(a[rest].quantile(0.9)),
                "neuron_first": float(act[0, 0]), "neuron_positions_above_10": int((a[rest] > 10).sum()),
                "gate_input_median_rest": {"residual": float(resid.cpu()[rest].median()),
                                           **{f"head{h}": float(heads[:, h].cpu()[rest].median()) for h in range(kv.n)}},
                "attention": att, "other_neurons_abs_median_rest": [float(act[:, j].abs().cpu()[rest].median()) for j in (1, 2)]}

    res = {"model": NAME, "revision": REVISION, "exploratory": True, "dim": DIM, "neurons": top_neurons,
           "neuron_weights": [float(wd[j]) for j in top_neurons], "window_tokens": T, "configs": {}}
    configs = {"unconverted": [], "pair_8_18_alone": [(1, [PAIR])], "layer0_caga": [(0, caga[0])],
               "layer0_caga_pair_8_18": [(0, caga[0]), (1, [PAIR])], "layer0_fuse_pair_8_18": [(0, fuse[0]), (1, [PAIR])],
               "layer0_caga_layer1_fuse": [(0, caga[0]), (1, fuse[1])]}
    for name, steps in configs.items():
        for l, groups in steps:
            fuse_groups(l, groups)
        r = record()
        r["ppl_4_windows"] = Z.perplexity(model, Z.wikitext2_windows(tok, T=2048)[:4], DEV, bs=2)
        res["configs"][name] = r
        top = sorted(r["gate_input_median_rest"].items(), key=lambda kv_: -abs(kv_[1]))[:5]
        log(f"{name}: ppl {r['ppl_4_windows']:.3f}; neuron {top_neurons[0]} |activation| median over ordinary tokens {r['neuron_abs_median_rest']:.3f} "
            f"(p90 {r['neuron_abs_p90_rest']:.3f}, {r['neuron_positions_above_10']} above 10, first {r['neuron_first']:.1f}); "
            f"largest gate-input parts {[(k_, round(v, 3)) for k_, v in top]}; attention {r['attention']}")
        for l in sorted({l for l, _ in steps}):
            restore(l)
        json.dump(res, open(OUT, "w"), indent=1)
    log("NEURON_DONE")
