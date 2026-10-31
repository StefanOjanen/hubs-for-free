# Post hoc correction to round 31 (not preregistered, not scored). attnw_round.moments accumulates S and S_att
# without the ones column that fuse_kv.second_moments appends when a layer's projections carry biases, so
# fuse_kv.fuse_kv_group took Qwen2.5-7B for an unbiased model: it fitted the weights and left every bias in the old
# basis (plain arm 39,381 against round 30's 23.87 with the same pairs, calibration and windows; the four unbiased
# models reproduce round 30's fused values to four decimals). This reruns Qwen2.5-7B with the column appended to both
# moments (the attention weight multiplies it like every other entry); everything else is attnw_round.run_model. It
# then recomputes A2's pooled ratio with the corrected Qwen2.5-7B and over the four unbiased models alone, both
# labelled post hoc; the registered result stays alignment_study/attnw_results.json.
#   python alignment_study/attnw_biasfix.py
import gc
import json
import math
import os
import sys
import time
import torch

sys.argv.append("--dev=Qwen/Qwen2.5-7B")                      # attnw_round refuses to import without a mode
sys.path.insert(0, "alignment_study")
import attnw_round as R
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

NAME, OUT = "Qwen/Qwen2.5-7B", "alignment_study/attnw_correction"
ROUND30_FUSED = 23.8729                                         # alignment_study/balance/Q_Qwen2.5-7B.json, "fused"


def moments(model, kv, cal):
    """attnw_round.moments with the bias column: S and S_att of [x, 1] in every layer whose projections carry biases."""
    d, L, hd, n = kv.d, kv.L, kv.hd, kv.n
    S = {l: torch.zeros(d + kv.has_bias(l), d + kv.has_bias(l)) for l in range(L)}
    Sa = {l: torch.zeros(d + kv.has_bias(l), d + kv.has_bias(l)) for l in range(L)}
    conc = {l: {"entropy": 0.0, "first": 0.0} for l in range(L)}
    lin = torch.nn.functional.linear
    for w in range(cal.shape[0]):
        T = cal.shape[1]
        pos = torch.arange(T, device=R.DEV)[None]
        rows = torch.arange(R.QSTRIDE - 1, T, R.QSTRIDE, device=R.DEV)
        mask = torch.ones(T, T, dtype=torch.bool, device=R.DEV).tril()[rows]

        def mk(l):
            def pre(mod, inp):
                x = inp[0][0].float()
                a = kv.attn(l)
                cos, sin = model.model.rotary_emb(x[None], position_ids=pos)
                fb = lambda m: m.bias.float() if m.bias is not None else None
                q = lin(x, a.q_proj.weight.float(), fb(a.q_proj)).view(1, T, n, hd).transpose(1, 2)
                k = lin(x, a.k_proj.weight.float(), fb(a.k_proj)).view(1, T, kv.nkv, hd).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                recv = torch.zeros(T, device=x.device)
                for h in range(n):
                    p = torch.softmax(((q[0, h][rows] @ k[0, h // kv.rep].T) / hd ** 0.5).masked_fill(~mask, float("-inf")), -1)
                    recv += p.sum(0)
                share = recv / recv.sum()
                conc[l]["entropy"] += float(-(share * share.clamp_min(1e-30).log()).sum() / math.log(T))
                conc[l]["first"] += float(share[0])
                wgt = share * T
                if kv.has_bias(l):
                    x = torch.cat([x, torch.ones(T, 1, device=x.device)], 1)
                S[l] += (x.T @ x).cpu()
                Sa[l] += ((x * wgt[:, None]).T @ x).cpu()
            return pre
        hooks = [kv.attn(l).k_proj.register_forward_pre_hook(mk(l)) for l in range(L)]
        model(input_ids=cal[w:w + 1].to(R.DEV), use_cache=False)
        for h in hooks:
            h.remove()
    ntok = cal.shape[0] * cal.shape[1]
    S = {l: v / ntok for l, v in S.items()}
    Sa = {l: v / ntok for l, v in Sa.items()}
    conc = {l: {k_: v / cal.shape[0] for k_, v in c.items()} for l, c in conc.items()}
    return S, Sa, conc


def pooled(per):
    num = sum(math.log(v["attention"] / v["base"]) for v in per.values())
    den = sum(math.log(v["plain"] / v["base"]) for v in per.values())
    return num / den


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/Qwen2.5-7B.json"
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    if not os.path.exists(path):
        R.moments = moments
        res = R.run_model(NAME, log)
        res.update({"registered": None, "post_hoc_correction": "bias column appended to S and S_att", "round30_fused": ROUND30_FUSED})
        json.dump(res, open(path, "w"), indent=1)
        gc.collect()
    q = json.load(open(path))
    log(f"  corrected plain arm {q['plain']['ppl']:.4f} against round 30's {ROUND30_FUSED} (same pairs, calibration and windows)")
    names = ["OLMo-1B-hf", "h2o-danube-1.8b-base", "Mistral-7B-Instruct-v0.2", "Llama-2-7b-hf"]
    per = {m: {k: (r["base"] if k == "base" else r[k]["ppl"]) for k in ("base", "plain", "attention")}
           for m, r in ((m, json.load(open(f"alignment_study/attnw/{m}.json"))) for m in names)}
    four = pooled(per)
    per["Qwen2.5-7B"] = {"base": q["base"], "plain": q["plain"]["ppl"], "attention": q["attention"]["ppl"]}
    summary = {"post_hoc": True, "registered_result": "alignment_study/attnw_results.json",
               "qwen_corrected": per["Qwen2.5-7B"], "qwen_plain_reproduces_round30": abs(q["plain"]["ppl"] - ROUND30_FUSED) < 0.01,
               "A2_pooled_ratio_with_corrected_qwen": pooled(per), "A2_pooled_ratio_four_unbiased_models": four,
               "lower_with_corrected_qwen": sum(v["attention"] < v["plain"] for v in per.values()),
               "per_model_ratio": {m: math.log(v["attention"] / v["base"]) / math.log(v["plain"] / v["base"]) for m, v in per.items()}}
    # post hoc pattern: does the weighting's gain grow with the plain fusion's cost? Round 31's five models (Qwen2.5-7B
    # corrected) and the three development models at lambda 0 and 1 (attnw_fuse_dev/, other calibration and windows)
    dev = {m: json.load(open(f"alignment_study/attnw_fuse_dev/{m}.json")) for m in ("llama-160m", "TinyLlama_v1.1", "open_llama_3b_v2")}
    allm = dict(per, **{m: {"base": r["base"], "plain": r["arms"]["0.0"], "attention": r["arms"]["1.0"]} for m, r in dev.items()})
    cost = {m: math.log(v["plain"] / v["base"]) for m, v in allm.items()}
    gain = {m: math.log(v["attention"] / v["plain"]) for m, v in allm.items()}
    rank = lambda xs: {m: sorted(xs, key=xs.get).index(m) for m in xs}
    rc, rg = rank(cost), rank(gain)
    nm = len(allm)
    summary["posthoc_gain_against_cost"] = {"cost_ln_plain_over_base": cost, "gain_ln_attention_over_plain": gain,
                                           "spearman": 1 - 6 * sum((rc[m] - rg[m]) ** 2 for m in allm) / (nm * (nm * nm - 1)), "models": nm}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    log(f"  {summary}")
    log("ATTNW_BIASFIX_DONE")
