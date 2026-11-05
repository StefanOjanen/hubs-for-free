# Development test (exploratory, labelled): a sink-aware exemption for key-value head merging. pair_start_diag.py found
# that the pairs whose merge is catastrophic on inputs without the beginning-of-sequence token are layer 0's
# sink-serving heads, and bos_ppl_check.py and retrieval_decode_diag.py --bos put a fifth of Mistral-7B-Instruct's
# fused perplexity cost and most of its retrieval loss in that regime. The rule tested here mirrors the retrieval-head
# exemption: score every query head by the attention it puts on position 0 (the sink) over sampled positions of
# held-out C4 windows without the token, give each key-value head the largest score of the query heads it serves,
# keep the top-k key-value heads unmerged and match the rest by the residual as before. Prediction, stated before
# running: on Mistral-7B-Instruct at half the key-value heads, exempting the top 8 sink-serving key-value heads (3
# percent of the cache above half) restores token-less retrieval to at least 40 of 50 (plain fusion 10) and lowers
# token-less perplexity below 14.5 (plain 16.68, the token-first value 13.48), while 8 random heads drawn from the
# lower half of the sink scores do neither; under token-first inputs the exemption changes little. Per arm: perplexity
# on the 8 registered WikiText-2 windows as they are and with the token prepended, the registered needle protocol (ten
# codes, five depths, 50 instances) without the token, and with the token for the plain fusion and the top-8 arm.
# Writes alignment_study/sink_exempt_dev/<model>.json after every arm and <model>_sink_scores.npy.
#   python alignment_study/sink_exempt_dev.py --model=mistralai/Mistral-7B-Instruct-v0.2 [--ks=4,8,16] [--seeds=100,101]
#          [--prefix=1024] [--positions=128] [--score-windows=2] [--calib=128] [--calib-len=2048] [--eval-windows=8] [--score-bos]
# --score-bos scores on windows with the beginning-of-sequence token prepended (the sink at position 0 is then unambiguous;
# without the token Mistral-7B-Instruct puts 1.4 percent of its attention on position 0 and the top heads sit in layers 3 to 7).
import gc
import json
import os
import random
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
arg = lambda n, d=None: next((a.split("=", 1)[1] for a in argv0 if a.startswith(f"--{n}=")), d)
NAME = arg("model")
if not NAME:
    sys.exit("--model=<name>")
sys.argv = [argv0[0], f"--model={NAME}"]
import retrieval_protect_dev as R                     # load (bfloat16, pinned revisions), K, C, Z, DEV, ND, groups_with, needle_eval
sys.argv = argv0
K, C, Z, kvfuse, DEV, ND = R.K, R.C, R.Z, R.kvfuse, R.DEV, R.ND
KS = [int(x) for x in arg("ks", "4,8,16").split(",")]; SEEDS = [int(x) for x in arg("seeds", "100,101").split(",")]
PREFIX, POSITIONS, N_SCORE = int(arg("prefix", 1024)), int(arg("positions", 128)), int(arg("score-windows", 2))
N_CAL, T_CAL, N_EVAL = int(arg("calib", 128)), int(arg("calib-len", 2048)), int(arg("eval-windows", 8))
SCORE_BOS = "--score-bos" in argv0                       # score on windows with the beginning-of-sequence token prepended, where the sink sits at position 0
OUT = f"alignment_study/sink_exempt_dev/{NAME.split('/')[-1]}{'_scorebos' if SCORE_BOS else ''}.json"
torch.set_grad_enabled(False)


def sink_scores(tok, windows, L, H):
    """Mean attention on position 0 per (layer, query head) over POSITIONS sampled query positions after a PREFIX of
    each window: a prefill without attention weights, then one token at a time over the cache with output_attentions."""
    model = R.AutoModelForCausalLM.from_pretrained(NAME, revision=R.REV, dtype=torch.bfloat16, attn_implementation="eager").eval().to(DEV)
    acc = np.zeros((L, H)); n = 0
    for w in range(windows.shape[0]):
        x = windows[w:w + 1].to(DEV)
        pre = model(input_ids=x[:, :PREFIX], use_cache=True); past = pre.past_key_values; del pre
        for t in range(PREFIX, PREFIX + POSITIONS):
            step = model(input_ids=x[:, t:t + 1], past_key_values=past, use_cache=True, output_attentions=True); past = step.past_key_values
            for l in range(L):
                acc[l] += step.attentions[l][0, :, 0, 0].float().cpu().numpy()
            n += 1
            del step
    del model; R.free()
    return acc / n


def needle_eval_bos(model, tok, context_ids):
    inst = []
    for colour, code in ND.FACTS:
        needle = f" The secret code for the {colour} door is {code}. "
        question = f" Based on the content of the book, Question: What is the secret code for the {colour} door?\nAnswer:"
        for depth in ND.DEPTHS:
            ids = ND.build(tok, context_ids, needle, question, depth)
            ids = torch.cat([torch.tensor([[tok.bos_token_id]], dtype=ids.dtype), ids], dim=1)
            a = ND.answer(model, tok, ids); inst.append({"colour": colour, "code": code, "depth": depth, "answer": a, "hit": code in a})
    return {"hits": sum(i["hit"] for i in inst), "of": len(inst), "by_depth": {str(d): sum(i["hit"] for i in inst if i["depth"] == d) for d in ND.DEPTHS}, "instances": inst}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    tok = R.AutoTokenizer.from_pretrained(NAME, revision=R.REV)
    ND.FACTS, ND.DEPTHS = R.REG_FACTS, R.REG_DEPTHS
    cfg = R.AutoConfig.from_pretrained(NAME, revision=R.REV) if hasattr(R, "AutoConfig") else None
    score_win = C.windows_c4(tok, N_CAL + 1 + N_SCORE, T_CAL, bos=SCORE_BOS)[-N_SCORE:]        # held-out documents after the calibration and check documents
    model = R.load(); kv = K.KV(model); n, L, rep, H = kv.nkv, kv.L, kv.rep, kv.n
    del model, kv; R.free()
    sc = sink_scores(tok, score_win, L, H); np.save(OUT[:-5] + "_sink_scores.npy", sc)
    kvsc = sc.reshape(L, n, rep).max(axis=2)
    order = sorted(((float(kvsc[l, i]), l, i) for l in range(L) for i in range(n)), reverse=True)
    log(f"{NAME}: sink scores over {N_SCORE} windows, {POSITIONS} positions after a prefix of {PREFIX}; mean attention on position 0 {sc.mean():.3f}; "
        f"top key-value heads " + ", ".join(f"layer {l} head {i} {s:.2f}" for s, l, i in order[:8]))
    win = Z.wikitext2_windows(tok, T=2048)[:N_EVAL]; win = torch.as_tensor(win) if not torch.is_tensor(win) else win
    win_bos = torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1) if tok.bos_token_id is not None else None
    context_ids = tok.encode(ND.haystack_text(3 * ND.CTX), add_special_tokens=False)
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    model = R.load(); kv = K.KV(model)
    S, ntok = K.second_moments(model, kv, cal, DEV); costs = {l: kvfuse.pair_costs(kv, l, S[l].double()) for l in range(L)}
    res = {"model": NAME, "revision": R.REV, "exploratory": True, "sink_scoring": {"windows": N_SCORE, "prefix": PREFIX, "positions": POSITIONS, "mean_attention_on_position_0": float(sc.mean()), "scored_with_token": SCORE_BOS},
           "top_kv_heads": [{"layer": l, "head": i, "score": s} for s, l, i in order[:32]], "calibration": [N_CAL, T_CAL], "eval_windows": N_EVAL, "arms": {}}
    pool = [(l, i) for s, l, i in order[len(order) // 2:]]                                      # the lower half of the sink scores, for the random controls
    arms = [("fusion", set(), None)] + [(f"sink_top{k}", {(l, i) for s, l, i in order[:k]}, None) for k in KS]
    arms += [(f"random{KS[min(1, len(KS) - 1)]}_seed{sd}", set(random.Random(sd).sample(pool, KS[min(1, len(KS) - 1)])), set(pool)) for sd in SEEDS]
    first = True
    for name, protected, allowed in arms:
        t1 = time.time()
        if not first:
            del model, kv; R.free(); model = R.load(); kv = K.KV(model)
        first = False
        groups = R.groups_with(kv, costs, protected, allowed); K.convert(kv, S, groups, "fuse"); kept = sum(len(g) for g in groups)
        a = {"protected": sorted(protected), "n_protected": len(protected), "kv_heads_kept": kept, "cache_fraction": kept / (L * n),
             "ppl_as_is": Z.perplexity(model, win, DEV, bs=2), "ppl_bos_prepended": Z.perplexity(model, win_bos, DEV, bs=2) if win_bos is not None else None,
             "needle": R.needle_eval(model, tok, context_ids)}
        if name in ("fusion", f"sink_top{KS[min(1, len(KS) - 1)]}") and win_bos is not None:
            a["needle_bos"] = needle_eval_bos(model, tok, context_ids)
        a["seconds"] = round(time.time() - t1, 1); res["arms"][name] = a
        json.dump(res, open(OUT, "w"), indent=1)
        log(f"{NAME}, {name}: {a['n_protected']} exempt, cache {a['cache_fraction']:.3f}; perplexity {a['ppl_as_is']:.3f} as is" + (f", {a['ppl_bos_prepended']:.3f} with the token" if a["ppl_bos_prepended"] else "")
            + f"; needle {a['needle']['hits']} of {a['needle']['of']} {a['needle']['by_depth']}" + (f"; with the token {a['needle_bos']['hits']} of {a['needle_bos']['of']}" if "needle_bos" in a else ""))
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(OUT, "w"), indent=1)
    log("SINK_EXEMPT_DONE")
