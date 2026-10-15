# PREREGISTRATION21: is the attention sink a constant bias or a gate, across
# architectures; and can a constant per-head bias replace the sink token in
# streaming attention? Round 13 edited the first-token sink of four
# grouped-query decoders. This instrument generalizes the edit: a custom
# attention function, equal to eager attention when no edit is active
# (checked per model), for multi-head and grouped-query decoders and for
# bidirectional encoders, with the sink given as a set of columns (the first
# token in decoders, the [SEP] tokens in BERT, the end-of-sentence token in a
# translation encoder).
#
# Part E, edits after the softmax in every layer at once, for all rows other
# than the sink rows:
#   zero  the sink columns' weights set to 0 (each head loses its sink term)
#   mean  each sink column's weight replaced by its mean over those rows
#         (the sink term becomes a constant vector per head)
#   rand  the weight of one ordinary position set to 0 (for scale)
# Bias share = 1 - KL(mean) / KL(zero): the part of the sink's effect that a
# constant per-head vector reproduces.
# Outcomes: decoders, KL of the next-token distribution (round 11's measure);
# BERT, KL of the masked-LM distribution at 15 percent masked positions;
# translation encoder, KL of the decoder's next-token distribution
# teacher-forced on the unchanged model's greedy translation (round 16).
#
# Part W, streaming: sliding-window attention (each query sees the last W
# keys) on 2,048-token windows, scored by next-token NLL on positions >= W:
#   full    full causal attention
#   window  the window only (the sink is evicted once r >= W)
#   stream  the window plus the first four tokens (StreamingLLM)
#   bias    the window plus a constant virtual sink: per layer and head,
#           out = w * v + (1 - w) * (softmax over the window) V, with w the
#           head's mean first-token weight and v the mean first-token value
#           vector, both calibrated under full attention on separate windows
# Recovery = (NLL_window - NLL_bias) / (NLL_window - NLL_stream).
#   python alignment_study/sink_arch.py E|W [--dev] --registered=<freeze permalink>
import json
import os
import sys
import time
import numpy as np
import torch
from torch import nn

sys.path.insert(0, "alignment_study")
import functional_common as F
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.adapters import release_memory, device_label, pick_dtype
from transformers import AttentionInterface

PART = sys.argv[1] if len(sys.argv) > 1 else ""
DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION21.md>; use --dev")
OUT = "alignment_study/sink_arch_dev" if DEV_RUN else "alignment_study/sink_arch"
STATE = {"mode": None, "cols": [0], "rand_col": None, "causal": True, "window": None, "keep_first": 0, "virtual": None, "capture": None}
torch.set_grad_enabled(False)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def gen_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    rep = getattr(module, "num_key_value_groups", 1) or 1
    k = _repeat_kv(key, rep); v = _repeat_kv(value, rep)
    sc = scaling if scaling is not None else query.shape[-1] ** -0.5
    w = torch.matmul(query.float(), k.float().transpose(2, 3)) * sc
    q_len, k_len = query.shape[-2], k.shape[-2]
    allowed = None
    if STATE["causal"]:
        allowed = torch.ones(q_len, k_len, dtype=torch.bool, device=w.device).tril(diagonal=k_len - q_len)
        if STATE["window"] is not None:
            r = torch.arange(q_len, device=w.device)[:, None] + (k_len - q_len); c = torch.arange(k_len, device=w.device)[None, :]
            inwin = c > r - STATE["window"]
            if STATE["keep_first"]:
                inwin = inwin | (c < STATE["keep_first"])
            allowed = allowed & inwin
        w = w.masked_fill(~allowed, float("-inf"))
    elif attention_mask is not None:
        w = w + attention_mask[:, :, :, :k_len] if attention_mask.dtype != torch.bool else w.masked_fill(~attention_mask[:, :, :, :k_len], float("-inf"))
    w = nn.functional.softmax(w, dim=-1)
    li = getattr(module, "layer_idx", None)
    if STATE["capture"] is not None:
        STATE["capture"][li] = (w[0].detach(), v[0].float().detach())
    mode = STATE["mode"]
    if mode is not None:
        w = w.clone(); cols = STATE["cols"]
        rows = torch.ones(q_len, dtype=torch.bool, device=w.device); rows[[c for c in cols if c < q_len]] = False
        if mode == "zero":
            for c in cols:
                w[..., rows, c] = 0
        elif mode == "mean":
            for c in cols:
                w[..., rows, c] = w[..., rows, c].mean(dim=-1, keepdim=True)
        elif mode == "rand":
            c = STATE["rand_col"]; rr = rows.clone(); rr[:c] = False if STATE["causal"] else rr[:c]
            w[..., rr, c] = 0
    out = torch.matmul(w, v.float())
    if STATE["virtual"] is not None:                               # only rows whose window no longer holds the first token
        wv, vv = STATE["virtual"][li]                               # (n,), (n, hd)
        late = (torch.arange(q_len, device=w.device) + (k_len - q_len)) >= STATE["window"]
        mixed = (1 - wv)[None, :, None, None] * out + (wv[:, None] * vv)[None, :, None, :]
        out = torch.where(late[None, None, :, None], mixed, out)
    return out.to(query.dtype).transpose(1, 2).contiguous(), w


AttentionInterface.register("sinkgen", gen_attention)


def set_state(**kw):
    STATE.update({"mode": None, "rand_col": None, "window": None, "keep_first": 0, "virtual": None, "capture": None}); STATE.update(kw)


# ---------------- Part E ----------------
def decoder_E(name, n_eval, log):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    M.N_EVAL, M.T_EVAL = n_eval, 512; X = M.eval_windows(tok)
    dt = pick_dtype(name, DEV, "auto")
    eager = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    ref = eager(input_ids=X[:1].to(DEV)).logits.float(); del eager; release_memory(DEV)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sinkgen", dtype=getattr(torch, dt)).eval().to(DEV)
    set_state(causal=True, cols=[0])
    equiv = float((model(input_ids=X[:1].to(DEV)).logits.float() - ref).abs().max())
    base = F.base_cache(model, X)
    rand_col = int(np.random.default_rng(0).integers(16, 510))
    res = {"equivalence_max_logit_diff": equiv, "rand_col": rand_col}
    for mode in ("zero", "mean", "rand"):
        set_state(causal=True, cols=[0], mode=mode, rand_col=rand_col)
        kls, dn = F.kl_eval(model, X, base); res[mode] = {"kl": float(np.mean(kls)), "dnll": float(np.mean(dn))}
    set_state(causal=True, cols=[0]); res["restore_check_kl"] = float(np.mean(F.kl_eval(model, X, base)[0]))
    del model; release_memory(DEV)
    return res


def bert_E(name, nseg, log):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    sys.path.insert(0, "audits/readings")
    _argv = sys.argv; sys.argv = [sys.argv[0], "--check"]
    import clark_source_format as K
    sys.argv = _argv
    tok = AutoTokenizer.from_pretrained(name)
    segs = K.wikipedia_segments(tok, nseg)
    rg = np.random.default_rng(0); masked = []
    for ids, tt, keep in segs:
        T = ids.shape[1]; cand = np.array([p for p in range(1, T - 1) if p != keep[1]])
        pos = np.sort(rg.choice(cand, max(1, int(0.15 * len(cand))), replace=False))
        x = ids.clone(); x[0, pos] = tok.mask_token_id; masked.append((x, tt, keep, torch.tensor(pos)))
    eager = AutoModelForMaskedLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    ref = eager(input_ids=masked[0][0].to(DEV), token_type_ids=masked[0][1].to(DEV)).logits; del eager; release_memory(DEV)
    model = AutoModelForMaskedLM.from_pretrained(name, attn_implementation="sinkgen", dtype=torch.float32).eval().to(DEV)
    set_state(causal=False, cols=[])
    equiv = float((model(input_ids=masked[0][0].to(DEV), token_type_ids=masked[0][1].to(DEV)).logits - ref).abs().max())

    def run():
        out = []
        for x, tt, keep, pos in masked:
            STATE["cols"] = [keep[1], keep[2]]
            lg = model(input_ids=x.to(DEV), token_type_ids=tt.to(DEV)).logits[0, pos.to(DEV)].float()
            out.append(torch.log_softmax(lg, -1))
        return out

    set_state(causal=False, cols=[]); base = run()

    def kl(cur):
        tot = 0.0; cnt = 0
        for lp, lq in zip(base, cur):
            top, ids = lp.topk(256, -1); p = top.exp(); q_l = lq.gather(1, ids); q = q_l.exp()
            rp = (1 - p.sum(1)).clamp_min(1e-12); rq = (1 - q.sum(1)).clamp_min(1e-12)
            tot += float(((p * (top - q_l)).sum(1) + rp * (rp.log() - rq.log())).sum()); cnt += lp.shape[0]
        return tot / cnt

    res = {"equivalence_max_logit_diff": equiv, "sink": "both [SEP] columns", "n_segments": len(masked)}
    for mode in ("zero", "mean", "rand"):
        set_state(causal=False, cols=[], mode=mode, rand_col=20); res[mode] = {"kl": kl(run())}
    set_state(causal=False, cols=[]); res["restore_check_kl"] = kl(run())
    del model; release_memory(DEV)
    return res


def marian_E(name, log):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import voita_definition as VD
    tok = AutoTokenizer.from_pretrained(name)
    _, ev = VD.mt_sentences(); ev = ev[:60]
    eager = AutoModelForSeq2SeqLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    cache = VD.mt_base(eager, tok, ev)                                  # greedy translations and base distributions
    x0, d0, _, _ = cache[0]; ref = eager(input_ids=x0, decoder_input_ids=d0).logits; del eager; release_memory(DEV)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, attn_implementation="sinkgen", dtype=torch.float32).eval().to(DEV)
    # the edit applies to encoder self-attention only: decoder layers are causal and use their own path
    enc_ids = {id(l.self_attn) for l in model.model.encoder.layers}
    dec_self_ids = {id(l.self_attn) for l in model.model.decoder.layers}
    orig = gen_attention

    def routed(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if id(module) in enc_ids:
            STATE["causal"] = False
            T = key.shape[-2]; STATE["cols"] = [T - 1]
            return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
        saved = dict(STATE); STATE.update({"mode": None, "virtual": None, "capture": None, "causal": id(module) in dec_self_ids})
        o = orig(module, query, key, value, attention_mask, scaling, dropout, **kw); STATE.update(saved); return o

    AttentionInterface.register("sinkgen", routed)
    set_state(causal=False, cols=[])
    equiv = float((model(input_ids=x0, decoder_input_ids=d0).logits - ref).abs().max())
    res = {"equivalence_max_logit_diff": equiv, "sink": "end-of-sentence column", "n_sentences": len(ev)}
    for mode in ("zero", "mean", "rand"):
        set_state(causal=False, cols=[], mode=mode, rand_col=2); res[mode] = {"kl": VD.mt_kl(model, cache)}
    set_state(causal=False, cols=[]); res["restore_check_kl"] = VD.mt_kl(model, cache)
    AttentionInterface.register("sinkgen", orig)
    del model; release_memory(DEV)
    return res


# ---------------- Part W ----------------
def streaming_W(name, n_cal, n_eval, T, W, log):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt")["input_ids"][0]
    m = ids.numel() // T; picks = np.linspace(0, m - 1, n_cal + n_eval).round().astype(int)
    wins = [ids[p * T:(p + 1) * T][None] for p in picks]
    cal, ev = wins[:n_cal], wins[n_cal:]
    dt = pick_dtype(name, DEV, "auto")
    eager = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    ref = eager(input_ids=ev[0][:, :512].to(DEV)).logits.float(); del eager; release_memory(DEV)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sinkgen", dtype=getattr(torch, dt)).eval().to(DEV)
    set_state(causal=True, cols=[0])
    equiv = float((model(input_ids=ev[0][:, :512].to(DEV)).logits.float() - ref).abs().max())
    L = model.config.num_hidden_layers
    # calibration under full attention: mean first-token weight over rows >= W, mean first-token value vector
    sums = {}
    for x in cal:
        cap = {}; set_state(causal=True, cols=[0], capture=cap); model(input_ids=x.to(DEV))
        for l, (A, V) in cap.items():
            wv = A[:, W:, 0].mean(-1); vv = V[:, 0, :]
            s = sums.setdefault(l, [0, 0]); s[0] = s[0] + wv; s[1] = s[1] + vv
    virtual = {l: (s[0] / len(cal), s[1] / len(cal)) for l, s in sums.items()}

    def nll(**kw):
        tot, cnt = 0.0, 0
        for x in ev:
            set_state(causal=True, cols=[0], **kw)
            lg = model(input_ids=x.to(DEV)).logits[0, W - 1:-1].float()
            tgt = x[0, W:].to(DEV)
            tot += float(nn.functional.cross_entropy(lg, tgt, reduction="sum")); cnt += tgt.numel()
        return tot / cnt

    res = {"T": T, "W": W, "n_cal": n_cal, "n_eval": n_eval, "equivalence_max_logit_diff": equiv,
           "full": nll(), "window": nll(window=W), "stream": nll(window=W, keep_first=4), "stream1": nll(window=W, keep_first=1),
           "bias": nll(window=W, virtual=virtual)}
    res["recovery"] = (res["window"] - res["bias"]) / (res["window"] - res["stream"]) if abs(res["window"] - res["stream"]) > 1e-9 else None
    res["virtual_weight_mean"] = float(np.mean([float(v[0].mean()) for v in virtual.values()]))
    set_state(causal=True, cols=[0])
    del model; release_memory(DEV)
    return res


PLAN = {
    "E": {"dev": [("decoder", "gpt2"), ("decoder", "Qwen/Qwen2.5-0.5B"), ("bert", "bert-base-uncased"), ("marian", "Helsinki-NLP/opus-mt-en-es")],
          "conf": [("decoder", "gpt2-medium"), ("decoder", "EleutherAI/pythia-410m"), ("decoder", "facebook/opt-6.7b"),
                   ("bert", "bert-base-cased"), ("marian", "Helsinki-NLP/opus-mt-en-de")]},
    "W": {"dev": ["Qwen/Qwen2.5-0.5B"], "conf": ["TinyLlama/TinyLlama_v1.1", "Qwen/Qwen2.5-1.5B", "EleutherAI/pythia-410m"]},
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    for item in PLAN[PART]["dev" if DEV_RUN else "conf"]:
        kind, name = item if PART == "E" else ("streaming", item)
        path = f"{OUT}/{PART}_{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        log(f"{PART} {kind} {name}")
        if kind == "decoder":
            res = decoder_E(name, 4 if DEV_RUN else 8, log)
        elif kind == "bert":
            res = bert_E(name, 40 if DEV_RUN else 200, log)
        elif kind == "marian":
            res = marian_E(name, log)
        else:
            res = streaming_W(name, 2 if DEV_RUN else 4, 2 if DEV_RUN else 8, 2048, 256, log)
        if PART == "E":
            res["bias_share"] = 1 - res["mean"]["kl"] / res["zero"]["kl"] if res["zero"]["kl"] > 0 else None
            res["zero_over_rand"] = res["zero"]["kl"] / res["rand"]["kl"] if res["rand"]["kl"] > 0 else None
        res.update({"part": PART, "kind": kind, "model": name, "registered": REGISTERED, "dev": DEV_RUN, "device": DEV, "runtime_s": round(time.time() - t0, 1)})
        json.dump(res, open(path, "w"), indent=1)
        log(f"RESULT {PART} {name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k in ("bias_share", "zero_over_rand", "equivalence_max_logit_diff", "full", "window", "stream", "bias", "recovery", "restore_check_kl")}))
    print("SINK_ARCH_DONE", PART, flush=True)
