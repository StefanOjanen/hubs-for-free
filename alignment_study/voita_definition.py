# PREREGISTRATION16: the "confidence" of Voita et al. (2019) as they define it,
# the mean largest attention weight excluding the end-of-sentence token, which
# they leave out "on the grounds that it is not a real token" (their footnote
# 2). In their translation encoders that token is where attention goes when a
# head has nothing to attend to, the counterpart of the decoder's sink.
# Rounds 12, 14 and 15 computed confidence over all tokens, sink included.
#
# Part D, decoder language models. On the selection windows of rounds 12 and
# 15 (identical windows, model, precision and sink-column rule), per head,
# rows >= ROW0:
#   conf_all     largest attention weight, mean over rows (rounds 12 to 15;
#                checked against the stored value)
#   conf_voita   largest weight on a key outside the layer's sink columns
#                (Voita et al.'s definition with the sink in place of EOS)
#   conf_first   the same with only the first token excluded
#   conf_renorm  conf_voita's row maximum divided by the row's non-sink mass
#   sink_mass    attention on the sink columns (checked against the stored value)
# Outcomes are the removal damages already stored by rounds 12 and 15.
#
# Part T, Voita et al.'s setting: encoder self-attention heads of Marian
# Transformer translation models (OPUS-MT, 6 layers, 8 heads, d 512) for
# English to Russian, German and French, Voita et al.'s language pairs. Source
# sentences: English side of newstest2013 (WMT14 validation), 8 to 40 words;
# the first 200 for statistics, the next 100 for outcomes. Per head, over
# source tokens (the EOS row excluded, every token weighted equally):
#   conf_voita   largest weight on a token other than EOS (their definition)
#   conf_all     largest weight, EOS included
#   eos_mass, neg_entropy, out_norm (norm of the head's output a V_h W_O,h^T)
# Outcome: the head's slice of the encoder layer's output projection is
# zeroed and the decoder's next-token distribution is compared with the
# unchanged model's (KL over the top 256 tokens plus remainder, rounds 11 to
# 15's measure), teacher-forced on the unchanged model's greedy translation.
#   python alignment_study/voita_definition.py [--dev] [--only=name]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import functional_common as F
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.stats import sink_columns
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

DEV_RUN = "--dev" in sys.argv
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
OUT = "alignment_study/voita_dev" if DEV_RUN else "alignment_study/voita"
DECODERS = ["gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
            "Qwen/Qwen2.5-1.5B", "TinyLlama/TinyLlama_v1.1", "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B"]
DECODERS_DEV = ["gpt2", "Qwen/Qwen2.5-0.5B"]
TRANSLATION = ["Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-en-fr"]
TRANSLATION_DEV = ["Helsinki-NLP/opus-mt-en-es"]
N_SEL_MT, N_EVAL_MT, MIN_W, MAX_W = 200, 100, 8, 40
ROW0, TOPK = F.ROW0, F.TOPK
DKEYS = ("conf_all", "conf_voita", "conf_first", "conf_renorm", "sink_mass")
TKEYS = ("conf_voita", "conf_all", "eos_mass", "neg_entropy", "out_norm")
torch.set_grad_enabled(False)


# ---------------- Part D: decoder language models ----------------
def decoder_stats(model, sel, L, n):
    acc = {l: [{k: [] for k in DKEYS} for _ in range(n)] for l in range(L)}
    for x in sel:
        out = model(input_ids=x.to(DEV), output_attentions=True)
        for l in range(L):
            A = out.attentions[l][0].float().cpu().numpy().astype(np.float64)
            T = A.shape[1]; S = list(sink_columns(A))
            keep = np.ones(T, bool); keep[S] = False
            keep0 = np.ones(T, bool); keep0[0] = False
            for h in range(n):
                a = A[h][ROW0:]
                sm = a[:, S].sum(1); mx = a[:, keep].max(1)
                d = acc[l][h]
                d["conf_all"].append(a.max(1).mean()); d["conf_voita"].append(mx.mean()); d["conf_first"].append(a[:, keep0].max(1).mean())
                d["conf_renorm"].append((mx / np.maximum(1.0 - sm, 1e-12)).mean()); d["sink_mass"].append(sm.mean())
        del out
    return {l: [{k: float(np.mean(v)) for k, v in d.items()} for d in acc[l]] for l in range(L)}


def stored_head_stats(c, l):
    s = c["stats"][str(l)]
    return s["heads"] if isinstance(s, dict) else s


def run_decoder(name, outcome_path):
    t0 = time.time()
    log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    dt = pick_dtype(name, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    c = model.config
    L = getattr(c, "num_hidden_layers", None) or c.n_layer; n = getattr(c, "num_attention_heads", None) or c.n_head
    log(f"{name} DTYPE {dt} layers {L} heads {n}")
    M.N_SEL, M.T_SEL = F.N_SEL, F.T_SEL
    stats = decoder_stats(model, M.selection_windows(tok), L, n)
    del model; release_memory(DEV)
    ref = json.load(open(outcome_path))
    dc = max(abs(stats[l][h]["conf_all"] - stored_head_stats(ref, l)[h]["conf"]) for l in range(L) for h in range(n))
    ds = max(abs(stats[l][h]["sink_mass"] - stored_head_stats(ref, l)[h]["sink_mass"]) for l in range(L) for h in range(n))
    log(f"check against {outcome_path}: max |conf_all - stored conf| {dc:.2e}, max |sink_mass - stored| {ds:.2e}")
    return {"model": name, "part": "decoder", "dtype": dt, "device": DEV, "layers": L, "heads": n, "outcome_file": outcome_path,
            "check_max_abs_diff": {"conf": dc, "sink_mass": ds}, "stats": {str(l): stats[l] for l in range(L)}, "runtime_s": round(time.time() - t0, 1)}


DEC_OUTCOMES = {"gpt2": "alignment_study/importance_mha_dev/gpt2.json", "Qwen/Qwen2.5-0.5B": "alignment_study/standard_dev/Qwen2.5-0.5B.json",
                "gpt2-medium": "alignment_study/importance_mha/gpt2-medium.json",
                "EleutherAI/pythia-160m": "alignment_study/importance_mha/pythia-160m.json",
                "EleutherAI/pythia-410m": "alignment_study/importance_mha/pythia-410m.json",
                "Qwen/Qwen2.5-1.5B": "alignment_study/standard/Qwen2.5-1.5B.json",
                "TinyLlama/TinyLlama_v1.1": "alignment_study/standard/TinyLlama_v1.1.json",
                "mistralai/Mistral-7B-v0.1": "alignment_study/standard/Mistral-7B-v0.1.json",
                "Qwen/Qwen2.5-7B": "alignment_study/standard/Qwen2.5-7B.json"}


# ---------------- Part T: translation encoders ----------------
def mt_sentences():
    from datasets import load_dataset
    ds = load_dataset("wmt/wmt14", "de-en", split="validation", streaming=True)
    ok = [ex["translation"]["en"] for ex in ds if MIN_W <= len(ex["translation"]["en"].split()) <= MAX_W]
    return ok[:N_SEL_MT], ok[N_SEL_MT:N_SEL_MT + N_EVAL_MT]


def enc_attn(model, l):
    return model.model.encoder.layers[l].self_attn


def mt_stats(model, tok, sents, L, n, hd):
    grams = {}
    for l in range(L):
        W = enc_attn(model, l).out_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        grams[l] = [W[:, h * hd:(h + 1) * hd].T @ W[:, h * hd:(h + 1) * hd] for h in range(n)]
    store, handles = {}, []
    for l in range(L):
        def pre(m, args, l=l):
            store[l] = args[0][0].detach().float().cpu().numpy().astype(np.float64)
        handles.append(enc_attn(model, l).out_proj.register_forward_pre_hook(pre))
    acc = {l: [{k: [] for k in TKEYS} for _ in range(n)] for l in range(L)}
    for s in sents:
        x = tok(s, return_tensors="pt")["input_ids"]
        assert int(x[0, -1]) == tok.eos_token_id
        out = model.model.encoder(input_ids=x.to(DEV), output_attentions=True)
        for l in range(L):
            A = out.attentions[l][0].float().cpu().numpy().astype(np.float64)[:, :-1, :]      # query rows without EOS
            Xin = store[l][:-1]
            for h in range(n):
                a = A[h]; d = acc[l][h]
                d["conf_voita"] += list(a[:, :-1].max(1)); d["conf_all"] += list(a.max(1)); d["eos_mass"] += list(a[:, -1])
                d["neg_entropy"] += list((a * np.log(np.clip(a, 1e-30, 1.0))).sum(1))
                xh = Xin[:, h * hd:(h + 1) * hd]
                d["out_norm"] += list(np.sqrt(np.clip(np.einsum("rd,de,re->r", xh, grams[l][h], xh), 0, None)))
        del out
    for hnd in handles:
        hnd.remove()
    return {l: [{k: float(np.mean(v)) for k, v in d.items()} for d in acc[l]] for l in range(L)}


def mt_base(model, tok, sents):
    cache = []
    for s in sents:
        x = tok(s, return_tensors="pt")["input_ids"].to(DEV)
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), num_beams=1, do_sample=False, max_new_tokens=128)
        dec = g[:, :-1]
        logp = torch.log_softmax(model(input_ids=x, decoder_input_ids=dec).logits[0].float(), -1)
        lp, ids = logp.topk(TOPK, dim=-1)
        cache.append((x, dec, ids, lp))
    return cache


def mt_kl(model, cache):
    tot, cnt = 0.0, 0
    for x, dec, ids, lp in cache:
        logq = torch.log_softmax(model(input_ids=x, decoder_input_ids=dec).logits[0].float(), -1)
        lq = logq.gather(1, ids); p, q = lp.exp(), lq.exp()
        rp = (1 - p.sum(1)).clamp_min(1e-12); rq = (1 - q.sum(1)).clamp_min(1e-12)
        kl = (p * (lp - lq)).sum(1) + rp * (rp.log() - rq.log())
        tot += float(kl.sum()); cnt += kl.numel()
    return tot / cnt


def run_translation(name):
    t0 = time.time()
    log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    c = model.config; L, n = c.encoder_layers, c.encoder_attention_heads; hd = c.d_model // n
    log(f"{name} encoder layers {L} heads {n} head_dim {hd}")
    sel, ev = mt_sentences()
    stats = mt_stats(model, tok, sel, L, n, hd)
    log("selection statistics done")
    cache = mt_base(model, tok, ev)
    n_tgt = int(sum(d.shape[1] for _, d, _, _ in cache))
    abl = {}
    for l in range(L):
        W = enc_attn(model, l).out_proj.weight; abl[str(l)] = {}
        for h in range(n):
            sl = slice(h * hd, (h + 1) * hd); bk = W[:, sl].clone(); W[:, sl] = 0
            abl[str(l)][str(h)] = {"kl": mt_kl(model, cache)}
            W[:, sl] = bk
        log(f"  encoder layer {l}: {n} heads")
    check = mt_kl(model, cache)
    del model; release_memory(DEV)
    return {"model": name, "part": "translation", "dtype": "float32", "device": DEV, "layers": L, "heads": n, "head_dim": hd,
            "n_sel_sentences": len(sel), "n_eval_sentences": len(ev), "eval_target_tokens": n_tgt, "restore_check_kl": check,
            "stats": {str(l): stats[l] for l in range(L)}, "ablation": abl, "runtime_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", device_label(DEV), flush=True)
    jobs = [(m, "T") for m in (TRANSLATION_DEV if DEV_RUN else TRANSLATION)] + [(m, "D") for m in (DECODERS_DEV if DEV_RUN else DECODERS)]
    for name, part in jobs:
        if ONLY and name not in ONLY:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        try:
            res = run_translation(name) if part == "T" else run_decoder(name, DEC_OUTCOMES[name])
            json.dump(res, open(path, "w"), indent=1)
            print(f"MODEL_DONE {name} {res['runtime_s']}s", flush=True)
        except Exception as e:
            import traceback
            print(f"MODEL_FAILED {name}: {type(e).__name__}: {str(e)[:300]}", flush=True); print(traceback.format_exc()[-1500:], flush=True)
    print("VOITA_DONE", flush=True)
