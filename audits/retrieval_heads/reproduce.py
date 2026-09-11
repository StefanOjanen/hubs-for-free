# Target 5 (Retrieval Heads, Wu et al. 2024, arXiv:2404.15574), step T2.1:
# reproduce the base result only. Claim: a small set of heads copies tokens
# from the context during needle-in-a-haystack decoding; the retrieval score
# of head h is |g_h intersect k| / |k| where k is the needle's token set and
# g_h the tokens the head copies (generated token w is in the needle and the
# head's argmax attention at that step points to the needle position of w),
# averaged over test instances; 3 to 6 percent of heads score above 0.1
# across models (their Section 3), and masking those heads breaks retrieval.
# Their code (nightdessert/Retrieval_Head) is public; this is a compact
# reimplementation of the score at reduced scale (contexts 1K to 4K, nine
# depths, two haystack texts) on an open model that fits the local machine.
#   python audits/retrieval_heads/reproduce.py <model> <ctx_lengths e.g. 1024,2048,4096> <n_depths> [out_prefix]
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
from hubsfree.adapters import pick_device, pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
CTX = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "512").split(",")]
NDEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PREFIX = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "audits/retrieval_heads/" + NAME.split("/")[-1]
# --rope=linear:10 overrides the checkpoint's rope parameters (transformers 5
# `rope_parameters`). Needed for yaofu/llama-2-7b-80k, whose saved config says
# "dynamic" but whose weights generate fluently only under linear position
# interpolation with factor 10 (audits/retrieval_heads/README note, 2026-09-11).
ROPE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rope=")), None)
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)

NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
QUESTION = "\n\nQuestion: What is the best thing to do in San Francisco?\nAnswer: The best thing to do in San Francisco is"
MAX_NEW = 24


def haystacks(tok, n_tokens, k=2):
    """Two filler texts of at least n_tokens tokens built from WikiText-103
    validation paragraphs (the paper used essays; the filler only needs to be
    fluent and unrelated to the needle)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    pool = [t.strip() for t in ds["text"] if len(t.split()) > 40]
    outs, i = [], 0
    for _ in range(k):
        buf = []
        while len(tok(" ".join(buf))["input_ids"]) < n_tokens + 64:
            buf.append(pool[i]); i += 7
        outs.append(" ".join(buf))
    return outs


def greedy_decode_with_argmax(model, x, max_new, eos_id):
    """Greedy decoding that keeps only each step's per-head argmax attention
    position (L, H) instead of the full attention tensors. The prompt is
    prefilled without attention outputs except for its last token, whose
    step (the one that emits the first generated token) is run separately
    with output_attentions=True; every later step likewise returns attention
    over the cache for its single input token. step_argmax[i] is therefore
    the attention of the pass that emitted gen[i], the same alignment as
    generate(output_attentions=True), at the memory cost of the KV cache plus
    one (H, T) row per layer."""
    pre = model(input_ids=x[:, :-1], use_cache=True)
    past = pre.past_key_values
    del pre
    inp = x[:, -1:]
    gen, arg = [], []
    for _ in range(max_new):
        step = model(input_ids=inp, past_key_values=past, use_cache=True, output_attentions=True)
        past = step.past_key_values
        arg.append(np.stack([a[0, :, -1, :].argmax(-1).cpu().numpy() for a in step.attentions]))
        next_tok = step.logits[0, -1].argmax().item()
        del step
        gen.append(next_tok)
        if next_tok == eos_id:
            break
        inp = torch.tensor([[next_tok]], device=x.device)
    return gen, arg


def build(tok, hay, ctx, depth):
    ids = tok(hay)["input_ids"][:ctx]
    pos = int(depth * (len(ids) - 1))
    needle_ids = tok(" " + NEEDLE)["input_ids"]
    full = ids[:pos] + needle_ids + ids[pos:]
    q_ids = tok(QUESTION)["input_ids"]
    return full + q_ids, pos, needle_ids


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(NAME)
    dt = pick_dtype(NAME, DEV, "auto")
    kw = {}
    if ROPE:
        from transformers import AutoConfig
        rtype, factor = ROPE.split(":")
        cfg = AutoConfig.from_pretrained(NAME)
        theta = (cfg.rope_parameters or {}).get("rope_theta", 10000.0)
        cfg.rope_parameters = {"rope_type": rtype, "factor": float(factor), "rope_theta": theta}
        kw["config"] = cfg
        print("rope override:", cfg.rope_parameters, flush=True)
    model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager", dtype=getattr(torch, dt), **kw).eval().to(DEV)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    print(NAME, "DEVICE", device_label(DEV), "DTYPE", dt, f"layers {L} heads {H} contexts {CTX} depths {NDEPTH}", flush=True)
    score_sum = np.zeros((L, H)); n_inst = 0; correct = 0; t0 = time.time()
    per_instance = []
    depths = np.linspace(0.05, 0.95, NDEPTH)
    for ctx in CTX:
        hays = haystacks(tok, ctx)
        for hi, hay in enumerate(hays):
            for depth in depths:
                prompt, pos, needle_ids = build(tok, hay, ctx, float(depth))
                needle_span = set(range(pos, pos + len(needle_ids)))
                needle_tok_at = {}                                             # token id -> positions in the needle
                for j, t in enumerate(needle_ids):
                    needle_tok_at.setdefault(t, []).append(pos + j)
                x = torch.tensor([prompt], device=DEV)
                gen, step_argmax = greedy_decode_with_argmax(model, x, MAX_NEW, tok.eos_token_id)
                text = tok.decode(gen)
                hit = np.zeros((L, H)); k_needle = len(needle_ids)
                copied = [[set() for _ in range(H)] for _ in range(L)]
                for step, tok_id in enumerate(gen):
                    if tok_id not in needle_tok_at or step >= len(step_argmax):
                        continue
                    am_all = step_argmax[step]                                 # (L, H) argmax position of each head in the pass that emitted gen[step]
                    for l in range(L):
                        for h in range(H):
                            if int(am_all[l, h]) in needle_tok_at[tok_id]:
                                copied[l][h].add(int(am_all[l, h]))
                for l in range(L):
                    for h in range(H):
                        hit[l, h] = len(copied[l][h]) / k_needle
                score_sum += hit; n_inst += 1
                ok = "dolores park" in text.lower() or "sandwich" in text.lower()
                correct += int(ok)
                per_instance.append({"ctx": ctx, "haystack": hi, "depth": round(float(depth), 2), "answer": text.strip()[:80], "retrieved": ok,
                                     "max_head_score": round(float(hit.max()), 3), "n_heads_above_0.1": int((hit > 0.1).sum())})
                print(f"  ctx {ctx} hay {hi} depth {depth:.2f}: retrieved={ok} max head score {hit.max():.2f} heads>0.1 {(hit > 0.1).sum():3d} | {text.strip()[:60]!r} ({time.time()-t0:.0f}s)", flush=True)
    del model; release_memory(DEV)
    score = score_sum / max(n_inst, 1)
    np.save(PREFIX + "_scores.npy", score)
    flat = score.ravel()
    summary = {"n_instances": n_inst, "retrieval_accuracy": round(correct / max(n_inst, 1), 3),
               "frac_heads_above_0.1": round(float((flat > 0.1).mean()), 4), "frac_heads_above_0.5": round(float((flat > 0.5).mean()), 4),
               "n_heads_above_0.1": int((flat > 0.1).sum()), "n_heads": int(flat.size), "max_score": round(float(flat.max()), 3),
               "top_heads": [{"layer": int(i // H), "head": int(i % H), "score": round(float(flat[i]), 3)} for i in np.argsort(flat)[::-1][:10]],
               "paper": "3 to 6 percent of heads above 0.1 (Wu et al. 2024, contexts 1K to 50K, about 600 instances)"}
    json.dump({"model": NAME, "dtype": dt, "contexts": CTX, "n_depths": NDEPTH, "needle": NEEDLE, "max_new_tokens": MAX_NEW,
               "per_instance": per_instance, "summary": summary}, open(PREFIX + "_base_result.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary)); print("RETRIEVAL_DONE", PREFIX)
