# Target 5 inclusion rule for yaofu/llama-2-7b-80k (PREREGISTRATION4.md,
# Target 5, frozen 2026-09-11): repeat the reproduction with the source
# repository's own haystack, needle, question and prompt template
# (nightdessert/Retrieval_Head, commit 3ac171a of 2024-08-02) at 1K and 2K
# tokens on 20 instances; the model joins the battery if the needle is
# retrieved in at least 18 of 20 instances and the fraction of heads above
# 0.1 lies between 2 and 8 percent, and stays excluded otherwise.
#
# Source protocol reimplemented from retrieval_head_detection.py: the
# haystack for needle k is the text of haystack_for_detect/part{k+1}
# (repeated until long enough), trimmed to context_length tokens and then to
# context_length - 200 - len(needle) tokens; the needle from needles.jsonl
# (with its newlines) is inserted after the last sentence-ending '.' token
# before depth percent of the context; the prompt is context + "Based on the
# content of the book, Question: {q}\nAnswer:" (the source's non-chat
# branch); greedy decoding for 50 steps, stopping at a newline; per step, a
# head whose top-1 attention position lies inside the needle span at a
# position whose prompt token equals the generated token scores
# 1 / span length; head scores accumulate over the instances whose response
# has ROUGE-1 recall above 50 percent of the "real needle" (the source's
# success test) and are averaged over those instances.
#   python audits/retrieval_heads/inclusion_test.py <model> --source-dir=<clone of nightdessert/Retrieval_Head>
#          [--rope=linear:10] [--contexts=1024,2048] [--depths=5] [--needles=2] [--dry-run]
import glob
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
from hubsfree.adapters import pick_device, pick_dtype, release_memory, device_label
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

DRY = "--dry-run" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
NAME = "Qwen/Qwen2.5-0.5B" if DRY else (ARGS[0] if ARGS else "yaofu/llama-2-7b-80k")
opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
SRC = opt("source-dir", None)
ROPE = opt("rope", None)
CTX = [int(x) for x in opt("contexts", "512" if DRY else "1024,2048").split(",")]
NDEPTH = int(opt("depths", "2" if DRY else "5"))
NNEEDLE = int(opt("needles", "1" if DRY else "2"))
BUFFER, MAX_NEW = 200, 50
SHORT = NAME.split("/")[-1]
OUT = "/tmp/inclusion_test_dryrun.json" if DRY else f"audits/retrieval_heads/{SHORT}_inclusion_test.json"
DEV = pick_device(os.environ.get("HUBSFREE_DEVICE", "auto"))
torch.set_grad_enabled(False)
assert SRC and os.path.isdir(os.path.join(SRC, "haystack_for_detect")), "--source-dir must point at a clone of nightdessert/Retrieval_Head"
SRC_COMMIT = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


def needles():
    return [json.loads(l) for l in open(os.path.join(SRC, "haystack_for_detect", "needles.jsonl")) if l.strip()]


def haystack_text(k, max_words):
    files = sorted(glob.glob(os.path.join(SRC, "haystack_for_detect", f"part{k + 1}", "*.txt")))
    context = ""
    while len(context.split()) < max_words:
        for f in files:
            context += open(f).read()
    return context


def period_tokens(tok):
    ids = set(tok.encode(".", add_special_tokens=False)) | set(tok.encode(" .", add_special_tokens=False)) | set(tok.encode("sentence.", add_special_tokens=False)[-1:])
    return {i for i in ids if tok.decode([i]).strip() == "."}


def build(tok, context, needle, question, ctx, depth_percent, periods):
    """The source's generate_context and prompt: trim, insert at a sentence end, decode, add the question."""
    toks = tok.encode(context, add_special_tokens=False)[:ctx]
    tn = tok.encode(needle, add_special_tokens=False)
    limit = ctx - BUFFER
    if len(toks) + len(tn) > limit:
        toks = toks[:limit - len(tn)]
    if depth_percent >= 100:
        new = toks + tn
    else:
        ip = int(len(toks) * depth_percent / 100)
        new = toks[:ip]
        while new and new[-1] not in periods:
            ip -= 1; new = toks[:ip]
        new = new + tn + toks[ip:]
    text = tok.decode(new) + f"Based on the content of the book, Question: {question}\nAnswer:"
    ids = tok(text, return_tensors="pt")["input_ids"]
    # the source's find_needle_idx: first window with more than 0.9 token-set overlap with the needle
    pid = ids[0].tolist(); nset = set(tn); span = len(tn); start = -1
    for i in range(len(pid) - span + 1):
        if len(set(pid[i:i + span]) & nset) / len(nset) > 0.9:
            start = i; break
    return ids, start, start + span if start >= 0 else -1


def decode(model, tok, ids, start, end):
    """Greedy decoding with the source's per-step scoring; returns the response tokens and the (L, H) score."""
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    score = np.zeros((L, H)); x = ids.to(DEV); pid = ids[0]
    pre = model(input_ids=x[:, :-1], use_cache=True); past = pre.past_key_values; del pre
    inp = x[:, -1:]; gen = []
    for _ in range(MAX_NEW):
        step = model(input_ids=inp, past_key_values=past, use_cache=True, output_attentions=True)
        past = step.past_key_values
        nxt = step.logits[0, -1].argmax().item()
        if start >= 0:
            for l, a in enumerate(step.attentions):
                top = a[0, :, -1, :].argmax(-1).cpu().numpy()          # top-1 position per head
                hit = (top >= start) & (top < end)
                for h in np.nonzero(hit)[0]:
                    if int(pid[int(top[h])]) == nxt:
                        score[l, h] += 1.0 / (end - start)
        del step
        gen.append(nxt)
        piece = tok.decode([nxt])
        if nxt == tok.eos_token_id or "\n" in piece:
            break
        inp = torch.tensor([[nxt]], device=DEV)
    return gen, score


if __name__ == "__main__":
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(NAME)
    dt = pick_dtype(NAME, DEV, "auto"); kw = {}
    if ROPE:
        rtype, factor = ROPE.split(":"); cfg = AutoConfig.from_pretrained(NAME)
        theta = (cfg.rope_parameters or {}).get("rope_theta", 10000.0)
        cfg.rope_parameters = {"rope_type": rtype, "factor": float(factor), "rope_theta": theta}; kw["config"] = cfg
        print("rope override:", cfg.rope_parameters, flush=True)
    model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager", dtype=getattr(torch, dt), **kw).eval().to(DEV)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    print(NAME, "DEVICE", device_label(DEV), "DTYPE", dt, f"layers {L} heads {H} contexts {CTX} depths {NDEPTH} needles {NNEEDLE} source {SRC_COMMIT[:7]}", flush=True)
    periods = period_tokens(tok); print("sentence-end tokens", sorted(periods), flush=True)
    depths = [10, 30, 50, 70, 90][:NDEPTH] if NDEPTH <= 5 else list(np.linspace(0, 100, NDEPTH))
    per, acc, n_ok = [], np.zeros((L, H)), 0
    all_acc = np.zeros((L, H))
    for k, nd in enumerate(needles()[:NNEEDLE]):
        context = haystack_text(k, max(CTX))
        for ctx in CTX:
            for dp in depths:
                ids, start, end = build(tok, context, nd["needle"], nd["question"], ctx, dp, periods)
                gen, score = decode(model, tok, ids, start, end)
                resp = tok.decode(gen, skip_special_tokens=True).strip()
                recall = scorer.score(nd["real_needle"], resp)["rouge1"].recall * 100
                ok = recall > 50
                n_ok += ok; acc += score if ok else 0; all_acc += score
                per.append({"needle": k, "ctx": ctx, "depth_percent": dp, "prompt_tokens": int(ids.shape[1]), "needle_span": [start, end],
                            "retrieved": bool(ok), "rouge1_recall": round(recall, 1), "response": resp[:100], "max_head_score": round(float(score.max()), 3)})
                print(f"  needle {k} ctx {ctx} depth {dp:3d}: retrieved={ok} recall {recall:5.1f} max head {score.max():.2f} | {resp[:60]!r} ({time.time()-t0:.0f}s)", flush=True)
    del model; release_memory(DEV)
    n = len(per); head = acc / max(n_ok, 1); flat = head.ravel()
    frac = float((flat > 0.1).mean()); frac_all = float(((all_acc / max(n, 1)).ravel() > 0.1).mean())
    included = (n_ok >= 18) and (0.02 <= frac <= 0.08) and n == 20
    res = {"model": NAME, "rope": ROPE, "dtype": dt, "source_repo": "nightdessert/Retrieval_Head", "source_commit": SRC_COMMIT, "contexts": CTX,
           "depths_percent": depths, "needles": NNEEDLE, "instances": n, "retrieved": int(n_ok),
           "frac_heads_above_0.1": round(frac, 4), "frac_heads_above_0.1_all_instances": round(frac_all, 4), "n_heads": int(flat.size),
           "top_heads": [{"layer": int(i // H), "head": int(i % H), "score": round(float(flat[i]), 3)} for i in np.argsort(flat)[::-1][:10]],
           "rule": "included if retrieved in at least 18 of 20 instances and 2 to 8 percent of heads above 0.1 (PREREGISTRATION4.md, Target 5)",
           "included": bool(included), "dry_run": DRY, "per_instance": per, "runtime_s": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    print("SUMMARY", json.dumps({k: res[k] for k in ("model", "instances", "retrieved", "frac_heads_above_0.1", "included")}))
    print("INCLUSION_DONE", OUT)
