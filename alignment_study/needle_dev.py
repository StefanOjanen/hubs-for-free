# Development (exploratory, labelled): does sharing key-value heads hurt retrieval from context? A planted fact
# ("the secret code for the <colour> door is <5 digits>") is inserted at 10, 50 and 90 percent of a 2,048-token
# haystack of C4 validation text (documents after the calibration and check documents), followed by the question in
# the Retrieval_Head source's non-chat form; greedy decoding for 24 tokens; a hit when the digits appear in the
# answer. Five facts by three depths, the unconverted model against the plain fusion at half the key-value heads
# (round 28's calibration, pairs by the residual). Mistral-7B-Instruct-v0.2 by default; bfloat16.
#   python alignment_study/needle_dev.py [--model=mistralai/Mistral-7B-Instruct-v0.2] [--ctx=2048]
import gc
import gzip
import json
import os
import sys
import time
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import kvfuse
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "mistralai/Mistral-7B-Instruct-v0.2")
REVISIONS = {"mistralai/Mistral-7B-Instruct-v0.2": "63a8b081895390a26e140280378bc85ec8bce07a"}
CTX, N_CAL, T_CAL, SKIP, MAX_NEW = int(arg("ctx", 2048)), 128, 2048, 130, 24
FACTS = [("blue", "48213"), ("green", "90457"), ("red", "27365"), ("yellow", "61928"), ("white", "35074")]
DEPTHS = (10, 50, 90)
OUT = f"alignment_study/needle_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


def haystack_text(min_words):
    """C4 validation shard 0 text after the first SKIP documents (the calibration and check documents come earlier)."""
    p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    text, k = "", 0
    with gzip.open(p, "rt") as f:
        for line in f:
            k += 1
            if k <= SKIP:
                continue
            text += json.loads(line)["text"].replace("\n", " ") + " "
            if len(text.split()) >= min_words:
                break
    return text


def build(tok, context_ids, needle, question, depth):
    tn = tok.encode(needle, add_special_tokens=False)
    tq = tok.encode(question, add_special_tokens=False)
    body = context_ids[:CTX - len(tn) - len(tq) - 8]
    ip = int(len(body) * depth / 100)
    ids = body[:ip] + tn + body[ip:] + tq
    return torch.tensor([ids])


def answer(model, tok, ids):
    x = ids.to(DEV)
    out = model(input_ids=x, use_cache=True); past = out.past_key_values
    nxt = out.logits[0, -1].argmax().item(); gen = [nxt]
    for _ in range(MAX_NEW - 1):
        out = model(input_ids=torch.tensor([[nxt]], device=DEV), past_key_values=past, use_cache=True); past = out.past_key_values
        nxt = out.logits[0, -1].argmax().item(); gen.append(nxt)
        if nxt == tok.eos_token_id or "\n" in tok.decode([nxt]):
            break
    return tok.decode(gen)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISIONS.get(NAME))
    context_ids = tok.encode(haystack_text(3 * CTX), add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISIONS.get(NAME), dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    res = {"model": NAME, "exploratory": True, "context_tokens": CTX, "facts": len(FACTS), "depths": list(DEPTHS), "max_new_tokens": MAX_NEW, "arms": {}}
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    for arm in ("unconverted", "fusion"):
        if arm == "fusion":
            S, _ = K.second_moments(model, kv, cal, DEV)
            groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
            K.convert(kv, S, groups, "fuse"); del S
        inst = []
        for colour, code in FACTS:
            needle = f" The secret code for the {colour} door is {code}. "
            question = f" Based on the content of the book, Question: What is the secret code for the {colour} door?\nAnswer:"
            for depth in DEPTHS:
                ids = build(tok, context_ids, needle, question, depth)
                a = answer(model, tok, ids)
                inst.append({"colour": colour, "code": code, "depth": depth, "tokens": int(ids.shape[1]), "answer": a, "hit": code in a})
        hits = sum(i["hit"] for i in inst)
        res["arms"][arm] = {"instances": inst, "hits": hits, "of": len(inst), "by_depth": {str(d): sum(i["hit"] for i in inst if i["depth"] == d) for d in DEPTHS},
                            "seconds": round(time.time() - t0, 1)}
        log(f"{NAME}, {arm}: {hits} of {len(inst)} retrieved; by depth {res['arms'][arm]['by_depth']}")
        json.dump(res, open(OUT, "w"), indent=1)
    del model, kv; gc.collect()
    log("NEEDLE_DEV_DONE")
