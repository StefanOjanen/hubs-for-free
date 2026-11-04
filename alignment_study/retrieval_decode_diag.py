# Diagnostic after round 35 (development, exploratory): does the fused base model stop answering, or does the
# protocol's stop rule cut an answer that follows a newline? needle_dev.answer decodes greedily and ends at an
# end-of-sequence token or at the first token containing a newline after the first; the fused Mistral-7B-v0.1 gave
# "\n\n" for 46 of 50 plain-fusion prompts (retrieval_answers.py). For one model, the registered prompts (ten codes at
# five depths of a 2,048-token C4 haystack) under the unconverted model and under the registered conversions named by
# --arms (the plain fusion at half the key-value heads; the retrieval heads at 0.3 exempt): the next-token distribution
# at the answer position (mass on the end-of-sequence token, on tokens containing a newline, on tokens beginning with a
# digit; the rank of the code's first token), and one 24-token greedy decode that stops only at the end of sequence,
# scored twice, as the protocol scores it (cut at the first newline after the first token) and uncut. Writes
# alignment_study/retrieval_diag/<model>.json after every arm.
#   python alignment_study/retrieval_decode_diag.py --model=mistralai/Mistral-7B-v0.1 [--arms=fusion,retrieval_0.3] [--bos]
# --bos prepends the beginning-of-sequence token to every prompt (the registered prompts carry none; pair_start_diag.py found
# layer 0's sink-serving pairs fail only without it) and writes retrieval_diag/<model>_bos.json.
import json
import os
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
NAME = next((a.split("=", 1)[1] for a in argv0 if a.startswith("--model=")), None)
if not NAME:
    sys.exit("--model=<name>")
sys.argv = [argv0[0], f"--model={NAME}"]
import retrieval_protect_dev as R                     # ND, K, C, Z, DEV, the registered facts and depths, groups_with, load
sys.argv = argv0
ND, K, C, Z, kvfuse, DEV = R.ND, R.K, R.C, R.Z, R.kvfuse, R.DEV
ARMS = [a for a in next((x.split("=", 1)[1] for x in argv0 if x.startswith("--arms=")), "fusion,retrieval_0.3").split(",") if a]
BOS = "--bos" in argv0                                  # prepend the beginning-of-sequence token to every prompt (the registered prompts carry none)
OUT = f"alignment_study/retrieval_diag/{NAME.split('/')[-1]}{'_bos' if BOS else ''}.json"
torch.set_grad_enabled(False)


def token_classes(tok):
    """Vocabulary ids whose decoded text contains a newline, and ids whose decoded text begins with a digit."""
    nl, dg = [], []
    for t in range(len(tok)):
        s = tok.decode([t])
        if "\n" in s:
            nl.append(t)
        if s.strip()[:1].isdigit():
            dg.append(t)
    return torch.tensor(nl), torch.tensor(dg)


def decode(model, tok, ids, nl_ids, dg_ids, code):
    x = ids.to(DEV)
    out = model(input_ids=x, use_cache=True); past = out.past_key_values
    p = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu()
    firsts = set()
    for enc in (tok.encode(" " + code, add_special_tokens=False), tok.encode(code, add_special_tokens=False)):
        firsts.add(next(t for t in enc if any(ch.isdigit() for ch in tok.decode([t]))))      # the first token carrying a digit (a space marker may precede it)
    order = torch.argsort(p, descending=True); rank = min(int((order == f).nonzero()[0, 0]) + 1 for f in firsts)
    dist = {"p_eos": float(p[tok.eos_token_id]), "p_newline": float(p[nl_ids].sum()), "p_digit": float(p[dg_ids].sum()),
            "p_code_first_token": float(max(p[t] for t in firsts)),
            "top_token": tok.decode([int(order[0])]), "rank_of_code_first_token": rank}
    nxt = int(out.logits[0, -1].argmax()); gen = [nxt]
    for _ in range(ND.MAX_NEW - 1):
        if nxt == tok.eos_token_id:
            break
        out = model(input_ids=torch.tensor([[nxt]], device=DEV), past_key_values=past, use_cache=True); past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax()); gen.append(nxt)
    cut = len(gen)
    for i in range(1, len(gen)):                              # the protocol's rule: stop at the first newline (or end) after the first token
        if gen[i] == tok.eos_token_id or "\n" in tok.decode([gen[i]]):
            cut = i + 1; break
    full = tok.decode(gen); prot = tok.decode(gen[:cut])
    return {**dist, "answer_protocol": prot, "hit_protocol": code in prot, "answer_uncut": full, "hit_uncut": code in full, "tokens_generated": len(gen)}


def run_arm(model, tok, context_ids, nl_ids, dg_ids):
    inst = []
    for colour, code in ND.FACTS:
        needle = f" The secret code for the {colour} door is {code}. "
        question = f" Based on the content of the book, Question: What is the secret code for the {colour} door?\nAnswer:"
        for depth in ND.DEPTHS:
            ids = ND.build(tok, context_ids, needle, question, depth)
            if BOS:
                ids = torch.cat([torch.tensor([[tok.bos_token_id]], dtype=ids.dtype), ids], dim=1)
            inst.append({"colour": colour, "code": code, "depth": depth, **decode(model, tok, ids, nl_ids, dg_ids, code)})
    summ = {k: float(np.mean([i[k] for i in inst])) for k in ("p_eos", "p_newline", "p_digit", "p_code_first_token")}
    summ["median_rank_of_code_first_token"] = float(np.median([i["rank_of_code_first_token"] for i in inst]))
    summ["hits_protocol"] = sum(i["hit_protocol"] for i in inst); summ["hits_uncut"] = sum(i["hit_uncut"] for i in inst); summ["of"] = len(inst)
    return {"instances": inst, "summary": summ}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    ND.FACTS, ND.DEPTHS = R.REG_FACTS, R.REG_DEPTHS
    tok = R.AutoTokenizer.from_pretrained(NAME, revision=R.REV)
    if BOS and tok.bos_token_id is None:
        print(f"{NAME}: the tokenizer has no beginning-of-sequence token (Qwen2.5 prompts carry none in deployment either); --bos does not apply", flush=True); sys.exit(0)
    nl_ids, dg_ids = token_classes(tok)
    context_ids = tok.encode(ND.haystack_text(3 * ND.CTX), add_special_tokens=False)
    scores = np.load(R.SCORES)
    model = R.load(); kv = K.KV(model); n, L, rep = kv.nkv, kv.L, kv.rep
    res = {"model": NAME, "revision": R.REV, "exploratory": True, "bos_prepended": BOS, "arms": {}, "newline_tokens": int(len(nl_ids)), "digit_tokens": int(len(dg_ids)),
           "needle": {"context_tokens": ND.CTX, "facts": len(ND.FACTS), "depths": list(ND.DEPTHS)}}
    a = run_arm(model, tok, context_ids, nl_ids, dg_ids); res["arms"]["unconverted"] = a
    json.dump(res, open(OUT, "w"), indent=1); log(f"{NAME}, unconverted: {a['summary']}")
    cal = C.windows_c4(tok, ND.N_CAL, ND.T_CAL)
    S, ntok = K.second_moments(model, kv, cal, DEV)
    costs = {l: kvfuse.pair_costs(kv, l, S[l].double()) for l in range(L)}
    for i, arm in enumerate(ARMS):
        if i > 0:
            del model, kv; R.free(); model = R.load(); kv = K.KV(model)
        protected = set() if arm == "fusion" else {(int(l), int(h) // rep) for l, h in np.argwhere(scores >= float(arm.split("_")[1]))}
        groups = R.groups_with(kv, costs, protected); K.convert(kv, S, groups, "fuse")
        a = run_arm(model, tok, context_ids, nl_ids, dg_ids); a["protected"] = sorted(protected); a["kv_heads_kept"] = sum(len(g) for g in groups)
        res["arms"][arm] = a; json.dump(res, open(OUT, "w"), indent=1); log(f"{NAME}, {arm}: {a['summary']}")
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(OUT, "w"), indent=1)
    log("RETRIEVAL_DIAG_DONE")
