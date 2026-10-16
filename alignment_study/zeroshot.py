# Zero-shot evaluation in the formats of lm-evaluation-harness (v0.4 task
# definitions, reproduced here so the pinned environment needs no new
# packages): prompts, the context/continuation split of HFLM._encode_pair, no
# BOS token, summed continuation log-likelihood, acc and acc_norm (log-likelihood
# divided by the character length of the choice), LAMBADA accuracy as a greedy
# match of every continuation token. WikiText-2 perplexity is token-level over
# non-overlapping 1,024-token windows of the raw test split joined by "\n\n"
# (the convention of the compression literature, e.g. SparseGPT).
#
# Tasks and the metric that enters the average (fixed before any result):
#   lambada_openai  test, all 5,153                acc (greedy)
#   hellaswag       validation, every 10th (1,005) acc_norm
#   piqa            validation, all 1,838          acc
#   winogrande      winogrande_xl validation 1,267 acc
#   arc_easy        test, all 2,376                acc
#   arc_challenge   test, all 1,172                acc_norm
#   openbookqa      main test, all 500             acc_norm
import re
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

HELLASWAG_STRIDE = 10         # the validation split is ordered by source; a stride keeps both sources
AVG_METRIC = {"lambada_openai": "acc", "hellaswag": "acc_norm", "piqa": "acc", "winogrande": "acc",
              "arc_easy": "acc", "arc_challenge": "acc_norm", "openbookqa": "acc_norm"}
TOKEN_BUDGET = 12288          # padded tokens per forward batch
CACHE_EVERY = 1               # empty the MPS allocator cache after every batch (memory only; values unchanged)


def _pq(repo, path):
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset"))


def _hs_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def load_tasks(subset=None):
    """{task: list of docs} where a doc is (contexts, continuations, gold, choice_strings).

    Multiple-choice docs share one context across choices except WinoGrande,
    whose choices are different contexts with one shared continuation."""
    T = {}
    df = pd.read_json(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset"), lines=True)
    T["lambada_openai"] = [([t.rsplit(" ", 1)[0]], [" " + t.rsplit(" ", 1)[1]], 0, None) for t in df["text"]]
    df = _pq("Rowan/hellaswag", "data/validation-00000-of-00001.parquet").iloc[::HELLASWAG_STRIDE]
    T["hellaswag"] = []
    for r in df.itertuples():
        q = _hs_pre(r.activity_label + ": " + r.ctx_a + " " + r.ctx_b.capitalize())
        ch = [_hs_pre(e) for e in r.endings]
        T["hellaswag"].append(([q] * 4, [" " + c for c in ch], int(r.label), ch))
    df = _pq("baber/piqa", "piqa_validation.parquet")
    T["piqa"] = [([f"Question: {r.goal}\nAnswer:"] * 2, [" " + r.sol1, " " + r.sol2], int(r.label), [r.sol1, r.sol2]) for r in df.itertuples()]
    df = _pq("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet")
    T["winogrande"] = []
    for r in df.itertuples():
        i = r.sentence.index("_")
        cont = " " + r.sentence[i + 1:].strip()
        T["winogrande"].append(([r.sentence[:i] + r.option1, r.sentence[:i] + r.option2], [cont, cont], {"1": 0, "2": 1}[r.answer], None))
    for name, sub in (("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")):
        df = _pq("allenai/ai2_arc", f"{sub}/test-00000-of-00001.parquet")
        T[name] = []
        for r in df.itertuples():
            texts, labels = list(r.choices["text"]), list(r.choices["label"])
            T[name].append(([f"Question: {r.question}\nAnswer:"] * len(texts), [" " + t for t in texts], labels.index(r.answerKey), texts))
    df = _pq("allenai/openbookqa", "main/test-00000-of-00001.parquet")
    T["openbookqa"] = []
    for r in df.itertuples():
        texts, labels = list(r.choices["text"]), list(r.choices["label"])
        T["openbookqa"].append(([r.question_stem] * len(texts), [" " + t for t in texts], labels.index(r.answerKey.strip()), texts))
    if subset:
        T = {k: v[:subset] for k, v in T.items()}
    return T


def encode_pair(tok, context, continuation):
    """lm-evaluation-harness HFLM._encode_pair: trailing spaces move to the continuation."""
    n = len(context) - len(context.rstrip())
    if n > 0:
        continuation = context[-n:] + continuation
        context = context[:-n]
    whole = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tok(context, add_special_tokens=False)["input_ids"]
    return ctx, whole[len(ctx):]


def build_requests(tok, tasks):
    reqs = []                  # (task, doc index, choice index, ids, n_continuation)
    for t, docs in tasks.items():
        for di, (ctxs, conts, _, _) in enumerate(docs):
            for ci, (c, k) in enumerate(zip(ctxs, conts)):
                a, b = encode_pair(tok, c, k)
                reqs.append((t, di, ci, a + b, len(b)))
    return reqs


def _parts(model):
    base = getattr(model, model.base_model_prefix)
    return base, model.get_output_embeddings()


def _bucket(L):
    """Padded length: multiples of 32 up to 256, of 64 beyond (few distinct shapes keep the GPU kernels cached)."""
    return int(np.ceil(L / 32) * 32) if L <= 256 else int(np.ceil(L / 64) * 64)


@torch.no_grad()
def score_requests(model, reqs, device, pad_id):
    """Summed log-likelihood of each continuation and whether greedy decoding reproduces it.
    Batches have a fixed shape per length bucket (padding rows and positions are masked out)."""
    base, head = _parts(model)
    order = sorted(range(len(reqs)), key=lambda i: -len(reqs[i][3]))
    ll, greedy = np.zeros(len(reqs)), np.zeros(len(reqs), dtype=bool)
    i, nb = 0, 0
    while i < len(order):
        nb += 1
        if device == "mps" and nb % CACHE_EVERY == 0:
            torch.mps.empty_cache()
        Lb = _bucket(len(reqs[order[i]][3])); bs = max(1, TOKEN_BUDGET // Lb)
        idx = order[i:i + bs]; i += len(idx)
        ids = torch.full((bs, Lb), pad_id, dtype=torch.long)
        att = torch.zeros(bs, Lb, dtype=torch.long); att[:, 0] = 1
        for r, j in enumerate(idx):
            s = reqs[j][3]; ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
        h = base(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False).last_hidden_state
        rows, pos, tgt = [], [], []
        for r, j in enumerate(idx):
            s, k = reqs[j][3], reqs[j][4]
            for p in range(len(s) - k - 1, len(s) - 1):
                rows.append(r); pos.append(p); tgt.append(s[p + 1])
        nr = len(rows); npad = int(np.ceil(nr / 512) * 512)
        rows += [0] * (npad - nr); pos += [0] * (npad - nr); tgt += [0] * (npad - nr)
        rows_t, pos_t, tgt_t = (torch.tensor(v, device=device) for v in (rows, pos, tgt))
        logp = torch.log_softmax(head(h[rows_t, pos_t]).float(), -1)
        lp = logp.gather(1, tgt_t[:, None])[:, 0].cpu().numpy()[:nr]
        hit = (logp.argmax(-1) == tgt_t).cpu().numpy()[:nr]
        o = 0
        for r, j in enumerate(idx):
            k = reqs[j][4]
            ll[j] = lp[o:o + k].sum(); greedy[j] = hit[o:o + k].all(); o += k
    if device == "mps":
        torch.mps.empty_cache()
    return ll, greedy


def evaluate_tasks(model, tok, tasks, reqs, device):
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ll, greedy = score_requests(model, reqs, device, pad)
    by = {}
    for n, (t, di, ci, _, _) in enumerate(reqs):
        by.setdefault(t, {}).setdefault(di, {})[ci] = (ll[n], greedy[n])
    out = {}
    for t, docs in tasks.items():
        acc, accn = [], []
        for di, (ctxs, conts, gold, ch) in enumerate(docs):
            v = by[t][di]
            if t == "lambada_openai":
                acc.append(bool(v[0][1])); continue
            lls = np.array([v[c][0] for c in range(len(conts))])
            acc.append(int(lls.argmax()) == gold)
            if ch is not None:
                lens = np.array([float(len(c)) for c in ch])
                accn.append(int((lls / lens).argmax()) == gold)
        rec = {"n": len(acc), "acc": float(np.mean(acc)), "correct": "".join("1" if a else "0" for a in acc)}
        if accn:
            rec["acc_norm"] = float(np.mean(accn)); rec["correct_norm"] = "".join("1" if a else "0" for a in accn)
        out[t] = rec
    out["average"] = float(np.mean([out[t][AVG_METRIC[t]] for t in AVG_METRIC if t in out]))
    return out


def wikitext2_windows(tok, T=1024):
    df = _pq("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")
    ids = tok("\n\n".join(df["text"]), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    n = ids.numel() // T
    return ids[:n * T].view(n, T)


@torch.no_grad()
def perplexity(model, X, device, bs=4):
    base, head = _parts(model)
    nll, cnt = 0.0, 0
    for i in range(0, X.shape[0], bs):
        x = X[i:i + bs].to(device)
        h = base(input_ids=x, use_cache=False).last_hidden_state[:, :-1]
        for r in range(x.shape[0]):
            logp = torch.log_softmax(head(h[r]).float(), -1)
            nll += float(-logp.gather(1, x[r, 1:, None]).sum()); cnt += x.shape[1] - 1
    if device == "mps":
        torch.mps.empty_cache()
    return float(np.exp(nll / cnt))
