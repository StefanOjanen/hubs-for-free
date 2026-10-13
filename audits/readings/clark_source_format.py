# PREREGISTRATION17, Part K: Target 1 (Clark et al. 2019) in the source's own
# input format. Target 1 (preregistration 4) used bert-base-uncased on 38
# WikiText-103 windows of up to 64 tokens with one [SEP]. Clark et al. used
# BERT-base, cased in the main analysis (their Section 3.2 reports the
# uncased model as a check), on 1000 Wikipedia segments of at most 128
# tokens, [CLS] A [SEP] B [SEP], built by their preprocess_unlabeled.py.
# This script rebuilds the segments with that script's logic (sentence
# units, chunks to 125 wordpieces, a random split into A and B, pair
# truncation, random.seed(0), then a shuffle) on the Wikipedia articles of
# WikiText-103 validation, preregistration 4's text source, detokenized; it
# runs bert-base-cased with segment ids, and computes the registered
# statistic and nulls on the GPU (readings_common.py):
#   D = mean JS(different layer) - mean JS(same layer) over head pairs, with
#   JS averaged over query rows and segments; nn = share of heads whose
#   nearest neighbor is in their own layer.
#   (a) random bidirectional softmax maps; (b) each row permuted off the
#   diagonal; (c) the same with the [CLS] and both [SEP] columns fixed;
#   (d) five untrained bert-base-cased initializations. DRAWS per family.
#   python audits/readings/clark_source_format.py --registered=<freeze permalink>
#   python audits/readings/clark_source_format.py --check   (instrument check on
#       preregistration 4's inputs; reproduces known numbers, writes check_clark.json)
import json
import random
import re
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "audits/readings")
from readings_common import js_matrix, permute_free, free_order, random_maps
from hubsfree.adapters import pick_device

CHECK = "--check" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not CHECK and not REGISTERED:
    sys.exit("refusing to run on new data without --registered=<freeze permalink of audits/PREREGISTRATION17.md>; use --check for the instrument check")
NAME = "bert-base-uncased" if CHECK else "bert-base-cased"
NSEG = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--segments=")), 38 if CHECK else 1000))
DRAWS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--draws=")), 20))
N_INIT, MAXLEN, SEED = 5, 128, 0
OUT = "audits/readings/check_clark.json" if CHECK else "audits/readings/clark_source_format_result.json"
DEV = pick_device("auto")
torch.set_grad_enabled(False)


# ---- Clark et al.'s preprocess_unlabeled.py, prep_document and truncate_seq_pair ----
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_num_tokens:
            break
        trunc = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        if rng.random() < 0.5:
            del trunc[0]
        else:
            trunc.pop()


def prep_document(document, max_sequence_length):
    max_num_tokens = max_sequence_length - 3
    instances, chunk, length, i = [], [], 0, 0
    while i < len(document):
        chunk.append(document[i]); length += len(document[i])
        if i == len(document) - 1 or length >= max_num_tokens:
            if chunk:
                a_end = random.randint(1, len(chunk) - 1) if len(chunk) >= 2 else 1
                a = [t for j in range(a_end) for t in chunk[j]]
                b = [t for j in range(a_end, len(chunk)) for t in chunk[j]]
                truncate_seq_pair(a, b, max_num_tokens, random)
                if len(a) == 0 or len(b) == 0:
                    break
                instances.append((a, b))
            chunk, length = [], 0
        i += 1
    return instances


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
TITLE = re.compile(r"^ = [^=].* = $")


def detokenize(t):
    """The usual WikiText-103 detokenizer (as in Megatron-LM and the LM
    evaluation harness): undo the dataset's spacing of punctuation and its
    @-@, @,@ and @.@ number separators."""
    t = t.replace("s '", "s'")
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for a, b in ((" : ", ": "), (" ; ", "; "), (" . ", ". "), (" ! ", "! "), (" ? ", "? "), (" , ", ", ")):
        t = t.replace(a, b)
    t = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", t)
    t = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", t)
    t = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", t)
    t = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', t)
    t = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", t)
    t = t.replace(" " + chr(176) + " ", chr(176)).replace(" N ", " 1 ").replace(" 's", "'s")
    t = re.sub(r" ([.,;:!?])$", r"\1", t.strip())
    return t


def wikipedia_segments(tok, nseg):
    """Wikipedia articles of WikiText-103 validation (preregistration 4's
    text source), detokenized; article titles split documents, section
    headings are dropped; one sentence per unit; documents in order until
    nseg segments, then the source's shuffle."""
    from datasets import load_dataset
    random.seed(SEED)
    lines = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"]
    docs, cur = [], []
    for line in lines:
        if TITLE.match(line.rstrip("\n")):
            if cur:
                docs.append(cur)
            cur = []
        elif line.strip() and not line.strip().startswith("="):
            cur += [s for s in SENT.split(detokenize(line)) if s]
    if cur:
        docs.append(cur)
    segs = []
    for doc in docs:
        units = [t for t in (tok.tokenize(s) for s in doc) if t]
        for a, b in prep_document(units, MAXLEN):
            segs.append(["[CLS]"] + a + ["[SEP]"] + b + ["[SEP]"])
            if len(segs) >= nseg:
                break
        if len(segs) >= nseg:
            break
    random.shuffle(segs)
    out = []
    for s in segs:
        ids = tok.convert_tokens_to_ids(s); sep1 = s.index("[SEP]")
        tt = [0] * (sep1 + 1) + [1] * (len(s) - sep1 - 1)
        out.append((torch.tensor([ids]), torch.tensor([tt]), (0, sep1, len(s) - 1)))
    return out


def check_inputs(tok):
    """Preregistration 4's inputs: 38 WikiText-103 windows, single sequence, 64 tokens."""
    from hubsfree.adapters import wikitext_texts
    out = []
    for t in wikitext_texts(NSEG):
        e = tok(t, return_tensors="pt", truncation=True, max_length=64)
        out.append((e["input_ids"], e["token_type_ids"], (0, e["input_ids"].shape[1] - 1)))
    return out


def maps(model, ids, tt):
    out = model(input_ids=ids.to(DEV), token_type_ids=tt.to(DEV), output_attentions=True)
    return torch.cat([a[0].float() for a in out.attentions], 0)            # (L*H, T, T)


def stats(JS, layer):
    n = len(layer); same = (layer[:, None] == layer[None, :]) & ~np.eye(n, dtype=bool); diff = layer[:, None] != layer[None, :]
    nn = float(np.mean([layer[np.argsort(JS[k] + np.eye(n)[k] * 1e9)[0]] == layer[k] for k in range(n)]))
    return {"D": float(JS[diff].mean() - JS[same].mean()), "nn_same_layer": nn}


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    t0 = time.time()
    log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModel.from_pretrained(NAME, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads; n = L * H
    layer = np.repeat(np.arange(L), H)
    segs = check_inputs(tok) if CHECK else wikipedia_segments(tok, NSEG)
    lens = [int(s[0].shape[1]) for s in segs]
    log(f"{NAME} segments {len(segs)} tokens mean {np.mean(lens):.1f} max {max(lens)} share at 128 {np.mean(np.array(lens) == MAXLEN):.2f}")
    gen = torch.Generator(device=DEV); gen.manual_seed(SEED)
    J_real = np.zeros((n, n)); J = {f: np.zeros((DRAWS, n, n)) for f in ("a_random", "b_marginal", "c_colfix")}
    for w, (ids, tt, keep) in enumerate(segs):
        A = maps(model, ids, tt); T = A.shape[1]
        J_real += js_matrix(A)
        offdiag = ~torch.eye(T, dtype=torch.bool, device=DEV)
        free_c = offdiag.clone(); free_c[:, list(keep)] = False
        dst_b, dst_c = free_order(offdiag, n), free_order(free_c, n)
        full = torch.ones(T, T, dtype=torch.bool, device=DEV)
        for k in range(DRAWS):
            J["a_random"][k] += js_matrix(random_maps(n, T, T, full, gen, DEV))
            J["b_marginal"][k] += js_matrix(permute_free(A, offdiag, gen, dst_b))
            J["c_colfix"][k] += js_matrix(permute_free(A, free_c, gen, dst_c))
        if w % 50 == 0:
            log(f"  segment {w}")
    cfg = AutoConfig.from_pretrained(NAME); J_unt = []
    for s in range(N_INIT):
        torch.manual_seed(s); um = AutoModel.from_config(cfg, attn_implementation="eager").eval().to(DEV)
        J_unt.append(sum(js_matrix(maps(um, ids, tt)) for ids, tt, _ in segs) / len(segs)); del um
    log("untrained done")
    real = stats(J_real / len(segs), layer)
    res = {"part": "K", "registered": REGISTERED, "check": CHECK, "model": NAME, "n_segments": len(segs), "tokens_mean": float(np.mean(lens)),
           "share_at_128": float(np.mean(np.array(lens) == MAXLEN)), "draws": DRAWS, "device": DEV, "real": real, "nulls": {}}
    fams = {f: [stats(J[f][k] / len(segs), layer) for k in range(DRAWS)] for f in J}
    fams["d_untrained"] = [stats(Ju, layer) for Ju in J_unt]
    for f, vals in fams.items():
        Ds = np.array([v["D"] for v in vals]); med = float(np.median(Ds))
        res["nulls"][f] = {"D_median": med, "D_p5_p95": [float(np.percentile(Ds, 5)), float(np.percentile(Ds, 95))],
                           "percentile_of_real": float((Ds < real["D"]).mean() * 100), "shrinkage": med / real["D"] if real["D"] else None,
                           "nn_median": float(np.median([v["nn_same_layer"] for v in vals])), "n": len(vals)}
    p95 = all(res["nulls"][f]["percentile_of_real"] > 95 for f in res["nulls"])
    shrink_bc = max(res["nulls"]["b_marginal"]["shrinkage"] or 0, res["nulls"]["c_colfix"]["shrinkage"] or 0)
    res["outcome"] = ("survives" if p95 and shrink_bc < 0.5 else "shrinks" if any(res["nulls"][f]["percentile_of_real"] > 95 for f in res["nulls"]) and shrink_bc >= 0.5 else "matched")
    if CHECK:
        reg = json.load(open("audits/clark2019/battery_result.json"))
        res["registered_battery"] = {"real": reg["real"], "b_median": reg["nulls"]["b_marginal"]["D_median"], "c_median": reg["nulls"]["c_colfix"]["D_median"]}
        res["check_abs_diff"] = {"D": abs(real["D"] - reg["real"]["D"]), "nn": abs(real["nn_same_layer"] - reg["real"]["nn_same_layer"]),
                                 "b_median": abs(res["nulls"]["b_marginal"]["D_median"] - reg["nulls"]["b_marginal"]["D_median"]),
                                 "c_median": abs(res["nulls"]["c_colfix"]["D_median"] - reg["nulls"]["c_colfix"]["D_median"])}
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "nulls"}, indent=1)); print(json.dumps(res["nulls"], indent=1)); print("CLARK_READING_DONE", OUT)
