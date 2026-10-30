# PREREGISTRATION30: rebalancing fused key-value heads, for rounding and for uptraining. The weighted fusion
# (fuse_kv.py) is exact in function but leaves value rows and output columns, and key and query planes, far apart in
# scale (it builds shared heads in a whitened basis). fuse_balance.py rescales each pair of factors to equal norms
# without changing the function. This round tests, on models not used in development:
#   Q  rounding: WikiText-2 perplexity of each model converted to half its key-value heads, unbalanced and
#      rebalanced, with the attention projections as computed and rounded to 4 and 3 bits per output channel
#      (asymmetric round to nearest);
#   T  uptraining: the attention projections trained on C4 text from three starts, averaging, the unbalanced fusion
#      and the rebalanced fusion, at each learning rate of a grid, with perplexity along the way.
#   python alignment_study/balance_round.py --registered=<freeze permalink> [--only=<model>] [--part=Q|T]
#                                           | --dev=<model> [--part=Q|T] [--tokens=N] [--lrs=a,b] ...
import gc
import gzip
import json
import math
import os
import sys
import time
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import fuse_balance as B
import kvfuse
import zeroshot as Z
from gqa_export import export_gqa
from scale_round_v2 import DEV
from hubsfree.adapters import device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)   # rtn(), frozen with its probe


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


REGISTERED, DEV_MODEL = arg("registered"), arg("dev")
if not REGISTERED and not DEV_MODEL:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION30.md>; use --dev=<model>")
if REGISTERED and any(arg(a) for a in ("tokens", "lrs", "pplwin", "ncal", "tcal")):
    sys.exit("the registered run uses the frozen settings")
Q_MODELS = ["allenai/OLMo-1B-hf", "h2oai/h2o-danube-1.8b-base", "openlm-research/open_llama_3b_v2",
            "NousResearch/Llama-2-7b-hf", "mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen2.5-7B"]
T_MODELS = ["allenai/OLMo-1B-hf", "h2oai/h2o-danube-1.8b-base"]
REVISIONS = {"NousResearch/Llama-2-7b-hf": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"}
PART = arg("part", "QT")
OUT = "alignment_study/balance_dev" if DEV_MODEL else "alignment_study/balance"
BF16_ABOVE = 4e9
N_CAL, T_CAL = int(arg("ncal", 128)), int(arg("tcal", 2048))                    # round 28's calibration
N_PPL_Q = int(arg("pplwin", 8))                                                 # WikiText-2 windows of 2,048 (rounding)
TOKENS = int(float(arg("tokens", 3.3e5)))                                       # per training run (20 steps of 16,384)
LRS = [float(x) for x in arg("lrs", "5e-6,2e-5,8e-5,2e-4").split(",")]
SEQ, MICRO, ACCUM = 1024, 4, 4
WARMUP_FRAC, WARMUP_MIN, LR_FLOOR = 0.02, 10, 0.1
CHECKPOINTS = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
N_PPL_T, ZS_SUBSET, SKIP_DOCS = 40, 200, 10000                                  # WikiText-2 windows of 1,024 (training)
torch.set_grad_enabled(False)


def load(name, train=False):
    m = AutoModelForCausalLM.from_pretrained(name, revision=REVISIONS.get(name), dtype=torch.bfloat16, attn_implementation="sdpa")
    n = sum(p.numel() for p in m.parameters())
    if train or n < BF16_ABOVE:
        m = m.float()
    return m.eval().to(DEV), n


def groups_for(kv, S):
    """Pairs of KV heads: the fusion's own minimum-cost matching, or neighbouring pairs where the pair costs do not
    apply (projection biases, or partial rotary embedding)."""
    if kv.has_bias(0) or kv.r != kv.hd:
        return [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)], "neighbour"
    return [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)], "fusion-residual matching"


def run_q(name, log):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, revision=REVISIONS.get(name))
    model, nparam = load(name)
    kv = K.KV(model)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL_Q]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)
    res = {"model": name, "params": nparam, "dtype": str(next(model.parameters()).dtype), "base": ppl()}
    S, _ = K.second_moments(model, kv, C.windows_c4(tok, N_CAL, T_CAL), DEV)
    groups, kind = groups_for(kv, S)
    K.convert(kv, S, groups, "fuse"); del S; gc.collect()
    res["pairing"] = kind
    x = win[:1, :512].to(DEV)
    logits = lambda: model(input_ids=x).logits.float()
    snap = lambda: {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
    put = lambda s: [p.copy_(t) for l in range(kv.L) for p, t in zip(kv.params(l), s[l])]
    def rounded(s, key):
        for bits in (4, 3):
            put(s)
            for l in range(kv.L):
                a = kv.attn(l)
                for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    m.weight.copy_(P.rtn(m.weight, bits))
            res[f"{key}_w{bits}"] = ppl()
    # one copy of the attention weights at a time (two float32 copies of OpenLLaMA-3B's, beside the model, exhausted
    # memory in the first registered attempt); the quantities are those of the frozen version
    L0 = logits().cpu(); s = snap()
    res["fused"] = ppl()
    rounded(s, "fused")
    put(s)
    B.balance(kv, groups)
    del s; gc.collect()
    L1 = logits().cpu(); s = snap()
    res["balanced"] = ppl()
    res["max_logit_change_relative"] = float((L1 - L0).abs().max() / L0.abs().max())
    del L0, L1
    rounded(s, "balanced")
    del s
    log(f"  {name}: unconverted {res['base']:.3f}, fused {res['fused']:.3f}, rebalanced {res['balanced']:.3f} (logits within "
        f"{res['max_logit_change_relative']:.1e}); 4 bits {res['fused_w4']:.3f} against {res['balanced_w4']:.3f}; 3 bits "
        f"{res['fused_w3']:.3f} against {res['balanced_w3']:.3f} ({time.time() - t0:.0f}s)")
    del model, kv; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return res


def c4_stream(tok, n_tokens):
    p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    ids = []
    with gzip.open(p, "rt") as f:
        for k, line in enumerate(f):
            if k < SKIP_DOCS:
                continue
            ids.extend(tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"] + [eos])
            if len(ids) >= n_tokens + SEQ:
                break
    n = len(ids) // SEQ
    return torch.tensor(ids[:n * SEQ]).view(n, SEQ)


def lr_at(step, total, lr):
    w = max(WARMUP_MIN, int(WARMUP_FRAC * total))
    if step < w:
        return lr * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return lr * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * t)))


def train_run(name, tok, start, lr, data, win, log):
    """One uptraining run from a start ('mean', 'fuse' or 'fuse_balanced') at one learning rate."""
    model, _ = load(name, train=True)
    kv = K.KV(model)
    if start != "mean":
        S, _ = K.second_moments(model, kv, C.windows_c4(tok, N_CAL, T_CAL), DEV)
        groups, _ = groups_for(kv, S)
        K.convert(kv, S, groups, "fuse"); del S
        if start == "fuse_balanced":
            B.balance(kv, groups)
    else:
        groups = [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)]
        K.convert(kv, None, groups, "mean")
    export_gqa(model, groups, inplace=True)
    params = [p for blk in model.model.layers for m in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.self_attn.o_proj)
              for p in m.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    per_step = MICRO * ACCUM
    total = TOKENS // (per_step * SEQ)
    marks = {int(round(f * total)) for f in CHECKPOINTS}
    curve, t0 = [], time.time()

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            p_ = Z.perplexity(model, win, DEV, bs=4)
        model.train()
        curve.append({"step": step, "tokens": step * per_step * SEQ, "ppl": p_, "seconds": round(time.time() - t0, 1)})
    model.train()
    evaluate(0)
    torch.set_grad_enabled(True)
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, lr)
        for a in range(ACCUM):
            x = data[(step * per_step + a * MICRO) % len(data):][:MICRO].to(DEV)
            (model(input_ids=x, labels=x).loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if DEV == "mps":
            torch.mps.empty_cache()
        if step + 1 in marks:
            evaluate(step + 1)
    torch.set_grad_enabled(False)
    model.gradient_checkpointing_disable(); model.eval()
    log(f"  {start} at {lr:g}: " + ", ".join(f"{c['ppl']:.2f}" for c in curve) + f" ({time.time() - t0:.0f}s)")
    del model, kv, opt, params; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return {"start": start, "lr": lr, "steps": total, "curve": curve, "final": curve[-1]["ppl"]}


def run_t(name, log, partial):
    tok = AutoTokenizer.from_pretrained(name, revision=REVISIONS.get(name))
    res = json.load(open(partial)) if os.path.exists(partial) else {"model": name, "runs": []}
    data = c4_stream(tok, TOKENS)
    win = Z.wikitext2_windows(tok, T=1024)[:N_PPL_T]
    if "base" not in res:
        m, _ = load(name, train=True)
        res["base"] = Z.perplexity(m, win, DEV, bs=4); del m; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
    done = {(r["start"], r["lr"]) for r in res["runs"]}
    for start in ("mean", "fuse", "fuse_balanced"):
        for lr in LRS:
            if (start, lr) in done:
                continue
            res["runs"].append(train_run(name, tok, start, lr, data, win, log))
            json.dump(res, open(partial, "w"))
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    only = arg("only")
    if "Q" in PART:
        for name in ([DEV_MODEL] if DEV_MODEL else Q_MODELS):
            path = f"{OUT}/Q_{name.split('/')[-1]}.json"
            if (only and name != only) or os.path.exists(path):
                continue
            res = run_q(name, log)
            res.update({"registered": REGISTERED, "dev": bool(DEV_MODEL), "calibration": [N_CAL, T_CAL], "ppl_windows": N_PPL_Q})
            json.dump(res, open(path, "w"), indent=1)
    if "T" in PART:
        for name in ([DEV_MODEL] if DEV_MODEL else T_MODELS):
            path = f"{OUT}/T_{name.split('/')[-1]}.json"
            if (only and name != only) or os.path.exists(path):
                continue
            res = run_t(name, log, path + ".partial")
            res.update({"registered": REGISTERED, "dev": bool(DEV_MODEL), "tokens": TOKENS, "lrs": LRS, "seq": SEQ,
                        "batch_sequences": MICRO * ACCUM, "warmup": [WARMUP_FRAC, WARMUP_MIN], "lr_floor": LR_FLOOR,
                        "checkpoints": CHECKPOINTS, "ppl_windows": N_PPL_T})
            json.dump(res, open(path, "w"), indent=1)
            if os.path.exists(path + ".partial"):
                os.remove(path + ".partial")
    log("BALANCE_ROUND_DONE")
