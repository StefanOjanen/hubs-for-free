# PREREGISTRATION30 (draft): does the fused start survive uptraining? A multi-head model is converted to half its
# key-value heads without training (rounds 26 and 28), exported as a real grouped-query model (gqa_export.py), and
# its attention projections are trained on held-out C4 text; WikiText-2 perplexity is measured along the way.
# Arms:
#   base           the unconverted model, trained the same way (what the training alone does on this text)
#   mean           neighbouring heads averaged (Ainslie et al. 2023, the standard GQA uptraining start)
#   fuse           the query- and output-weighted fusion on pairs matched by its own residual (round 28's best)
# Training: the attention projections only (q, k, v, o; the shared k and v of a group are one tensor after the
# export), AdamW, linear warmup then cosine decay, gradient clipping 1.0, float32 on the GPU, gradient
# checkpointing; sequences of 1,024 tokens packed from C4 validation shard 0 documents after the 10,000th (the
# calibration documents come earlier), the same sequences in the same order for every arm.
#   python alignment_study/uptrain_round.py --registered=<freeze permalink> | --dev=<model> [--tokens=N] ...
import gc
import json
import math
import os
import sys
import time
import gzip
import numpy as np
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


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


REGISTERED, DEV_MODEL = arg("registered"), arg("dev")
if not REGISTERED and not DEV_MODEL:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION30.md>; use --dev=<model>")
NAME = DEV_MODEL or "allenai/OLMo-1B-hf"
OUT = "alignment_study/uptrain_dev" if DEV_MODEL else "alignment_study/uptrain"
TOKENS = int(float(arg("tokens", "5e6" if not DEV_MODEL else "2e5")))
SEQ, MICRO, ACCUM = 1024, int(arg("micro", 4)), int(arg("accum", 4))          # 16 sequences of 1,024 tokens per step
LR, WARMUP_FRAC, LR_FLOOR = float(arg("lr", 1e-4)), 0.02, 0.1
WARMUP_MIN = int(arg("warmup-min", 10))          # steps; a one-step warmup diverged on TinyLlama (development)
BALANCE = arg("balance", "1") == "1"               # rebalance the fused weights (fuse_balance.py) before training
CHECKPOINTS = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]                                 # fractions of the token budget
N_CAL, T_CAL = (128, 2048) if not DEV_MODEL else (int(arg("ncal", 16)), int(arg("tcal", 1024)))
PPL_T, N_PPL = 1024, (None if not DEV_MODEL else int(arg("pplwin", 40)))
ARMS = tuple(arg("arms", "base,mean,fuse").split(","))
SKIP_DOCS = 10000
ZS_SUBSET = 500 if not DEV_MODEL else int(arg("subset", 100))


def c4_stream(tok, n_tokens, skip=SKIP_DOCS):
    """Token ids of C4 validation shard 0 documents after the first `skip`, joined by EOS, cut into sequences."""
    p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    ids, need = [], n_tokens + SEQ
    with gzip.open(p, "rt") as f:
        for k, line in enumerate(f):
            if k < skip:
                continue
            ids.extend(tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"] + [eos])
            if len(ids) >= need:
                break
    n = len(ids) // SEQ
    if n * SEQ < n_tokens:
        sys.exit(f"only {n * SEQ} tokens available after document {skip}")
    return torch.tensor(ids[:n * SEQ]).view(n, SEQ)


def convert_arm(model, kv, S, arm):
    neighbour = [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)]
    if arm == "base":
        return None
    if arm == "mean":
        groups = neighbour
    else:
        groups = [kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    K.convert(kv, S, groups, "mean" if arm == "mean" else "fuse")
    if arm == "fuse" and BALANCE:
        B.balance(kv, groups)                         # same function, scales a per-parameter optimizer can train
    export_gqa(model, groups, inplace=True)
    return groups


def lr_at(step, total):
    w = max(WARMUP_MIN, int(WARMUP_FRAC * total))
    if step < w:
        return LR * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return LR * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * t)))


def train_and_eval(model, data, win, log, label):
    params = [p for blk in model.model.layers for m in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.self_attn.o_proj)
              for p in m.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})   # frozen embeddings: reentrant would drop the gradients
    model.config.use_cache = False
    per_step = MICRO * ACCUM
    total = TOKENS // (per_step * SEQ)
    marks = {int(round(f * total)): f for f in CHECKPOINTS}
    curve, t0 = [], time.time()

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            ppl = Z.perplexity(model, win, DEV, bs=4)
        model.train()
        curve.append({"step": step, "tokens": step * per_step * SEQ, "ppl": ppl, "seconds": round(time.time() - t0, 1)})
        log(f"  {label}: step {step} of {total}, {step * per_step * SEQ / 1e6:.2f}M tokens, ppl {ppl:.3f}")
    model.train()
    evaluate(0)
    torch.set_grad_enabled(True)                       # the conversion modules switch gradients off at import
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total)
        loss_sum = 0.0
        for a in range(ACCUM):
            x = data[(step * per_step + a * MICRO) % len(data):][:MICRO].to(DEV)
            out = model(input_ids=x, labels=x)
            (out.loss / ACCUM).backward()
            loss_sum += out.loss.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if DEV == "mps":
            torch.mps.empty_cache()
        if step + 1 in marks:
            evaluate(step + 1)
        elif (step + 1) % 50 == 0:
            log(f"  {label}: step {step + 1}, train loss {loss_sum:.4f}")
    torch.set_grad_enabled(False)
    model.gradient_checkpointing_disable()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return curve, total


def run(log, path):
    torch.manual_seed(0)
    partial = path + ".partial"
    res = json.load(open(partial)) if os.path.exists(partial) else {"arms": {}}
    tok = AutoTokenizer.from_pretrained(NAME)
    res.update({"model": NAME, "tokens": TOKENS, "seq": SEQ, "batch_sequences": MICRO * ACCUM, "lr": LR, "warmup": WARMUP_FRAC,
                "lr_floor": LR_FLOOR, "warmup_min": WARMUP_MIN, "balance": BALANCE, "checkpoints": CHECKPOINTS, "calibration": [N_CAL, T_CAL], "ppl_tokens": PPL_T})
    data = c4_stream(tok, TOKENS)
    win = Z.wikitext2_windows(tok, T=PPL_T)[:N_PPL]
    tasks = Z.load_tasks(subset=ZS_SUBSET)
    reqs = Z.build_requests(tok, tasks)
    log(f"{NAME}: {len(data)} training sequences of {SEQ} tokens (C4 validation shard 0 after document {SKIP_DOCS}); "
        f"{TOKENS / 1e6:.2f}M tokens per arm; WikiText-2 {win.shape[0]} windows of {PPL_T}")
    for arm in ARMS:
        if arm in res["arms"]:
            log(f"  {arm}: cached"); continue
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
        kv = K.KV(model)
        groups = None
        if arm != "base":
            cal = C.windows_c4(tok, N_CAL, T_CAL)
            with torch.no_grad():
                S, _ = K.second_moments(model, kv, cal, DEV)
                groups = convert_arm(model, kv, S, arm)
            del S
        log(f"  {arm}: {model.config.num_key_value_heads} key-value heads per layer ({time.time() - t0:.0f}s)")
        curve, steps = train_and_eval(model, data, win, log, arm)
        zs = Z.evaluate_tasks(model, tok, tasks, reqs, DEV)                     # after training (reported, not scored)
        log(f"  {arm}: zero-shot average after training {zs['average']:.4f}")
        res["arms"][arm] = {"curve": curve, "steps": steps, "kv_heads": model.config.num_key_value_heads, "groups": groups,
                            "zeroshot_after": zs, "seconds": round(time.time() - t0, 1)}
        json.dump(res, open(partial, "w"))
        del model, kv                      # kv holds the model too: free both before loading the next copy
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    path = f"{OUT}/{NAME.split('/')[-1]}{'_' + arg('tag') if arg('tag') else ''}.json"
    if os.path.exists(path):
        sys.exit(f"exists: {path}")
    res = run(log, path)
    res["registered"] = REGISTERED; res["dev"] = bool(DEV_MODEL)
    json.dump(res, open(path, "w"), indent=1)
    if os.path.exists(path + ".partial"):
        os.remove(path + ".partial")
    log("UPTRAIN_ROUND_DONE")
