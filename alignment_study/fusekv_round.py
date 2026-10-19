# PREREGISTRATION27: reducing the key-value heads of models without training, in models that already share
# key-value heads (grouped-query attention) and in a 3B multi-head model. Neighbouring KV heads are merged in
# groups of 2 (half of the KV heads kept) and, where the count allows, 4; three ways of combining a group:
#   mean     the key and value projections averaged (Ainslie et al. 2023)
#   fuse_u   round 26's fusion without weighting (the objective of weight-space fits)
#   fuse     round 26's fusion weighted by what the heads compute, generalized to KV heads that serve several
#            query heads (fuse_kv.py)
# Calibration: the second moment of every layer's attention input on 8 C4 validation documents of 1,024
# tokens (round 24's). Evaluation: WikiText-2 perplexity and the seven tasks of zeroshot.py on the first 1,000
# examples of each (development: 500). Check: groups of one KV head reproduce the model. Memory: second
# moments kept in float32, the weight snapshot on disk (a 7B dry run with both in memory hung the machine).
#   python alignment_study/fusekv_round.py [--dev] --registered=<freeze permalink> [--only=<model>]
import json
import os
import shutil
import sys
import tempfile
import time
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from hubsfree.adapters import release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of alignment_study/PREREGISTRATION27.md>; use --dev")
if DEV_RUN:
    MODELS, OUT = ["Qwen/Qwen2.5-0.5B", "JackFram/llama-160m"], "alignment_study/fusekv_dev"
else:
    MODELS, OUT = ["TinyLlama/TinyLlama_v1.1", "Qwen/Qwen2.5-1.5B", "openlm-research/open_llama_3b_v2",
                   "mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen2.5-7B"], "alignment_study/fusekv"
BF16 = {"openlm-research/open_llama_3b_v2", "mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen2.5-7B"}
SUBSET = 500 if DEV_RUN else 1000                  # first examples of each task (a 7B model scores about 4 requests a second)
METHODS = ("mean", "fuse_u", "fuse")
N_CAL, T_CAL = 8, 1024
torch.set_grad_enabled(False)


def run_model(name, log, partial_path):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if name in BF16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    sizes = [g for g in (2, 4) if kv.nkv % g == 0 and kv.nkv // g >= 1]
    log(f"{name}: {dtype}, layers {kv.L}, query heads {kv.n}, KV heads {kv.nkv}, head dim {kv.hd}, rotary dims {kv.r}, group sizes {sizes}")
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    x0 = cal[:1, :256].to(DEV)
    ref = model(input_ids=x0).logits.float()
    S, ntok = K.second_moments(model, kv, cal, DEV)
    snapdir = tempfile.mkdtemp(prefix="fusekv_snapshot_")       # attention weights on disk, not in memory

    for l in range(kv.L):
        torch.save([p.detach().cpu() for p in kv.params(l)], f"{snapdir}/{l}.pt")

    def restore():
        for l in range(kv.L):
            for p, s in zip(kv.params(l), torch.load(f"{snapdir}/{l}.pt")):
                p.copy_(s.to(p.device))
    t1 = time.time()
    K.convert(kv, S, [[[i] for i in range(kv.nkv)] for _ in range(kv.L)], "fuse")
    ones_diff = float((model(input_ids=x0).logits.float() - ref).abs().max()); restore()
    log(f"  second moments from {ntok} tokens; groups of one reproduce the model: max logit difference {ones_diff:.2e} ({time.time() - t1:.0f}s)")
    tasks = Z.load_tasks(subset=SUBSET)
    reqs = Z.build_requests(tok, tasks)
    X = Z.wikitext2_windows(tok)
    res = json.load(open(partial_path)) if os.path.exists(partial_path) else {"evals": {}}
    res.update({"model": name, "dtype": str(dtype), "layers": kv.L, "query_heads": kv.n, "kv_heads": kv.nkv, "head_dim": kv.hd,
                "rotary_dims": kv.r, "group_sizes": sizes, "calibration": {"source": "allenai/c4 en validation shard 0", "windows": N_CAL, "tokens": T_CAL},
                "groups_of_one_max_logit_diff": ones_diff, "logit_scale": float(ref.abs().max())})

    def evaluate(key):
        if key in res["evals"]:
            log(f"  {key}: cached"); return
        t2 = time.time()
        r = Z.evaluate_tasks(model, tok, tasks, reqs, DEV); r["wikitext2_ppl"] = Z.perplexity(model, X, DEV)
        r["seconds"] = round(time.time() - t2, 1); res["evals"][key] = r
        json.dump(res, open(partial_path, "w"))
        log(f"  {key}: ppl {r['wikitext2_ppl']:.2f} avg {r['average']:.4f}")
    evaluate("base")
    for g in sizes:
        groups = [[list(range(s, s + g)) for s in range(0, kv.nkv, g)] for _ in range(kv.L)]
        for m in METHODS:
            key = f"g{g}/{m}"
            if key not in res["evals"]:
                t3 = time.time(); K.convert(kv, S, groups, m)
                log(f"  {key}: converted in {time.time() - t3:.0f}s")
            evaluate(key)
            restore()
    res["restore_check_ppl"] = Z.perplexity(model, X, DEV)
    res["runtime_s"] = round(time.time() - t0, 1)
    shutil.rmtree(snapdir, ignore_errors=True)
    del model; release_memory(DEV)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    print("DEVICE", device_label(DEV), flush=True)
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    for name in MODELS:
        if only and name not in only:
            continue
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        partial = path + ".partial"
        res = run_model(name, log, partial)
        res["registered"] = REGISTERED; res["dev"] = DEV_RUN
        json.dump(res, open(path, "w"), indent=1)
        if os.path.exists(partial):
            os.remove(partial)
        log(f"MODEL_DONE {name}")
    print("FUSEKV_ROUND_DONE", flush=True)
