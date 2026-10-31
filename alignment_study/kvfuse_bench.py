# What the exported grouped-query checkpoint saves (numbers for the paper's practical section). The key-value
# cache in bytes at a context length follows from the config (2 x layers x KV heads x head dim x tokens x bytes);
# the wall-clock time of a prefill of P tokens plus D single-token decode steps at batch 1 is measured, original
# model against exported checkpoint, median of R repeats after one warm-up, on the given device.
#   python alignment_study/kvfuse_bench.py --model TinyLlama/TinyLlama_v1.1 --exported <dir> [--prefill 2048]
#       [--decode 128] [--repeats 3] [--dtype bfloat16] [--device mps]
import argparse
import json
import os
import statistics
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import zeroshot as Z
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def cache_bytes(cfg, tokens, dtype):
    n = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", None) or n
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // n
    return 2 * cfg.num_hidden_layers * nkv * hd * tokens * torch.tensor([], dtype=dtype).element_size()


def timed_run(model, ids, decode, dev):
    sync(dev); t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    sync(dev); t1 = time.perf_counter()
    pkv, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(decode):
        out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
        pkv, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    sync(dev); t2 = time.perf_counter()
    return t1 - t0, t2 - t1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--exported", required=True)
    ap.add_argument("--prefill", type=int, default=2048); ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None); ap.add_argument("--revision", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    ids = Z.wikitext2_windows(tok, T=args.prefill)[:1].to(dev)
    res = {"model": args.model, "exported": args.exported, "device": dev, "dtype": args.dtype, "prefill": args.prefill, "decode": args.decode,
           "repeats": args.repeats, "arms": {}}
    for key, path in (("original", args.model), ("exported", args.exported)):
        model = AutoModelForCausalLM.from_pretrained(path, revision=args.revision if key == "original" else None, dtype=dtype,
                                                     attn_implementation="sdpa").eval().to(dev)
        timed_run(model, ids, 8, dev)                                             # warm-up
        runs = [timed_run(model, ids, args.decode, dev) for _ in range(args.repeats)]
        pre, dec = statistics.median(r[0] for r in runs), statistics.median(r[1] for r in runs)
        res["arms"][key] = {"kv_heads": getattr(model.config, "num_key_value_heads", None) or model.config.num_attention_heads,
                            "cache_bytes_at_prefill_plus_decode": cache_bytes(model.config, args.prefill + args.decode, dtype),
                            "prefill_seconds": pre, "decode_tokens_per_second": args.decode / dec,
                            "runs": [{"prefill_seconds": r[0], "decode_seconds": r[1]} for r in runs]}
        print(key, res["arms"][key], flush=True)
        del model
        if dev == "mps":
            torch.mps.empty_cache()
    o, e = res["arms"]["original"], res["arms"]["exported"]
    res["cache_ratio"] = e["cache_bytes_at_prefill_plus_decode"] / o["cache_bytes_at_prefill_plus_decode"]
    res["decode_speedup"] = e["decode_tokens_per_second"] / o["decode_tokens_per_second"]
    os.makedirs("alignment_study/kvfuse_bench", exist_ok=True)
    json.dump(res, open(f"alignment_study/kvfuse_bench/{args.model.split('/')[-1]}.json", "w"), indent=1)
    print(f"cache ratio {res['cache_ratio']:.3f}, decode speedup {res['decode_speedup']:.3f}", flush=True)
    print("KVFUSE_BENCH_DONE", flush=True)
