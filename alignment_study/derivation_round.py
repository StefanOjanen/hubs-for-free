# DERIVATION ROUND (PREREGISTRATION8.md, plan T4.2 and gate G3). Runs the
# derived shared-energy predictors of derivation_dev.py on the twelve
# non-development models, one JSON per model as it completes (finished
# models are skipped on rerun). Protocol: 12 WikiText-103 validation windows,
# stride 40, T = 64, as in derivation_dev.py.
#   python alignment_study/derivation_round.py [--purge-large] [--only=<id>,<id>]
import json
import os
import sys
import numpy as np

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import derivation_dev as D

MODELS = ["gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "TinyLlama/TinyLlama_v1.1", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct",
          "Qwen/Qwen2.5-3B", "microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-v0.1",
          "Qwen/Qwen2.5-7B", "allenai/OLMo-2-1124-7B"]
OUT = "alignment_study/derivation"
PURGE_LARGE = "--purge-large" in sys.argv
ONLY = None
for a in sys.argv:
    if a.startswith("--only="):
        ONLY = set(a.split("=", 1)[1].split(","))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("DEVICE", D.DEV, flush=True)
    for name in MODELS:
        if ONLY is not None and name not in ONLY:
            continue
        short = name.split("/")[-1]
        path = f"{OUT}/{short}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        try:
            rows = D.run(name)
            res = {"model": name, "protocol": {"n_windows": D.NWIN, "stride": D.STRIDE, "seq": D.SEQ, "device": D.DEV}, "rows": rows,
                   "summary_all_layers": D.summarize(rows), "summary_high_sink": D.summarize(rows, True)}
            json.dump(res, open(path, "w"), indent=1)
            hs = res["summary_high_sink"]
            print(f"MODEL_DONE {short} layers={len(rows)} high_sink={hs['n_layers']} R2_hs_ideal={hs['derived_ideal']['pearson_r2_linear'] if hs['n_layers'] > 2 else None}", flush=True)
            if PURGE_LARGE:
                from hubsfree.adapters import param_count
                if param_count(name) >= 3e9:
                    import shutil
                    from huggingface_hub.constants import HF_HUB_CACHE
                    d = os.path.join(HF_HUB_CACHE, "models--" + name.replace("/", "--")); shutil.rmtree(d, ignore_errors=True); print("PURGED cache", d, flush=True)
        except Exception as e:
            import traceback
            print(f"MODEL_FAILED {short}: {type(e).__name__}: {str(e)[:300]}", flush=True); print(traceback.format_exc()[-600:], flush=True)
    print("DERIVATION_ROUND_DONE", flush=True)
