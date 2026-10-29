# Check of zeroshot.py's chunked vocabulary projection (HEAD_CHUNK, added for round 27's Qwen2.5-7B, whose
# 152,064-token vocabulary made the unchunked projection of long-answer batches exhaust memory): per-example
# log-likelihoods, greedy matches and task decisions with and without chunking (bfloat16, GPU).
#   python alignment_study/zeroshot_head_chunk_check.py [model] [examples per task]
import json
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
subset = int(sys.argv[2]) if len(sys.argv) > 2 else 300
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
tasks = Z.load_tasks(subset=subset)
reqs = Z.build_requests(tok, tasks)
pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
scores, evals = {}, {}
for chunk in (10 ** 9, 512):
    Z.HEAD_CHUNK = chunk
    scores[chunk] = Z.score_requests(model, reqs, DEV, pad)
    evals[chunk] = Z.evaluate_tasks(model, tok, tasks, reqs, DEV)
(ll0, g0), (ll1, g1) = scores[10 ** 9], scores[512]
a, b = evals[10 ** 9], evals[512]
res = {"model": name, "dtype": "bfloat16", "examples_per_task": subset, "requests": len(reqs),
       "max_abs_loglik_diff": float(np.abs(ll0 - ll1).max()), "identical_loglik": int((ll0 == ll1).sum()),
       "greedy_identical": bool((g0 == g1).all()),
       "decisions_identical": {t: (a[t]["correct"] == b[t]["correct"]) and (a[t].get("correct_norm") == b[t].get("correct_norm")) for t in tasks},
       "average": [a["average"], b["average"]]}
print(json.dumps(res))
json.dump(res, open("alignment_study/zeroshot_head_chunk_check.json", "w"), indent=1)
