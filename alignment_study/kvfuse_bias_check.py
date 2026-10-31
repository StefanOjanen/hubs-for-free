# Check of the conversion command on a model whose projections carry biases (the Qwen2 layout), CPU, float32:
# kvfuse.pair_costs accepts the second moment of [x, 1], the command converts with matched pairs, and the checkpoint
# it saves loads with transformers as a grouped-query model whose logits equal the simulated conversion's
# (fuse_kv.convert then fuse_balance.balance: the same function). Qwen2.5-0.5B has 2 key-value heads, so the matching
# is trivial; what is checked is the bias path (fuse_kv._rows and _put, fuse_balance._scale_rows, gqa_export).
#   python alignment_study/kvfuse_bias_check.py [model]
import json
import os
import subprocess
import sys
import tempfile
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import fuse_balance as B
import kvfuse
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").eval()
kv = K.KV(model)
assert kv.has_bias(0), "this check is for projections with biases"
cal = C.windows_c4(tok, 8, 512)
S, _ = K.second_moments(model, kv, cal, "cpu")
assert S[0].shape[0] == kv.d + 1
costs = [kvfuse.pair_costs(kv, l, S[l].double()) for l in range(kv.L)]
groups = [kvfuse.min_cost_pairs(c) for c in costs]
out = tempfile.mkdtemp(prefix="kvfuse_bias_check_")
cmd = [sys.executable, "alignment_study/kvfuse.py", "--model", name, "--out", out, "--calib", "8", "--calib-len", "512",
       "--dtype", "float32", "--device", "cpu"]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
saved = json.load(open(os.path.join(out, "kvfuse.json")))
K.convert(kv, S, saved["groups"], "fuse")                     # the simulated conversion, same calibration and groups
B.balance(kv, saved["groups"])
x = cal[:2, :256]
ref = model(input_ids=x).logits
new = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32, attn_implementation="sdpa").eval()
got = new(input_ids=x).logits
res = {"model": name, "balanced": saved.get("balanced", False), "pair_costs_finite": all(bool(torch.isfinite(c).all()) for c in costs),
       "tool_pairs_equal_pairs_here": saved["groups"] == groups, "kv_heads": [kv.nkv, new.config.num_key_value_heads],
       "k_proj_bias_kept": new.model.layers[0].self_attn.k_proj.bias is not None,
       "max_abs_logit_diff_vs_simulated": float((got - ref).abs().max()), "logit_scale": float(ref.abs().max())}
print(json.dumps(res, indent=1), flush=True)
os.makedirs("alignment_study/llama2_dev", exist_ok=True)
json.dump(res, open(f"alignment_study/llama2_dev/kvfuse_bias_check_{name.split('/')[-1]}.json", "w"), indent=1)
__import__("shutil").rmtree(out, ignore_errors=True)
print("KVFUSE_BIAS_CHECK_DONE", flush=True)
