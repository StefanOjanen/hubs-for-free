# Check of kvfuse.py (CPU, float32): its pairing costs equal the frozen round 28 instrument's, and the checkpoint it
# saves loads with transformers as a grouped-query model whose logits equal the simulated conversion's.
#   python alignment_study/kvfuse_check.py [model]
import json
import os
import subprocess
import sys
import tempfile
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import kvfuse
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
torch.set_flush_denormal(True)
name = sys.argv[1] if len(sys.argv) > 1 else "JackFram/llama-160m"
argv0 = sys.argv
sys.argv = [argv0[0], "--dev=" + name]
import importlib.util
spec = importlib.util.spec_from_file_location("R", "alignment_study/llama2_round.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
sys.argv = argv0
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").eval()
kv = K.KV(model)
cal = C.windows_c4(tok, 8, 512)
S, _ = K.second_moments(model, kv, cal, "cpu")
diffs, same = [], 0
groups = []
for l in range(kv.L):
    a, b = kvfuse.pair_costs(kv, l, S[l].double()), R.pair_costs(kv, l, S[l].double())
    diffs.append(float((a - b).abs().max()))
    pa, pb = kvfuse.min_cost_pairs(a), R.min_cost_pairs(b)
    same += int(pa == pb); groups.append(pa)
out = tempfile.mkdtemp(prefix="kvfuse_check_")
cmd = [sys.executable, "alignment_study/kvfuse.py", "--model", name, "--out", out, "--calib", "8", "--calib-len", "512",
       "--dtype", "float32", "--device", "cpu"]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
saved = json.load(open(os.path.join(out, "kvfuse.json")))
K.convert(kv, S, saved["groups"], "fuse")                       # the simulated conversion, same calibration and groups
x = cal[:2, :256]
ref = model(input_ids=x).logits
new = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32, attn_implementation="sdpa").eval()
got = new(input_ids=x).logits
res = {"model": name, "max_cost_difference_vs_round28": max(diffs), "layers_with_identical_pairs": [same, kv.L],
       "tool_pairs_equal_round28_pairs": saved["groups"] == groups,
       "kv_heads": [kv.nkv, new.config.num_key_value_heads], "max_abs_logit_diff_vs_simulated": float((got - ref).abs().max()),
       "logit_scale": float(ref.abs().max())}
print(json.dumps(res, indent=1))
json.dump(res, open(f"alignment_study/llama2_dev/kvfuse_check_{name.split('/')[-1]}.json", "w"), indent=1)
__import__("shutil").rmtree(out, ignore_errors=True)
