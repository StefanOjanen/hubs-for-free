# Check of gqa_export.py: convert a model with matched (non-neighbouring) pairs, export it as a real grouped-query
# checkpoint with half the key-value heads, save and reload it with transformers, and compare its logits with the
# simulated conversion's (float32, CPU). Also reports the KV cache size of both.
#   python alignment_study/gqa_export_check.py [model] [ncal]
import json
import os
import sys
import tempfile
import torch

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
from gqa_export import export_gqa
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
torch.set_flush_denormal(True)
name = sys.argv[1] if len(sys.argv) > 1 else "JackFram/llama-160m"
ncal = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="eager").eval()
kv = K.KV(model)
cal = C.windows_c4(tok, ncal, 512)
S, _ = K.second_moments(model, kv, cal, "cpu")
spec = __import__("importlib.util").util.spec_from_file_location("R", "alignment_study/llama2_round.py")
sys.argv = [sys.argv[0], "--dev=" + name]
R = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(R)
paired = [R.min_cost_pairs(R.pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
K.convert(kv, S, paired, "fuse")
x = cal[:2, :256]
ref = model(input_ids=x).logits
new = export_gqa(model, paired)
out_dir = tempfile.mkdtemp(prefix="gqa_export_")
new.save_pretrained(out_dir)
re = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.float32, attn_implementation="eager").eval()
got = re(input_ids=x).logits
kvn = lambda m: sum(p.numel() for blk in m.model.layers for p in (blk.self_attn.k_proj.weight, blk.self_attn.v_proj.weight))
res = {"model": name, "pairs_not_neighbours": int(sum(1 for pl in paired for i, j in pl if not (j == i + 1 and i % 2 == 0))),
       "num_key_value_heads": [model.config.num_key_value_heads if hasattr(model.config, "num_key_value_heads") else kv.n, re.config.num_key_value_heads],
       "kv_projection_params": [kvn(model), kvn(re)], "max_abs_logit_diff": float((got - ref).abs().max()), "logit_scale": float(ref.abs().max()),
       "kv_cache_bytes_per_token_float16": [2 * 2 * kv.L * kv.nkv * kv.hd, 2 * 2 * kv.L * re.config.num_key_value_heads * kv.hd]}
print(json.dumps(res, indent=1))
json.dump(res, open(f"alignment_study/llama2_dev/gqa_export_check_{name.split('/')[-1]}.json", "w"), indent=1)
__import__("shutil").rmtree(out_dir, ignore_errors=True)
