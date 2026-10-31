# Check of the group cost and the two-level matching for groups of four (kvfuse.layer_stats, group_cost,
# quad_groups), CPU, float32, Llama-160m (12 KV heads to 3): group_cost on every pair equals pair_costs' entry; the
# quad groups fuse and the command's checkpoint (--group 4 --pairing matched) loads as a grouped-query model whose
# logits equal the simulated conversion's; WikiText-2 perplexity on 8 windows of 2,048 tokens for the matched quads
# against neighbouring quads (reported).
#   python alignment_study/kvfuse_quad_check.py [model]
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
import zeroshot as Z
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
name = sys.argv[1] if len(sys.argv) > 1 else "JackFram/llama-160m"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").eval()
kv = K.KV(model)
cal = C.windows_c4(tok, 8, 512)
S, _ = K.second_moments(model, kv, cal, "cpu")
diff, quads = 0.0, []
for l in range(kv.L):
    S64 = S[l].double()
    pc, st = kvfuse.pair_costs(kv, l, S64), kvfuse.layer_stats(kv, l, S64)
    for i in range(kv.nkv):
        for j in range(i + 1, kv.nkv):
            diff = max(diff, abs(kvfuse.group_cost(st, [i, j]) - float(pc[i, j])))
    quads.append(kvfuse.quad_groups(kv, l, S64))
neigh = [[list(range(s, s + 4)) for s in range(0, kv.nkv, 4)] for _ in range(kv.L)]
win = Z.wikitext2_windows(tok, T=2048)[:8]
snap = {l: [p.detach().clone() for p in kv.params(l)] for l in range(kv.L)}
put = lambda: [p.copy_(t) for l in range(kv.L) for p, t in zip(kv.params(l), snap[l])]
res = {"model": name, "max_pair_cost_difference": diff, "quads_layer0": quads[0], "base_ppl": Z.perplexity(model, win, "cpu", bs=2)}
K.convert(kv, S, quads, "fuse"); res["matched_quads_ppl"] = Z.perplexity(model, win, "cpu", bs=2); put()
K.convert(kv, S, neigh, "fuse"); res["neighbour_quads_ppl"] = Z.perplexity(model, win, "cpu", bs=2); put()
out = tempfile.mkdtemp(prefix="kvfuse_quad_check_")
cmd = [sys.executable, "alignment_study/kvfuse.py", "--model", name, "--out", out, "--group", "4", "--pairing", "matched",
       "--calib", "8", "--calib-len", "512", "--dtype", "float32", "--device", "cpu"]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
saved = json.load(open(os.path.join(out, "kvfuse.json")))
K.convert(kv, S, saved["groups"], "fuse"); B.balance(kv, saved["groups"])
x = cal[:2, :256]
ref = model(input_ids=x).logits
new = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32, attn_implementation="sdpa").eval()
got = new(input_ids=x).logits
res.update({"tool_groups_equal_quads_here": saved["groups"] == quads, "kv_heads": [kv.nkv, new.config.num_key_value_heads],
            "max_abs_logit_diff_vs_simulated": float((got - ref).abs().max()), "logit_scale": float(ref.abs().max())})
print(json.dumps(res, indent=1), flush=True)
os.makedirs("alignment_study/llama2_dev", exist_ok=True)
json.dump(res, open(f"alignment_study/llama2_dev/kvfuse_quad_check_{name.split('/')[-1]}.json", "w"), indent=1)
__import__("shutil").rmtree(out, ignore_errors=True)
print("KVFUSE_QUAD_CHECK_DONE", flush=True)
