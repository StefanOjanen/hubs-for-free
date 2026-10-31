# Check of kvfuse.py's rebalancing in its bfloat16 path (development model only). Preregistration 30 rebalanced
# weights that were already in bfloat16, which rounded them twice and moved perplexity by 0.7 and 4.2 percent in
# two 7B models (Q1). kvfuse.py fuses and rebalances each layer in float32 and rounds once. This converts a model to
# bfloat16 grouped-query checkpoints with and without rebalancing and compares their WikiText-2 perplexity on 8
# windows of 2,048 tokens as saved and with the attention projections rounded to 4 bits per output channel.
#   python alignment_study/kvfuse_balance_check.py [model]
import json
import os
import shutil
import subprocess
import sys
import tempfile
import torch

sys.path.insert(0, "alignment_study")
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)
torch.set_grad_enabled(False)
name = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama_v1.1"
tok = AutoTokenizer.from_pretrained(name)
win = Z.wikitext2_windows(tok, T=2048)[:8]
res = {"model": name, "windows": 8}
for key, extra in (("balanced", []), ("unbalanced", ["--no-balance"])):
    out = tempfile.mkdtemp(prefix="kvfuse_balance_check_")
    subprocess.run([sys.executable, "alignment_study/kvfuse.py", "--model", name, "--out", out, "--calib", "32", "--calib-len", "1024",
                    "--dtype", "bfloat16"] + extra, check=True, stdout=subprocess.DEVNULL)
    m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    r = {"kv_heads": m.config.num_key_value_heads, "ppl": Z.perplexity(m, win, DEV, bs=2)}
    for blk in m.model.layers:
        a = blk.self_attn
        for mod in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            mod.weight.copy_(P.rtn(mod.weight, 4))
    r["ppl_w4"] = Z.perplexity(m, win, DEV, bs=2)
    res[key] = r
    print(key, r, flush=True)
    del m
    shutil.rmtree(out, ignore_errors=True)
res["ppl_change_from_rebalancing"] = res["balanced"]["ppl"] / res["unbalanced"]["ppl"] - 1
print(json.dumps(res, indent=1))
os.makedirs("alignment_study/balance_dev", exist_ok=True)
json.dump(res, open(f"alignment_study/balance_dev/kvfuse_balance_check_{name.split('/')[-1]}.json", "w"), indent=1)
