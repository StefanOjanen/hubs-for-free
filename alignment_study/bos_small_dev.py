# Development (exploratory, labelled, not preregistered): in which small models does prepending a start token help or
# hurt perplexity on sliced WikiText-2 windows, and does the choice of start change how compressions rank? Starts: as is,
# beginning-of-sequence token, end-of-sequence token, the two newlines that separate WikiText articles. Compressions:
# 3-bit per-channel round-to-nearest of q, k, v, o (weights) and the 2-bit KIVI-layout cache (cachebudget_dev.py).
#   python alignment_study/bos_small_dev.py --model=TinyLlama/TinyLlama_v1.1 [--pplwin=8]
import json
import sys
import time
import torch

sys.path.insert(0, "alignment_study")
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.argv.append("--quant=kivi")
import cachebudget_dev as CB
spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)

NAME, N_PPL = CB.arg("model", "TinyLlama/TinyLlama_v1.1"), int(CB.arg("pplwin", 8))
OUT = f"alignment_study/bos_small_dev/{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    win = Z.wikitext2_windows(tok, T=2048)[:N_PPL]
    pre = {"as_is": []}
    if tok.bos_token_id is not None:
        pre["bos"] = [tok.bos_token_id]
    if tok.eos_token_id is not None and tok.eos_token_id != tok.bos_token_id:
        pre["eos"] = [tok.eos_token_id]
    pre["newlines"] = tok("\n\n", add_special_tokens=False)["input_ids"]
    # each start replaces the window's first tokens' slot by shifting the window right and dropping its tail
    wins = {k: torch.cat([torch.tensor(v, dtype=win.dtype).expand(win.shape[0], len(v)), win[:, :win.shape[1] - len(v)]], 1) for k, v in pre.items()}
    ev = lambda: {k: Z.perplexity(model, w, DEV, bs=2) for k, w in wins.items()}
    attn = [b.self_attn for b in model.model.layers]
    res = {"model": NAME, "exploratory": True, "windows": N_PPL, "starts": pre, "note": "perplexity counts every predicted token after the prepended ones, so the starts score slightly different token sets"}
    res["unconverted"] = ev(); log(f"unconverted {res['unconverted']}")
    hooks = []
    for a in attn:
        hooks.append(a.k_proj.register_forward_hook(lambda m, i, o: CB.rtn_channels(o, 2)))
        hooks.append(a.v_proj.register_forward_hook(lambda m, i, o: CB.rtn_tokens(o, a.head_dim if hasattr(a, "head_dim") else model.config.hidden_size // model.config.num_attention_heads, 2)))
    res["cache_2bit_kivi"] = ev(); log(f"2-bit cache {res['cache_2bit_kivi']}")
    for h in hooks:
        h.remove()
    for a in attn:
        for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            m.weight.copy_(P.rtn(m.weight, 3))
    res["weights_3bit_attn"] = ev(); log(f"3-bit attention weights {res['weights_3bit_attn']}")
    json.dump(res, open(OUT, "w"), indent=1)
    log("BOS_SMALL_DEV_DONE")
