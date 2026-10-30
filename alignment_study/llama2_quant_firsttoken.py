# Exploratory check (not preregistered), after llama2_quant_sink_probe.py: 3-bit round-to-nearest quantization
# per output channel takes Llama-2-7B to perplexity 399 on 8 windows (unconverted 5.03) without the spurious large
# activations at ordinary positions that the key-value head merges of the round 28 follow-up produce; keeping
# layer 1 in bfloat16 brings it to 68. This records what quantization does to the first position instead: its
# largest activation and its value in dimension 2533 after selected layers, and the attention that every head of
# selected later layers pays to it (the attention sink), unquantized and quantized.
#   python alignment_study/llama2_quant_firsttoken.py
import json
import sys
import time
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download

sys.path.insert(0, "alignment_study")
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

spec = __import__("importlib.util").util.spec_from_file_location("P", "alignment_study/llama2_quant_sink_probe.py")
P = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(P)
NAME, REVISION = P.NAME, P.REVISION
OUT = "alignment_study/llama2_quant_firsttoken.json"
SEEN, ATT_LAYERS, T = (1, 2, 15, 30, 31), (2, 8, 16, 24, 31), 2048
torch.set_grad_enabled(False)

if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    blocks, L, n, hd = model.model.layers, model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads
    index = json.load(open(hf_hub_download(NAME, "model.safetensors.index.json", revision=REVISION)))["weight_map"]
    files = {}

    def original(key):
        if index[key] not in files:
            files[index[key]] = safe_open(hf_hub_download(NAME, index[key], revision=REVISION), framework="pt", device="cpu")
        return files[index[key]].get_tensor(key).to(torch.bfloat16)

    def module(l, name):
        m = blocks[l]
        for part in name.split("."):
            m = getattr(m, part)
        return m
    x = Z.wikitext2_windows(tok, T=T)[:1].to(DEV)
    pos = torch.arange(T, device=DEV)[None]
    mask = torch.ones(T, T, dtype=torch.bool, device=DEV).tril()

    def record():
        H, A = {}, {}
        def out_hook(l):
            def hook(m, i, o):
                H[l] = (o[0] if isinstance(o, tuple) else o)[0].float()
            return hook
        def in_hook(l):
            def hook(m, i, o):
                A[l] = o.float()                                   # the attention input of layer l (after its norm)
            return hook
        hooks = [blocks[l].register_forward_hook(out_hook(l)) for l in SEEN] + \
                [blocks[l].input_layernorm.register_forward_hook(in_hook(l)) for l in ATT_LAYERS]
        model.model(input_ids=x, use_cache=False)
        for h in hooks:
            h.remove()
        r = {"first_position": {str(l): {"max_abs": float(H[l][0].abs().max()), "dim2533": float(H[l][0, 2533])} for l in SEEN},
             "attention_to_first": {}}
        for l in ATT_LAYERS:
            a = blocks[l].self_attn
            cos, sin = model.model.rotary_emb(A[l], position_ids=pos)
            q = a.q_proj(A[l].to(a.q_proj.weight.dtype)).view(1, T, n, hd).transpose(1, 2)
            k = a.k_proj(A[l].to(a.k_proj.weight.dtype)).view(1, T, n, hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            per_head = []
            for h in range(n):
                s = (q[0, h].float() @ k[0, h].float().T) / hd ** 0.5
                per_head.append(float(torch.softmax(s.masked_fill(~mask, float("-inf")), -1)[1:, 0].mean()))
            r["attention_to_first"][str(l)] = {"mean_over_heads": sum(per_head) / n, "per_head": per_head}
        return r

    configs = [("unquantized", None), ("w3_channel_all", []), ("w3_channel_keep_layer1", [(1, "all")]),
               ("w3_channel_only_layers01", [(l, "all") for l in range(2, L)]), ("w3_channel_keep_last", [(L - 1, "all")])]
    res = {"model": NAME, "revision": REVISION, "exploratory": True, "window_tokens": T, "configs": {}}
    for label, keep in configs:
        if keep is not None:
            kept = set(keep)
            for l in range(L):
                for name in P.ATTN + P.MLP:
                    part = "attn" if name in P.ATTN else "mlp"
                    if (l, "all") in kept or (l, part) in kept:
                        continue
                    m = module(l, name)
                    m.weight.copy_(P.rtn(m.weight, 3))
        r = record()
        res["configs"][label] = r
        log(f"{label}: first position after layer 1 max {r['first_position']['1']['max_abs']:.0f} (dim 2533 {r['first_position']['1']['dim2533']:.0f}), "
            f"after layer 30 {r['first_position']['30']['max_abs']:.0f}; attention to the first position, mean over heads: "
            + ", ".join(f"layer {l} {r['attention_to_first'][str(l)]['mean_over_heads']:.3f}" for l in ATT_LAYERS))
        if keep is not None:
            for l in range(L):
                for name in P.ATTN + P.MLP:
                    module(l, name).weight.copy_(original(f"model.layers.{l}.{name}.weight").to(DEV))
        json.dump(res, open(OUT, "w"), indent=1)
    log("FIRSTTOKEN_DONE")
