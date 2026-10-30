# Exploratory probe (not preregistered): does low-bit weight quantization of Llama-2-7B fail through the same unit
# as the key-value head merges of the round 28 follow-up? There, merging two key-value heads of layer 1 (once layer
# 0 was converted) made ordinary tokens take the large activation in residual dimension 2533 that layer 1's MLP
# otherwise writes only for the first position, and perplexity went to the thousands. Round-to-nearest weight
# quantization to 3 bits with one scale per output channel is reported to take Llama-2-7B to WikiText-2 perplexity
# 539 (Shao et al. 2024, OmniQuant, Table 1), against 25.7 for LLaMA-7B. Here every linear layer of the decoder blocks
# is quantized (asymmetric min-max per output channel, or per group), and selected layers or parts of layers are
# kept in bfloat16; perplexity on 8 windows of WikiText-2 at 2,048 tokens and, on one window of 2,048 tokens, how many
# ordinary positions carry an activation above 100 after layers 1 and 2.
#   python alignment_study/llama2_quant_sink_probe.py
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

NAME, REVISION = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
OUT = "alignment_study/llama2_quant_sink_probe.json"
N_WIN, BIG = 8, 100.0
ATTN, MLP = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"), ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
torch.set_grad_enabled(False)


def rtn(w, bits, group=None):
    """Asymmetric min-max round-to-nearest per output channel (group=None) or per group of input columns."""
    W = w.float()
    if group:
        W = W.view(W.shape[0], -1, group)
    lo, hi = W.amin(-1, keepdim=True), W.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(W / scale) + zero, 0, 2 ** bits - 1)
    return ((q - zero) * scale).view(w.shape).to(w.dtype)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    blocks, L = model.model.layers, model.config.num_hidden_layers
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

    def restore():
        for l in range(L):
            for name in ATTN + MLP:
                module(l, name).weight.copy_(original(f"model.layers.{l}.{name}.weight").to(DEV))
    assert torch.equal(module(1, "mlp.down_proj").weight.cpu(), original("model.layers.1.mlp.down_proj.weight"))
    win = Z.wikitext2_windows(tok, T=2048)
    xprof, win = win[:1], win[:N_WIN]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)

    def stream():
        H = {}
        def mk(l):
            def hook(m, i, o):
                H[l] = (o[0] if isinstance(o, tuple) else o)[0].float().cpu()
            return hook
        hooks = [blocks[l].register_forward_hook(mk(l)) for l in (1, 2)]
        model.model(input_ids=xprof.to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
        out = {}
        for l in (1, 2):
            mx = H[l][2:].abs().max(1).values                       # positions after the first two (the first is the sink)
            out[f"after_layer{l}"] = {"positions_above_100": int((mx > BIG).sum()), "median_position_max": float(mx.median()),
                                      "dim2533_median_abs": float(H[l][2:, 2533].abs().median())}
        return out

    # (label, bits, group, kept in bfloat16: list of (layer, part) with part "attn", "mlp" or "all")
    configs = [("w4_channel_all", 4, None, []), ("w3_channel_all", 3, None, []),
               ("w3_channel_keep_layer1", 3, None, [(1, "all")]), ("w3_channel_keep_layers01", 3, None, [(0, "all"), (1, "all")]),
               ("w3_channel_keep_layer1_attn", 3, None, [(1, "attn")]), ("w3_channel_keep_layer1_mlp", 3, None, [(1, "mlp")]),
               ("w3_channel_only_layers01", 3, None, [(l, "all") for l in range(2, L)]),
               ("w3_channel_keep_layer0", 3, None, [(0, "all")]), ("w3_channel_keep_last", 3, None, [(L - 1, "all")]),
               ("w3_g128_all", 3, 128, []), ("w2_g128_all", 2, 128, []), ("w2_g128_keep_layers01", 2, 128, [(0, "all"), (1, "all")])]
    res = {"model": NAME, "revision": REVISION, "exploratory": True, "windows": N_WIN, "window_tokens": 2048,
           "base": {"ppl": ppl(), **stream()}, "configs": {}}
    log(f"unconverted: ppl {res['base']['ppl']:.3f}; after layer 1 {res['base']['after_layer1']}")
    for label, bits, group, keep in configs:
        t = time.time()
        kept = {(l, p) for l, p in keep}
        for l in range(L):
            for name in ATTN + MLP:
                part = "attn" if name in ATTN else "mlp"
                if (l, "all") in kept or (l, part) in kept:
                    continue
                m = module(l, name)
                m.weight.copy_(rtn(m.weight, bits, group))
        r = {"bits": bits, "group": group, "kept_bf16": keep, "ppl": ppl(), **stream()}
        res["configs"][label] = r
        log(f"{label}: ppl {r['ppl']:.3f}; after layer 1 {r['after_layer1']}; after layer 2 {r['after_layer2']} ({time.time() - t:.0f}s)")
        restore()
        json.dump(res, open(OUT, "w"), indent=1)
    res["restore_check_ppl"] = ppl()
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"restored: ppl {res['restore_check_ppl']:.3f}")
    log("QUANT_PROBE_DONE")
