# Development (exploratory, labelled, not preregistered): does keeping the attention heads that write onto the
# super-weight neuron's gate in high precision rescue low-bit round-to-nearest quantization, beyond keeping the super
# weight itself (Yu et al. 2024)? In Llama-2-7B layer 1, KV heads 8, 11 and 25 write most strongly onto neuron 7890's
# gate (round 29's writer statistic, from the weights alone). Every linear layer of the decoder blocks is quantized
# to 3 bits per output channel (asymmetric min-max); kept entries are excluded from their row's range and restored in
# bfloat16. Arms: nothing kept; the super weight down_proj[2533, 7890]; the writer heads' o_proj columns; the same
# number of random heads' o_proj columns (three seeds); writers and super weight together; random heads and super
# weight. Perplexity on 8 WikiText-2 windows of 2,048 tokens.
#   python alignment_study/writer_guard_dev.py
import json
import random
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
OUT = "alignment_study/writer_guard_dev.json"
LAYER, DIM, NEURON = 1, 2533, 7890
N_WIN, BITS = 8, 3
LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
torch.set_grad_enabled(False)


def rtn_keep(w, bits, keep):
    """Per-output-channel asymmetric RTN; entries where keep is True are left out of the row range and kept."""
    W = w.float()
    lo = torch.where(keep, torch.full_like(W, float("inf")), W).amin(1, keepdim=True)
    hi = torch.where(keep, torch.full_like(W, float("-inf")), W).amax(1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8); zero = torch.round(-lo / scale)
    q = (torch.clamp(torch.round(W / scale) + zero, 0, 2 ** bits - 1) - zero) * scale
    return torch.where(keep, W, q).to(w.dtype)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    blocks, L, c = model.model.layers, model.config.num_hidden_layers, model.config
    n, nkv = c.num_attention_heads, c.num_key_value_heads or c.num_attention_heads
    hd, rep = c.hidden_size // n, n // nkv
    mod = lambda l, name: blocks[l].get_submodule(name)
    index = json.load(open(hf_hub_download(NAME, "model.safetensors.index.json", revision=REVISION)))["weight_map"]
    files = {}

    def original(l, name):
        # read from the checkpoint each time: a second copy of a 7B model in host memory would saturate 32 GB
        key = f"model.layers.{l}.{name}.weight"
        if index[key] not in files:
            files[index[key]] = safe_open(hf_hub_download(NAME, index[key], revision=REVISION), framework="pt", device="cpu")
        return files[index[key]].get_tensor(key).to(torch.bfloat16).to(DEV)
    assert torch.equal(mod(1, "mlp.down_proj").weight, original(1, "mlp.down_proj"))
    win = Z.wikitext2_windows(tok, T=2048)[:N_WIN]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)

    # writer statistic from the weights (round 29, guard_round.py)
    b = blocks[LAYER]
    assert int(b.mlp.down_proj.weight[DIM].float().abs().argmax()) == NEURON
    gvec = b.mlp.gate_proj.weight[NEURON].float() * b.post_attention_layernorm.weight.float()
    Wo = b.self_attn.o_proj.weight.float()
    strength = [float(torch.cat([Wo[:, h * hd:(h + 1) * hd].T @ gvec for h in range(i * rep, (i + 1) * rep)]).norm()) for i in range(nkv)]
    med = sorted(strength)[nkv // 2]
    writers = [i for i in range(nkv) if strength[i] >= 5 * med]
    log(f"writers {writers} at {[round(strength[i] / med, 1) for i in writers]} times the median")

    def cols(heads):
        return [h * hd + j for i in heads for h in range(i * rep, (i + 1) * rep) for j in range(hd)]

    others = [i for i in range(nkv) if i not in writers]
    arms = [("none", [], False), ("super_weight", [], True), ("writers", writers, False), ("writers_super", writers, True)]
    for s in range(3):
        rh = sorted(random.Random(s).sample(others, len(writers)))
        arms += [(f"random{s}", rh, False), (f"random{s}_super", rh, True)]

    res = {"model": NAME, "revision": REVISION, "exploratory": True, "bits": BITS, "windows": N_WIN, "writers": writers,
           "strength_over_median": [s / med for s in strength], "base_ppl": ppl(), "arms": {}}
    log(f"unquantized: ppl {res['base_ppl']:.3f}")
    for label, heads, sw in arms:
        for l in range(L):
            for name in LINEAR:
                m = mod(l, name); W = original(l, name)
                keep = torch.zeros_like(W, dtype=torch.bool)
                if l == LAYER and name == "self_attn.o_proj" and heads:
                    keep[:, cols(heads)] = True
                if l == LAYER and name == "mlp.down_proj" and sw:
                    keep[DIM, NEURON] = True
                m.weight.copy_(rtn_keep(W, BITS, keep))
        res["arms"][label] = {"heads": heads, "super_weight": sw, "ppl": ppl()}
        log(f"{label}: heads {heads}, super weight {sw}: ppl {res['arms'][label]['ppl']:.3f}")
        json.dump(res, open(OUT, "w"), indent=1)
    for l in range(L):
        for name in LINEAR:
            mod(l, name).weight.copy_(original(l, name))
    res["restore_check_ppl"] = ppl()
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"restored: ppl {res['restore_check_ppl']:.3f}")
    log("WRITER_GUARD_DEV_DONE")
