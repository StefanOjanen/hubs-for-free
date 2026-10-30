# Exploratory follow-up to round 28 (not preregistered). The bisection (llama2_followup_bisect.py) traced the
# collapse of the weighted fusion on pairs matched by CAGA's residual to layer 1's pairs together with any
# conversion of layer 0: ordinary tokens then leave layer 1 with a residual stream twenty times its usual size, the
# scale Llama-2-7B otherwise gives only the first position (its massive activations). Round 28's other catastrophic
# conversions (averaging, 1,081; CAGA's construction, 323 to 1,348) may fail the same way. For each construction and
# pairing, layers 2 to 31 are converted first, then layer 0, then layer 1 (every layer from the unconverted model's
# second moments, through round 28's frozen conversion functions); perplexity on 8 windows of WikiText-2 at 2,048
# tokens and, on one window of 1,024 tokens, the residual stream at ordinary positions after layers 1 and 2.
#   python alignment_study/llama2_followup_firstlayers.py
import json
import sys
import time
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_kv as K
import zeroshot as Z
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM

argv0 = sys.argv
sys.argv = [argv0[0], "--dev=NousResearch/Llama-2-7b-hf"]            # import the frozen round's functions unchanged
import importlib.util
spec = importlib.util.spec_from_file_location("R", "alignment_study/llama2_round.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
sys.argv = argv0
NAME, REVISION = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
OUT = "alignment_study/llama2_followup_firstlayers.json"
N_WIN, T_PROFILE, BIG = 8, 1024, 100.0
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
ARMS = [("mean", "neighbour"), ("caga", "neighbour"), ("caga", "paired"), ("caga", "caga_matched"),
        ("fuse", "neighbour"), ("fuse", "paired"), ("fuse", "caga_matched")]
torch.set_grad_enabled(False)


class OneLayer:
    """A view of the model as one layer (layer l), so that round 28's convert() converts that layer alone."""
    def __init__(self, kv, l):
        self.__dict__.update(kv.__dict__); self.L, self._kv, self._l = 1, kv, l

    def attn(self, i):
        return self._kv.attn(self._l)

    def has_bias(self, i):
        return self._kv.has_bias(self._l)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    body, _ = Z._parts(model)
    index = json.load(open(hf_hub_download(NAME, "model.safetensors.index.json", revision=REVISION)))["weight_map"]
    files = {}

    def original(l):
        out = []
        for p in PROJ:
            key = f"model.layers.{l}.self_attn.{p}.weight"
            if index[key] not in files:
                files[index[key]] = safe_open(hf_hub_download(NAME, index[key], revision=REVISION), framework="pt", device="cpu")
            out.append(files[index[key]].get_tensor(key).to(torch.bfloat16))
        return out

    def restore():
        for l in range(kv.L):
            for p, t in zip(kv.params(l), original(l)):
                p.copy_(t.to(p.device))
    for l in (0, kv.L - 1):
        assert all(torch.equal(p.detach().cpu(), t) for p, t in zip(kv.params(l), original(l)))
    pairings = {"neighbour": [[[i, i + 1] for i in range(0, kv.nkv, 2)] for _ in range(kv.L)],
                "paired": json.load(open("alignment_study/llama2/Llama-2-7b-hf.json"))["pairing"]["paired"],
                "caga_matched": json.load(open("alignment_study/llama2_followup.json"))["groups"]}
    S, ntok = K.second_moments(model, kv, C.windows_c4(tok, 128, 2048), DEV)
    log(f"second moments from {ntok} tokens")
    win = Z.wikitext2_windows(tok, T=2048)
    xprof, win = win[:1, :T_PROFILE], win[:N_WIN]
    ppl = lambda: Z.perplexity(model, win, DEV, bs=2)

    def stream():
        H = {}
        def mk(l):
            def hook(m, i, o):
                H[l] = (o[0] if isinstance(o, tuple) else o)[0].float().cpu()
            return hook
        hooks = [kv.blocks[l].register_forward_hook(mk(l)) for l in (1, 2)]
        body(input_ids=xprof.to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
        out = {}
        for l in (1, 2):
            mx = H[l][1:].abs().max(1).values
            out[f"after_layer{l}"] = {"median_norm_rest": float(H[l][1:].norm(dim=1).median()),
                                      "positions_above_100": int((mx > BIG).sum()), "first_position_max": float(H[l][0].abs().max())}
        return out

    res = {"model": NAME, "revision": REVISION, "exploratory": True, "windows": N_WIN, "window_tokens": 2048,
           "profile_tokens": T_PROFILE, "base": {"ppl": ppl(), **stream()}, "arms": {}}
    log(f"unconverted: ppl {res['base']['ppl']:.3f}, median norm after layer 1 {res['base']['after_layer1']['median_norm_rest']:.1f}")
    for construction, pairing in ARMS:
        name, rows = f"{construction}/{pairing}", {}
        t = time.time()
        for stage, layers in (("layers_2_to_31", range(2, kv.L)), ("plus_layer_0", [0]), ("all_layers", [1])):
            for l in layers:
                R.convert(OneLayer(kv, l), {0: S[l]}, [pairings[pairing][l]], construction)
            rows[stage] = {"ppl": ppl(), **stream()}
            log(f"{name}, {stage}: ppl {rows[stage]['ppl']:.3f}; after layer 1 median norm {rows[stage]['after_layer1']['median_norm_rest']:.1f}, "
                f"{rows[stage]['after_layer1']['positions_above_100']} positions above {BIG:.0f} ({time.time() - t:.0f}s)")
        res["arms"][name] = rows
        restore()
        json.dump(res, open(OUT, "w"), indent=1)
    res["restore_check_ppl"] = ppl()
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"restored: ppl {res['restore_check_ppl']:.3f}")
    log("FIRSTLAYERS_DONE")
