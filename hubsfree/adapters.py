"""Extract attention maps from Hugging Face models as float64 numpy arrays,
with the device and precision policy shared by every model run in this repo.

Precision policy: float32 whenever the weights fit FP32_HEADROOM of the
device's working memory, otherwise bfloat16 (float16 is never used: Qwen2.5
attention overflows in float16 on MPS). Registered history: rounds 1 and 2
ran float32 on CPU; round 3 ran bfloat16 on a 16 GB T4. Measured on
Qwen2.5-1.5B, 12 windows, T = 64 (RUNLOG.md, local platform entry): MPS
float32 reproduces CPU float32 layer medians to four decimals; MPS bfloat16
deviates by at most 0.055 in r1, 0.019 in shared energy, 0.020 in cos(S, sink)
and changes the empirical sink column in 6 of 336 layer-windows.

Overrides: HUBSFREE_DEVICE (cpu, mps, cuda, auto) and HUBSFREE_DTYPE
(float32, bfloat16) always win, so a run can be pinned from the shell.
"""
import os
import numpy as np

FP32_HEADROOM = 0.75


def pick_device(device=None):
    """Resolve a device name. None keeps the historical default, CPU.
    'auto' picks cuda, then Apple MPS, then cpu."""
    import torch
    dev = os.environ.get("HUBSFREE_DEVICE") or device or "cpu"
    if dev != "auto":
        return dev
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def memory_budget(device):
    """Bytes of working memory the device can give to model weights."""
    import torch
    if device == "cuda":
        return int(torch.cuda.get_device_properties(0).total_memory)
    if device == "mps":
        return int(torch.mps.recommended_max_memory())
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def param_count(name, revision=None):
    """Parameter count from the config alone. The model is instantiated on
    the meta device, so nothing is downloaded or allocated."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(name, revision=revision)
    causal = any("ForCausalLM" in a for a in (cfg.architectures or []))
    with torch.device("meta"):
        m = (AutoModelForCausalLM if causal else AutoModel).from_config(cfg)
    return sum(p.numel() for p in m.parameters())


def pick_dtype(name, device, dtype="float32", revision=None):
    """'auto' applies the precision policy; any other value is used as is
    unless HUBSFREE_DTYPE overrides it."""
    dt = os.environ.get("HUBSFREE_DTYPE") or dtype
    if dt != "auto":
        return dt
    n = param_count(name, revision)
    return "float32" if 4 * n <= FP32_HEADROOM * memory_budget(device) else "bfloat16"


def device_label(device):
    import torch
    if device == "cuda":
        return "cuda:" + torch.cuda.get_device_name(0)
    if device == "mps":
        try:
            import subprocess
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            chip = "Apple silicon"
        return f"mps:{chip} ({memory_budget('mps') / 2**30:.1f} GB working set)"
    return "cpu"


def release_memory(device):
    """Call after `del model` so the next model starts from a clean allocator."""
    import gc
    import torch
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def load_model(name, dtype="float32", device=None, revision=None, causal_lm=False):
    """Historical default: float32 on CPU, base model without a language-model
    head (rounds 1, 2 and the audits). device='auto' and dtype='auto' apply
    the policies above; revision selects a checkpoint branch; causal_lm=True
    returns the model with its head for perplexity work."""
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    dev = pick_device(device)
    dt = pick_dtype(name, dev, dtype, revision)
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    cls = AutoModelForCausalLM if causal_lm else AutoModel
    kw = {"output_attentions": True, "attn_implementation": "eager",
          "dtype": getattr(torch, dt), "revision": revision}
    if dev == "cuda":
        kw["device_map"] = "auto"
    model = cls.from_pretrained(name, **kw).eval()
    if dev != "cuda":
        model = model.to(dev)
    return tok, model


def attentions(tok, model, text, max_length=64):
    """Returns (list over layers of (n_heads, T, T) float64 arrays, token ids)."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    out = model(**ids)
    mats = [a.squeeze(0).float().cpu().numpy().astype(np.float64) for a in out.attentions]
    return mats, ids["input_ids"][0].cpu().numpy()


def wikitext_texts(n, stride=40, min_words=40):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    pool = [t for t in ds["text"] if len(t.split()) > min_words]
    return [pool[i * stride] for i in range(n) if i * stride < len(pool)]
