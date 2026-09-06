"""Extract attention maps from Hugging Face models as float64 numpy arrays."""
import numpy as np


def load_model(name, dtype="float32", device=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"output_attentions": True, "attn_implementation": "eager",
          "dtype": getattr(torch, dtype)}
    if device == "cuda":
        kw["device_map"] = "auto"
    model = AutoModel.from_pretrained(name, **kw).eval()
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
