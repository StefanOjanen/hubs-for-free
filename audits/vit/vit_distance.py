# PREREGISTRATION19: does attention distance grow with depth in vision
# transformers because late layers attend widely, or because they attend to a
# few artifact tokens? Dosovitskiy et al. (ICLR 2021, Appendix D.7, Figure 11,
# ViT-L/16): "Attention distance was computed for 128 example images by
# averaging the distance between the query pixel and all other pixels,
# weighted by the attention weight"; it increases with depth, and in the second
# half of the network most heads attend widely. Darcet et al. (ICLR 2024) found
# high-norm artifact patch tokens that absorb attention in large ViTs and
# removed them with register tokens.
#
# Per model, image, layer and head, over patch queries and patch keys
# (positions are patch centers, distance in pixels):
#   D_all  attention-weighted mean distance, weights renormalized over patch keys
#   D_raw  the same without renormalizing (attention to CLS or registers counts zero)
#   D_ex   D_all with the layer's artifact columns removed and the rest renormalized
# Artifact columns of an image and layer: patch columns that receive at least
# ten times the uniform share (10 / number of patches) of the patch queries'
# attention, averaged over heads, at most 8, largest first.
# Depth increase: mean over the second half of the layers minus mean over the
# first half. Artifact share of the increase: (Delta_all - Delta_ex) / Delta_all.
#   python audits/vit/vit_distance.py [--dev] --registered=<freeze permalink>
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
from hubsfree.adapters import pick_device, release_memory

DEV_RUN = "--dev" in sys.argv
REGISTERED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--registered=")), None)
if not DEV_RUN and not REGISTERED:
    sys.exit("refusing to run without --registered=<freeze permalink of audits/PREREGISTRATION19.md>; use --dev")
MODELS = ["facebook/dinov2-small", "facebook/dinov2-with-registers-small", "facebook/dinov2-base", "facebook/dinov2-with-registers-base"] if DEV_RUN else \
         ["facebook/dinov2-large", "facebook/dinov2-with-registers-large", "google/vit-large-patch16-224", "google/vit-base-patch16-224"]
N_IMG = 32 if DEV_RUN else 128
OUT = "audits/vit/dev" if DEV_RUN else "audits/vit"
DEV = pick_device("auto")
MAX_ART, ART_FACTOR = 8, 10.0
torch.set_grad_enabled(False)


def images(n, seed=0):
    import io
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    p = hf_hub_download("johnowhitaker/imagenette2-320", "data/train-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p)
    col = [c for c in t.column_names if "image" in c][0]
    idx = np.random.default_rng(seed).permutation(t.num_rows)[:n]
    out = []
    for i in idx:
        v = t.column(col)[int(i)].as_py()
        out.append(Image.open(io.BytesIO(v["bytes"] if isinstance(v, dict) else v)).convert("RGB"))
    return out, [int(i) for i in idx]


def layout(model, n_tokens):
    """(patch start index, grid side, patch size in pixels, number of registers)."""
    cfg = model.config
    regs = int(getattr(cfg, "num_register_tokens", 0) or 0)
    ps = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
    n_p = n_tokens - 1 - regs
    side = int(round(n_p ** 0.5)); assert side * side == n_p, (n_tokens, regs)
    return 1 + regs, side, ps, regs


def preprocessor(name):
    """The model's own preprocessor_config.json applied with PIL and numpy (transformers 5's
    image processors need torchvision): resize (a fixed size, or the shortest edge with the
    given resample filter), center crop, rescale to [0, 1], normalize."""
    from huggingface_hub import hf_hub_download
    from PIL import Image
    c = json.load(open(hf_hub_download(name, "preprocessor_config.json")))
    size, resample = c.get("size", 224), Image.Resampling(c.get("resample", 2))
    mean, std = np.array(c.get("image_mean", [0.5] * 3)), np.array(c.get("image_std", [0.5] * 3))

    def f(im):
        if isinstance(size, int) or "height" in size:
            h = w = size if isinstance(size, int) else size["height"]
            im = im.resize((w, h), resample)
        else:
            se = size["shortest_edge"]; W, H = im.size; r = se / min(W, H)
            im = im.resize((max(se, round(W * r)), max(se, round(H * r))), resample)
        if c.get("do_center_crop"):
            ch, cw = c["crop_size"]["height"], c["crop_size"]["width"]; W, H = im.size
            l, t = (W - cw) // 2, (H - ch) // 2; im = im.crop((l, t, l + cw, t + ch))
        x = np.asarray(im, dtype=np.float64) * c.get("rescale_factor", 1 / 255)
        x = (x - mean) / std
        return torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32)
    return f


def run_model(name, imgs, log):
    from transformers import AutoModel
    proc = preprocessor(name)
    model = AutoModel.from_pretrained(name, attn_implementation="eager", dtype=torch.float32).eval().to(DEV)
    acc = None
    for k, im in enumerate(imgs):
        x = proc(im).to(DEV)
        out = model(pixel_values=x, output_attentions=True)
        atts = out.attentions; L = len(atts); n = atts[0].shape[1]; Ntok = atts[0].shape[-1]
        p0, side, ps, regs = layout(model, Ntok)
        if acc is None:
            rc = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1).reshape(-1, 2).float().to(DEV)
            dist = torch.cdist(rc, rc) * ps                                    # (Np, Np) pixels between patch centers
            Np = side * side
            unif = float(dist.sum() / (Np * (Np - 1)))
            acc = {key: np.zeros((L, n)) for key in ("D_all", "D_raw", "D_ex")}
            acc.update({key: np.zeros(L) for key in ("mass_cls", "mass_reg", "mass_art", "n_art")})
        for l, A in enumerate(atts):
            A = A[0].float()                                                    # (n, Ntok, Ntok)
            Q = A[:, p0:, :]                                                    # patch queries
            P = Q[:, :, p0:]                                                    # patch keys
            colmass = P.mean((0, 1))                                            # (Np,)
            order = colmass.argsort(descending=True)[:MAX_ART]
            art = order[colmass[order] >= ART_FACTOR / Np]
            keep = torch.ones(Np, dtype=torch.bool, device=DEV); keep[art] = False
            raw = (P * dist[None]).sum(-1)                                      # (n, Np)
            acc["D_raw"][l] += raw.mean(-1).cpu().numpy()
            acc["D_all"][l] += (raw / P.sum(-1).clamp_min(1e-12)).mean(-1).cpu().numpy()
            Pk = P * keep[None, None, :].float()
            acc["D_ex"][l] += ((Pk * dist[None]).sum(-1) / Pk.sum(-1).clamp_min(1e-12)).mean(-1).cpu().numpy()
            acc["mass_cls"][l] += float(Q[:, :, 0].mean()); acc["mass_reg"][l] += float(Q[:, :, 1:p0].sum(-1).mean()) if regs else 0.0
            acc["mass_art"][l] += float(P[:, :, art].sum(-1).mean()) if len(art) else 0.0; acc["n_art"][l] += len(art)
        if (k + 1) % 32 == 0:
            log(f"  {name} image {k + 1}/{len(imgs)}")
    del model; release_memory(DEV)
    res = {key: (v / len(imgs)).tolist() for key, v in acc.items()}
    res.update({"layers": L, "heads": n, "patches": side * side, "patch_size": ps, "registers": regs, "uniform_distance": unif})
    half = L // 2
    for key in ("D_all", "D_raw", "D_ex"):
        m = np.array(res[key]).mean(1)
        res[key + "_by_layer"] = m.tolist(); res["delta_" + key] = float(m[half:].mean() - m[:half].mean())
        res["second_half_" + key] = float(m[half:].mean())
    res["artifact_share_of_increase"] = (res["delta_D_all"] - res["delta_D_ex"]) / res["delta_D_all"] if abs(res["delta_D_all"]) > 1e-9 else None
    res["heads_increase_with_depth_all"] = float(np.mean([np.corrcoef(np.arange(L), np.array(res["D_all"])[:, h])[0, 1] > 0 for h in range(n)]))
    return res


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    imgs, idx = images(N_IMG)
    log(f"{len(imgs)} images")
    for name in MODELS:
        path = f"{OUT}/{name.split('/')[-1]}.json"
        if os.path.exists(path):
            print("SKIP (exists)", path, flush=True); continue
        res = run_model(name, imgs, log)
        res.update({"model": name, "registered": REGISTERED, "dev": DEV_RUN, "n_images": len(imgs), "image_rows": idx, "device": DEV,
                    "artifact_rule": {"factor_over_uniform": ART_FACTOR, "max_columns": MAX_ART}, "runtime_s": round(time.time() - t0, 1)})
        json.dump(res, open(path, "w"), indent=1)
        log(f"RESULT {name}: delta D_all {res['delta_D_all']:.1f}px, delta D_ex {res['delta_D_ex']:.1f}px, artifact share {res['artifact_share_of_increase']}, "
            f"second-half D_all {res['second_half_D_all']:.1f} D_ex {res['second_half_D_ex']:.1f} (uniform {res['uniform_distance']:.1f}), "
            f"artifact mass {np.mean(res['mass_art'][res['layers']//2:]):.2f}, registers mass {np.mean(res['mass_reg'][res['layers']//2:]):.2f}")
    print("VIT_DONE", flush=True)
