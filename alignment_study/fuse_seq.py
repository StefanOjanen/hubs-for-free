# Sequential conversion (exploratory). Rounds 26 to 28 fused every layer's key-value heads with the second moment
# of the unconverted model's attention inputs. Here each layer is fused with the second moment of the inputs that
# the already converted layers below it produce, as GPTQ quantizes and SliceGPT slices layer by layer on the
# compressed model's own signals. With alpha > 0 each layer's query, key and value projections are first corrected
# toward the unconverted model's input, W <- W M with M = (1 - alpha) I + alpha E[x0 x^T] (E[x x^T] + lambda I)^-1
# (x0 the unconverted model's attention input, x the converted model's), the closed-form correction of Arai and
# Ichikawa (2025, QEP) for linear layers, so that the converted layer compensates part of the error it receives.
# The calibration windows are carried through the model one layer at a time (one forward pass in all, two with
# alpha > 0); llama layout, no biases.
import torch
import fuse_common as F
import fuse_kv as K

torch.set_grad_enabled(False)


def run_layer(model, blk, h):
    pos = torch.arange(h.shape[1], device=h.device)[None]
    return blk(h, attention_mask=None, position_ids=pos, position_embeddings=model.model.rotary_emb(h, position_ids=pos))


def moments(blk, H, H0, d, device):
    """E[x x^T] of the layer's attention input on the stream H, and E[x0 x^T] against the stream H0."""
    S, X, n = torch.zeros(d, d, dtype=torch.float64), None, 0
    for w in range(len(H)):
        x = blk.input_layernorm(H[w].to(device)).reshape(-1, d).float()
        S += F.host64(x.T @ x)
        if H0 is not None:
            x0 = blk.input_layernorm(H0[w].to(device)).reshape(-1, d).float()
            X = F.host64(x0.T @ x) if X is None else X + F.host64(x0.T @ x)
        n += x.shape[0]
    return S / n, (X / n if X is not None else None)


def sequential_convert(model, kv, windows, device, groups_for, method="fuse", alpha=0.0, damp=0.01, log=print):
    """windows: (N, T) calibration token ids. groups_for(l, S) gives layer l's groups from its (corrected) weights
    and the second moment S of its input on the converted stream. Returns the groups of every layer."""
    emb = model.model.embed_tokens
    H = [emb(windows[w:w + 1].to(device)).cpu() for w in range(windows.shape[0])]
    H0 = [h.clone() for h in H] if alpha > 0 else None
    eye = torch.eye(kv.d, dtype=torch.float64)
    chosen = []
    for l in range(kv.L):
        assert not kv.has_bias(l), "llama layout without biases"
        blk = kv.blocks[l]
        S, X = moments(blk, H, H0, kv.d, device)
        if H0 is not None:
            for w in range(len(H0)):                                               # the unconverted layer, before any change
                H0[w] = run_layer(model, blk, H0[w].to(device)).cpu()                  # (in place: one copy of the stream)
            Sd = S + damp * S.diagonal().mean() * eye
            M = (1 - alpha) * eye + alpha * torch.linalg.solve(Sd, X.T).T        # X Sd^-1 (Sd symmetric)
            a = kv.attn(l)
            for mod in (a.q_proj, a.k_proj, a.v_proj):
                mod.weight.copy_((F.host64(mod.weight) @ M).to(mod.weight.dtype).to(mod.weight.device))
        groups = groups_for(l, S)
        chosen.append([list(G) for G in groups])
        if method == "mean":
            for G in groups:
                K.mean_pool(kv, l, list(G))
        else:
            roots = F.layer_roots(S, kv.d)
            Lsq = roots[0] @ roots[0] if method == "fuse" else None
            for G in groups:
                K.fuse_kv_group(kv, l, list(G), S, roots, qw=(method == "fuse"), ow=(method == "fuse"), Lsq=Lsq)
        for w in range(len(H)):
            H[w] = run_layer(model, blk, H[w].to(device)).cpu()
        if log:
            log(f"  layer {l}: fused on the converted stream" + (f" (input correction alpha {alpha})" if alpha > 0 else ""))
    return chosen
