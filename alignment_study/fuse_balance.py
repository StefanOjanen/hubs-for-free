# Rebalancing a fused model's weights without changing what it computes (exploratory). The weighted fusion
# (fuse_kv.py) builds each group's shared value in a whitened basis and lets the output projection absorb the
# inverse, and fixes each shared key plane to unit norm in the whitened metric while the queries absorb a complex
# factor per plane. The function is exact, but the scales end up far apart: in TinyLlama's layer 0 the shared value
# rows grow from a median norm of 0.46 to 27.5 and the output columns shrink from 0.51 to 0.006. Such a
# parameterization is hard to train with a per-parameter optimizer and hard to round. Two families of rescalings
# leave the function unchanged: a value row r times d_r with the matching output column of every query head of the
# group divided by d_r, and a rotary plane p of the shared key times c_p with the same plane of every query head of
# the group divided by c_p (the logit of a plane is the product of its query and key, and a real factor commutes
# with the rotation). Choosing d_r and c_p as the square root of the ratio of the two sides' norms gives both sides
# the same norm, the minimum-norm factorization of each product. Biases scale with their rows.
import torch

torch.set_grad_enabled(False)


def _scale_rows(mod, rows, f):
    W = mod.weight
    W[rows] = (W[rows].float() * f[:, None]).to(W.dtype)
    if mod.bias is not None:
        mod.bias[rows] = (mod.bias[rows].float() * f).to(mod.bias.dtype)


def balance(kv, groups_by_layer, layers=None):
    """Rebalance every group of every layer (or of the given layers) in place; groups_by_layer[l] lists the KV heads
    that share one head."""
    for l in (range(kv.L) if layers is None else layers):
        a, hd, half = kv.attn(l), kv.hd, kv.r // 2
        dev = a.v_proj.weight.device
        for G in groups_by_layer[l]:
            G = list(G)
            Q = [h for i in G for h in range(i * kv.rep, (i + 1) * kv.rep)]
            rows = lambda i: torch.arange(i * hd, (i + 1) * hd, device=dev)
            # values: row r of the shared value against column r of every query head's output slice
            Cv = a.v_proj.weight[rows(G[0])].float()
            on = torch.stack([a.o_proj.weight[:, h * hd:(h + 1) * hd].float().norm(dim=0) for h in Q]).pow(2).mean(0).sqrt()
            d = (on / Cv.norm(dim=1).clamp_min(1e-30)).sqrt()
            for i in G:
                _scale_rows(a.v_proj, rows(i), d)
            for h in Q:
                a.o_proj.weight[:, h * hd:(h + 1) * hd] = (a.o_proj.weight[:, h * hd:(h + 1) * hd].float() / d[None, :]).to(a.o_proj.weight.dtype)
            # keys: rotary plane p (dimensions p and p + half) of the shared key against the same plane of every query
            if half:
                B = a.k_proj.weight[rows(G[0])].float()
                kn = (B[:half].norm(dim=1) ** 2 + B[half:2 * half].norm(dim=1) ** 2).sqrt()
                qn = torch.stack([(a.q_proj.weight[rows(h)][:half].float().norm(dim=1) ** 2 +
                                   a.q_proj.weight[rows(h)][half:2 * half].float().norm(dim=1) ** 2) for h in Q]).mean(0).sqrt()
                c = (qn / kn.clamp_min(1e-30)).sqrt()
                cc = torch.cat([c, c, torch.ones(hd - 2 * half, device=dev)])
                for i in G:
                    _scale_rows(a.k_proj, rows(i), cc)
                for h in Q:
                    _scale_rows(a.q_proj, rows(h), 1.0 / cc)
