# Export a converted model as a real grouped-query checkpoint. The conversions of rounds 24 to 28 are simulated:
# every head keeps its own key and value rows, identical within a group, so the cache is not yet smaller. The
# export keeps one key and one value head per group, permutes each layer's query heads (and the matching columns
# of the output projection) so that the heads of a group are adjacent, and sets num_key_value_heads; the function
# is unchanged, and the check compares the exported model's logits with the simulated conversion's.
# Llama layout (q_proj, k_proj, v_proj, o_proj without biases), groups of equal size, multi-head or grouped source.
#   from gqa_export import export_gqa
#   new_model = export_gqa(model, groups_by_layer)        # groups_by_layer[l]: list of groups of KV heads
#   export_gqa(model, groups_by_layer, inplace=True)       # no copy (a 7B model does not fit twice in 32 GB)
import copy
import torch


@torch.no_grad()
def export_gqa(model, groups_by_layer, inplace=False):
    c = model.config
    n, d = c.num_attention_heads, c.hidden_size
    nkv = getattr(c, "num_key_value_heads", None) or n
    hd = getattr(c, "head_dim", None) or d // n
    rep = n // nkv
    sizes = {len(G) for gl in groups_by_layer for G in gl}
    assert len(sizes) == 1, "every group must have the same size"
    g = sizes.pop()
    new_nkv = nkv // g
    new = model if inplace else copy.deepcopy(model)
    new.config.num_key_value_heads = new_nkv
    for l, blk in enumerate(new.model.layers):
        a = blk.self_attn
        groups = [list(G) for G in groups_by_layer[l]]
        assert sorted(i for G in groups for i in G) == list(range(nkv)) and len(groups) == new_nkv
        qorder = [h for G in groups for i in G for h in range(i * rep, (i + 1) * rep)]   # query heads, group by group
        qrows = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in qorder])
        kvrows = torch.cat([torch.arange(G[0] * hd, (G[0] + 1) * hd) for G in groups])  # one KV head per group
        for G in groups:
            for i in G[1:]:
                for m in (a.k_proj, a.v_proj):
                    assert torch.equal(m.weight[i * hd:(i + 1) * hd], m.weight[G[0] * hd:(G[0] + 1) * hd]), \
                        f"layer {l}: KV heads {G} are not merged"
        dev, dt = a.q_proj.weight.device, a.q_proj.weight.dtype
        a.q_proj.weight.data = a.q_proj.weight.data[qrows.to(dev)].clone()
        a.o_proj.weight.data = a.o_proj.weight.data[:, qrows.to(dev)].clone()
        for name in ("k_proj", "v_proj"):
            old = getattr(a, name)
            lin = torch.nn.Linear(d, new_nkv * hd, bias=False, device=dev, dtype=dt)
            lin.weight.data = old.weight.data[kvrows.to(dev)].clone()
            setattr(a, name, lin)
        if hasattr(a, "num_key_value_groups"):
            a.num_key_value_groups = n // new_nkv
        if hasattr(a, "num_key_value_heads"):
            a.num_key_value_heads = new_nkv
    return new
