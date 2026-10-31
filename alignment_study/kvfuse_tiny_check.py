# Unit check of the conversion command's extensions on two-layer random models (CPU, seconds): kvfuse.group_cost
# on every pair equals kvfuse.pair_costs' entry; quad_groups' groups of four fuse, export (gqa_export) and reload
# with the same logits; a Qwen2-layout model (projections with biases) goes through pair_costs on the moment of
# [x, 1], the fusion, the rebalancing, the export and the reload with the same logits.
#   python alignment_study/kvfuse_tiny_check.py
import json
import os
import sys
import tempfile
import torch

sys.path.insert(0, "alignment_study")
import fuse_kv as K
import fuse_balance as B
import kvfuse
from gqa_export import export_gqa
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, AutoModelForCausalLM

torch.set_grad_enabled(False)
torch.manual_seed(0)


def run(cfg, cls, group):
    model = cls(cfg).eval()
    kv = K.KV(model)
    ids = torch.randint(0, cfg.vocab_size, (4, 64))
    S, _ = K.second_moments(model, kv, ids, "cpu")
    diff, groups = 0.0, []
    for l in range(kv.L):
        S64 = S[l].double()
        pc, st = kvfuse.pair_costs(kv, l, S64), kvfuse.layer_stats(kv, l, S64)
        for i in range(kv.nkv):
            for j in range(i + 1, kv.nkv):
                diff = max(diff, abs(kvfuse.group_cost(st, [i, j]) - float(pc[i, j])))
        groups.append(kvfuse.quad_groups(kv, l, S64) if group == 4 else kvfuse.min_cost_pairs(pc))
    K.convert(kv, S, groups, "fuse"); B.balance(kv, groups)
    x = ids[:2, :32]
    ref = model(input_ids=x).logits
    new = export_gqa(model, groups)
    d = tempfile.mkdtemp(prefix="kvfuse_tiny_")
    new.save_pretrained(d)
    re = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    out = {"bias": kv.has_bias(0), "S_augmented": S[0].shape[0] == kv.d + 1, "group_cost_vs_pair_costs_max_diff": diff,
           "groups_layer0": groups[0], "kv_heads": [kv.nkv, re.config.num_key_value_heads],
           "export_max_logit_diff": float((new(input_ids=x).logits - ref).abs().max()),
           "reload_max_logit_diff": float((re(input_ids=x).logits - ref).abs().max()), "logit_scale": float(ref.abs().max())}
    __import__("shutil").rmtree(d, ignore_errors=True)
    return out


if __name__ == "__main__":
    llama = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=8, max_position_embeddings=128)
    res = {"llama_quads": run(LlamaConfig(num_key_value_heads=8, **llama), LlamaForCausalLM, 4),
           "llama_pairs": run(LlamaConfig(num_key_value_heads=8, **llama), LlamaForCausalLM, 2),
           "qwen2_pairs_with_biases": run(Qwen2Config(num_key_value_heads=4, **llama), Qwen2ForCausalLM, 2)}
    os.makedirs("alignment_study/llama2_dev", exist_ok=True)
    json.dump(res, open("alignment_study/llama2_dev/kvfuse_tiny_check.json", "w"), indent=1)
    for k, v in res.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "groups_layer0"})
    print("KVFUSE_TINY_CHECK_DONE")
