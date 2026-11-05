# Small-model allocation work (the parallel session, 2026-11-05; exploratory, CPU, 512-token windows)

Committed from private/drafts on 2026-11-05 with ROOT made repository-relative; run from the repository root as
`python alignment_study/alloc_small_dev/<script> <model> [settings]`. Every artifact is labelled exploratory. The design and
its 7B run are in alignment_study/alloc7b_dev.py and NOTE.md.

- alloc_sweep.py (was alloc_dev2_fork.py; a third argument N_HELD, default 8, is its only difference from the original
  alloc_dev_fork.py): each layer's cost at pairs and at groups of four from converting it alone (NLL increase on 8 held-out C4
  windows), an exact dynamic programme at uniform pairs' head count, uniform pairs and three random allocations; 32 calibration
  windows of 512 tokens, 24 WikiText-2 windows of 512 and 8 held-out C4 windows. Artifacts: alloc_dev_TinyLlama_v1.1.json (also
  holds the residual-DP arm), alloc_dev_llama-160m.json, and alloc_dev_Llama-3.2-1B.json, run as `... <snapshot path> 32,512,24,512 8`
  on the local snapshot of unsloth/Llama-3.2-1B (9535bd9b), which is why its model field is a path.
- alloc_control.py (was alloc_control_dev_fork.py): the layer-0-whole control at uniform pairs' head count (layer 0 whole, the rest
  at pairs, the cheapest layers quartered until the head count matches) and layer 0 whole with the dynamic programme on the
  remaining layers; reads the measured costs from the sweep's artifact. Artifacts to come.
- alloc_residual_check.py (was alloc_residual_dev_fork.py): the free residual as the layer selector against the measured costs
  on TinyLlama (Spearman -0.21 at pairs, -0.13 at quads; the residual's allocation worse than uniform pairs).
- alloc_cache_bytes.py and alloc_cache_bytes_tagged.py (were alloc_cache_dev_fork.py and alloc_cache_dev2.py, the second taking
  a model path and a tag): matched cache bytes with KIVI-layout cache rounding against merging; artifacts
  alloc_cache_TinyLlama_v1.1.json and alloc_cache_Llama-3.2-1B.json.

Superadditivity of the selector (actual joint NLL increase on the held-out windows over the summed single-layer costs), per arm:
- alloc_dev_TinyLlama_v1.1.json: uniform_pairs 2.9, dp 3.1, random_0 2.7, random_1 2.7, random_2 3.1
- alloc_dev_llama-160m.json: uniform_pairs 1.8, dp 1.9, random_0 1.6, random_1 1.6, random_2 1.7
- alloc_dev_Llama-3.2-1B.json: uniform_pairs 1.5, dp 1.7, random_0 1.1, random_1 1.1, random_2 1.6

The 7B run (alloc7b_dev/Mistral-7B-v0.1.json) gives 2.7 to 4.2; the three small models give 1.1 to 3.1.
