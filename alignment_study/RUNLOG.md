# Scale round run log

Session 1 (2026-09-01, free Colab T4): scale_round.py ran as registered.
Qwen/Qwen2.5-3B completed all 12 windows; its per-layer block was captured
from the cell output and stored as scale_partial/qwen2.5-3b_capture.json.
The runtime was reclaimed by Colab about 65 minutes in, during window 4 of
microsoft/Phi-3-mini-4k-instruct; results for the remaining four models
were lost with the VM (scale_round.json was never retrievable).

Deviation for session 2: scale_round_v2.py reimplements the identical
registered protocol with vectorized surrogate generation (numpy
Generator.permuted over draws x heads per row) and GPU commutator norms
in float32. Surrogate distributions are unchanged; the concrete random
draws differ from v1's stream because the RNG call order differs; the
commutator norms move from float64 to float32, far below draw noise.
Validation against the development model (layers 2, 11, 20, stride-40
windows): real, plain, sinkfix, altsink-v2, and zmed statistics match
tier1_robust.json within window-subset variation, e.g. layer 20 alt2
0.971 (v2) vs 0.974 (tier1), layer 11 alt2 0.985 vs 0.986. A direct
same-window comparison of the two altsink implementations agreed to
three decimals. Model order for session 2 puts the 7B models first so
the highest-value results bank earliest if the session is reclaimed;
per-model results print as MODEL_RESULT blocks for progressive capture.
Evaluation of S1-S4 merges the captured session-1 Qwen2.5-3B block with
the session-2 models.
