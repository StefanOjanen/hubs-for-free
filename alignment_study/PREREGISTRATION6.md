# Preregistration 6: rerun of the thirteen-model set with the fixed instruments

Date: 2026-09-10. Frozen by commit and pushed to the public repository
before execution; the push timestamp is the ex-ante evidence. Runs locally
on Apple MPS in float32 up to Phi-3-mini and in bfloat16 for the three 7B
models (RUNLOG.md, local platform entry). OLMo-2-7B needs 27 GB of
downloads and runs under this same frozen file when disk allows; its run
date is recorded in RUNLOG.md.

## Purpose

Rounds 1 to 3 established S1 to S4 with instruments that had three known
weaknesses, each recorded as a registered failure or limitation: (a) r1 is
ill-conditioned when generator norms are homogeneous (cv(gn) < 0.1,
tier1_robust.json); (b) the altsink-v2 dissociation control plain-permutes
the rows above a head's target and loses r1 restoration at 32 heads (S3
failed in round 3); (c) the one-column surrogate loses its grip where the
sink sits on a mid-sequence, window-varying column (S2 failed on OLMo-2;
late Qwen2.5-3B layers). This round restates S1 to S4 with the fixed
instruments, a larger window sample from distinct documents, and one
protocol for all thirteen models.

## Development calibration (before freezing; labeled artifacts)

- Toy sweeps (`control_redesign_toy.py`, `control_redesign_toy2.py` and
  their JSON): three controls (the registered altsink-v2; wrapped targets;
  matched-geometry shift) applied to aligned toy ensembles at n in
  {8, 12, 14, 16, 32}, T in {64, 128, 256}, boosts 2 to 8, eight trials
  each. The two robust clauses, rho collapse (rho_ctrl < 0.5 rho_real) and
  commutator elevation (zmed_ctrl > zmed_real), hold in every trial for the
  wrapped-target control at every (n, T) with boost >= 4 (shared energy
  >= 0.67); altsink-v2 dips to 7 of 8 trials at (n = 32, T = 64, boost 4);
  the shift control fails the elevation clause at boost 4 in six rows,
  down to 0 of 8 at (n = 32, T = 256). At boost >= 6 all three pass every
  trial. The r1 restoration clause (r1_ctrl > 0.8) is limited by the
  causal width, not by the control: at T = 64 no control restores r1 for
  n = 32 (median 0.51 to 0.68 at boost 8) although a genuinely misaligned
  toy ensemble reaches 0.90 there, because half of the rows cannot host 32
  distinct targets; at T = 128 restoration is partial (0.38 to 0.75 of
  trials at boost 8); at T = 256 every control restores r1 in 8 of 8
  trials at n = 12, 16 and 32 (median 0.95 to 0.97). The shift control is
  the most faithful on r1 at intermediate concentration, where the
  permuting controls inflate it (n = 8, T = 256, boost 4: shift -0.42,
  misaligned world -0.25, altsink-v2 0.68). Hence: wrapped targets as the
  primary S3' control on the robust clauses at T = 256, r1 restoration
  secondary, shift and altsink-v2 reported alongside.
- Multi-column calibration (`multicolumn_dev_calibration.py`, dev model and
  Qwen2.5-3B): with the sink set defined as the modal column plus columns
  whose layer-level column mass reaches 0.10, at most three columns, every
  high-sink layer of the dev model and layers 3 to 30 of Qwen2.5-3B select
  one column, so the fixed instrument reduces to the registered one there.
  Qwen2.5-3B layers 31 to 33 select two columns and the column-set
  surrogate cuts the sufficiency residual R_r1 from 0.21 to 0.22 (single
  column) to 0.07 to 0.09.
- The development model's values under the new protocol were seen only in
  a smoke test (3 layers, 2 windows) that validated the scripts.

## Models

gpt2, gpt2-medium, EleutherAI/pythia-160m, EleutherAI/pythia-410m,
TinyLlama/TinyLlama_v1.1, Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B,
Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B, microsoft/Phi-3-mini-4k-instruct,
mistralai/Mistral-7B-v0.1, Qwen/Qwen2.5-7B, allenai/OLMo-2-1124-7B.
Qwen2.5-0.5B and Qwen2.5-3B were used to calibrate the instruments; they
are run and reported but excluded from the confirmatory tallies and from
the S4 pool. A model that cannot run on the available machine is recorded
as untestable. A model is testable for S1' and S2' with at least 3
high-sink layers in protocol A, for S3' with at least 3 in protocol B.

## Protocol

Windows: WikiText-103 validation texts of over 40 words that tokenize to at
least T tokens under the model's tokenizer, taken at an even stride
through that pool, each truncated to exactly T tokens. Protocol A: 48
windows, T = 64. Protocol B: 6 windows, T = 256. Statistics in float64 on
the attention probabilities. Modal sink column with the
16-contributing-rows restriction; sink set as defined above; sink mass on
the modal column excluding row 0; high-sink layer: sink mass > 0.4 (mean
over windows). Surrogates per layer and window: 12 plain draws (the z
reference; 8 in protocol B), 8 column-set draws preserving the sink set
and the diagonal, 8 altsink-v2 draws (continuity, secondary), and in
protocol B 8 wrapped-target draws (a permutation of 1..n assigns targets;
every row's maximum is swapped into the head's target, wrapped into the
causal range where needed, the rest permuted) and 8 shift draws
(head-specific cyclic offsets, a permutation of 1..n per draw). Per-layer values are medians over windows (means for sink
mass, set mass and shared energy). Distance ratios at a layer, from the
window-aggregated values: R_r1 = |r1_colfix - r1_real| / max(0.05,
|r1_plain - r1_real|) and R_z = |zmed_colfix - zmed_real| / max(1,
|zmed_real - zmed_plain|). Cosine to the sink-set operator: the normalized
sum of the ideal sink generators at the set's columns, equal to the
registered cosine when the set has one column. Implementation
`rerun_round.py`, evaluation `eval_rerun.py`, both committed with this
file.

## Registered predictions

Primary statistics are the per-pair z against the plain ensemble and the
rho excess over the plain surrogate; r1 is reported with the
ill-conditioning flag and enters only S4.

- S1' (shared sink operator): per model, at 2/3 or more of high-sink layers
  of protocol A, rho_real >= 2 rho_plain AND |cos(S, sink-set operator)|
  > 0.7; holds in at least 2/3 of testable models.
- S2' (column-set sufficiency): per model, at 2/3 or more of high-sink
  layers, R_z < 0.4; holds in at least 2/3 of testable models. Secondary:
  R_r1 < 0.4 at 2/3 or more of high-sink layers. Registered expectation
  for the round-3 exception: OLMo-2, at 19 percent with one column, passes
  S2' if the set captures its sink; if it still fails, the multi-column
  instrument does not resolve window-varying sinks and that is stated.
- S3' (dissociation, wrapped-target control, protocol B): per model, at
  2/3 or more of high-sink layers of protocol B, rho_wrapped < 0.5 rho_real
  AND zmed_wrapped > zmed_real; holds in at least 2/3 of testable models.
  Secondary: r1_wrapped > 0.8 at 2/3 or more of those layers (expected at
  T = 256 for every head count per the toy; reported, not primary); the
  matched-geometry shift control and altsink-v2 reported alongside on the
  same clauses.
- S4 (alignment-fraction law): pooled over high-sink layers of protocol A
  across the testable non-calibration models, restricted to layers with
  cv(gn) >= 0.1: Spearman(shared energy, r1_real) <= -0.5, with a
  bootstrap 95 percent interval over layers; the unrestricted pooled value
  is reported alongside.
- Sanity anchor (not a prediction): the development model's per-layer
  pass pattern under the new protocol agrees with rounds 1 and 2;
  deviations are reported.

## Falsification

S1' or S2' failing in 2 or more testable models rejects the sink-operator
statement under the fixed instruments. S3' failing leaves dissociation
supported synthetically only; S3' passing at 7B on the robust clauses while
the secondary r1 clause fails means the causal width limits r1
restoration, not dissociation. S4 failing demotes the law to the earlier
window sample. Post-hoc narrowings are labeled as such.

## Artifacts

`rerun/<model>.json` per model as it completes (per-layer aggregates plus
per-window arrays of the key statistics for the bootstrap),
`rerun_results.json` from `eval_rerun.py`, run notes in RUNLOG.md.
