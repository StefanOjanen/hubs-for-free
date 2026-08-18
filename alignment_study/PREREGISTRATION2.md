# Preregistration 2: held-out confirmation of the shared sink-operator claim

Date: 2026-08-18. Frozen after dev calibration on Qwen2.5-0.5B
(`tier1_robust.py`, commit d8d525e) and the synthetic theory runs
(`tier2_toy.py`), and BEFORE any held-out model listed below was downloaded
or run. Thresholds below were calibrated on the development model only;
the held-out models are the test. Git history is the ex-ante evidence.

## Held-out models

gpt2-medium (24 layers, 16 heads, MHA), EleutherAI/pythia-410m (24, 16,
MHA), TinyLlama/TinyLlama_v1.1 (22, 32, GQA), Qwen/Qwen2.5-1.5B (28, 12,
GQA), Qwen/Qwen2.5-1.5B-Instruct (instruction-tuned variant). None has
been analyzed in this project before. A model is testable for H1-H3 if it
has at least 3 high-sink layers.

## Protocol (identical machinery to `tier1_robust.py` / `common.py`)

12 WikiText-103 validation windows (texts over 40 whitespace words, stride
40 from index 0), T = 64, float32 weights, eager attention, float64
statistics. Per (layer, window): empirical sink column s (modal argmax of
column mass over heads, columns restricted to at least 16 contributing
rows; this restriction was added after the dev run exposed a near-final
column artifact and is the one protocol change relative to dev); sink mass
= mean attention on column s over rows below it, excluding row 0;
generators G_h, norms gn_h, coupling C, r1, rho, shared mode (S, a, e),
shared-energy fraction; cos(S, ideal sink generator at column s).
Surrogates: 12 plain draws (z reference), 8 sinkfix draws preserving
column s and the diagonal, 8 altsink-v2 draws (distinct target column
t_h = h + 1 per head, row max swapped into t_h, rest permuted). zmed =
median per-pair z of C against the plain ensemble. Per-layer values are
medians over windows (sink mass and shared energy: means). High-sink
layer: sink mass > 0.4. Diagnostic: layers with cv(gn) < 0.1 are flagged
as ill-conditioned for r1 and reported but excluded from no test.

Distance ratios at a layer (sufficiency metric):
R_r1 = |r1_sinkfix - r1_real| / max(0.05, |r1_plain - r1_real|),
R_z = |zmed_sinkfix - zmed_real| / max(1, |zmed_real - zmed_plain|),
with zmed_plain the median z of held-out plain draws against the
reference ensemble (near 0 by construction).

## Registered predictions

- H0 (null sanity): plain r1 > 0.9 in at least 75 percent of all
  (model, layer) pairs pooled over the five models.
- H1 (shared sink operator): per model, at 2/3 or more of high-sink
  layers: rho_real >= 2 x rho_plain AND |cos(S, sink)| > 0.7. Predicted
  to hold in at least 2/3 of testable models.
- H2 (one-column sufficiency): per model, at 2/3 or more of high-sink
  layers: R_r1 < 0.4 AND R_z < 0.4. Predicted in at least 2/3 of
  testable models. (Dev margins: max R_z 0.31, max R_r1 0.25.)
- H3 (dissociation, the A7 replacement): per model, at 2/3 or more of
  high-sink layers: r1_altsink2 > 0.8 AND rho_altsink2 < 0.5 x rho_real
  AND zmed_altsink2 > zmed_real. Predicted in at least 2/3 of testable
  models. (Dev margins: r1_a2 >= 0.95, rho ratio <= 0.35, z elevated at
  every high-sink layer; toy model predicts the same signature.)
- H4 (alignment-fraction law): pooled over all high-sink layers of all
  five models: Spearman(sharedE, r1_real) <= -0.5. H4b (regime edge):
  every pooled high-sink layer with sharedE >= 0.90 has r1_real < 0.2.
  (Dev: the r1/z regime flip sits between sharedE 0.85 and 0.92, where
  the aligned-world toy curve flips.)
- H5 (context transfer): Qwen2.5-0.5B at T = 256 (6 windows, 8 plain
  draws, same statistics): the number of high-sink layers with
  r1_real < 0.3 at T = 256 is greater than or equal to the number at
  T = 64 computed under the identical 6-window protocol in the same
  script.
- H6 (corpus transfer): Qwen2.5-0.5B on HumanEval prompts plus canonical
  solutions (12 windows, T = 64): Jaccard overlap between the layer sets
  {r1_real < 0.3} on code and on WikiText (12-window run in the same
  script) exceeds 0.4.

## Falsification

H1 or H2 failing in 2 or more testable models rejects the sink-operator
claim out of sample. H3 failing leaves the dissociation supported only
synthetically (toy curves). H4 failing demotes the alignment-fraction law
to a dev-model description. H5 or H6 failing restricts the claim to the
tested context length or corpus. Post-hoc narrowings, if any, will be
labeled as such.

## Artifacts

`heldout_round.py` writes `heldout_round.json`; both are committed after
the run with this file's history intact.
