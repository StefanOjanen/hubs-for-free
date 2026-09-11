# Audit targets (shortlist, 2026-09-06)

Selection rule (plan WS2): a published structural claim about attention
heads, public model, reproducible base result within two days, mix of
classic and recent. Availability verified on 2026-09-06.

| # | Target | Claim audited | Code | Fit | Effort |
|---|---|---|---|---|---|
| 1 | Clark et al. 2019, "What Does BERT Look At?" | Heads cluster by Jensen-Shannon distance between attention distributions; heads in the same layer are similar | Public, MIT (clarkkev/attention-analysis; TensorFlow-era, statistic simple to recompute) | Structural, attention maps | Low |
| 2 | Kovaleva et al. 2019, "Revealing the Dark Secrets of BERT" | Five attention-pattern types; the vertical (shared-column) pattern dominates | Public, Apache-2.0 (text-machine-lab/dark-secrets-of-BERT; notebook-based) | Structural, attention maps | Medium |
| 3 | CHAI, arXiv:2403.08058 (2024) | High redundancy across heads in which tokens they attend to (attention-map correlation), OPT-66B, Llama-7B/33B | None published; method simple enough to reimplement from the paper | Structural, attention maps; practical (KV compression) | Medium |
| 4 | Dewage et al., arXiv:2608.07921 (ICMLA 2026) | Marchenko-Pastur outliers in Q/K/V/O projection weights carry the learned structure (11 models) | Stated URL returns 404 and no matching repo exists under the author account as of 2026-09-06; MP fit reimplementable; contact authors | Structural, weight matrices (battery extended with weight-level nulls) | Medium |
| 5 | Retrieval Heads, arXiv:2404.15574 (2024) | Specific heads copy from context (retrieval score on attention maps); positive control the battery should not remove | Public (nightdessert/Retrieval_Head) | Functional detection statistic on attention maps | Medium |

Alternates, in order: Linear Predictability of Attention Heads
(arXiv:2603.13314, 2026; activation-level object, needs a position-permutation
null; no code URL), Voita et al. 2019 (public code but requires training NMT
models), Michel et al. 2019 and Abnar and Zuidema 2020 (public code; claims
are functional or flow-based rather than head-interaction structure).
Dropped: Bian et al. 2021 (no code available per Papers with Code and the
author's GitHub).

Registered expectation across the set: distribution-level similarity and
redundancy claims (1, 3) survive with reduced effect sizes; the shared
vertical pattern (2) is largely reproduced by column-preserving surrogates;
the weight-level spectral claim (4) is tested against a stronger null than
MP (norm-matched random weights); the positive control (5) survives every
null. Each expectation is falsifiable and is frozen in
PREREGISTRATION4.md before the corresponding battery run.

Integrity: every audit note goes to the original authors with at least four
weeks to respond before any preprint; responses are published alongside.

## Reproduction log

- Target 2 (Kovaleva et al. 2019), attempted 2026-09-11. The pattern
  proportions (heterogeneous 32 to 61 percent by task; vertical 30 percent
  of the annotated sample) come from a CNN trained on about 400 manually
  annotated maps of fine-tuned GLUE models. The repository
  (text-machine-lab/dark-secrets-of-BERT, checked at its current head)
  contains the visualization and [SEP]/[CLS] analysis notebook only: no
  annotations, no classifier weights, no fine-tuned checkpoints. The
  taxonomy proportions are therefore not reproducible under the two-day
  rule and Target 2 is labeled not reproducible in its stated form. The
  notebook's quantitative statistics (per-head mean attention to [CLS],
  per-head maximum attention to [SEP], pre-trained against fine-tuned) are
  reproducible for the pre-trained model but the paper reports them only as
  heat maps, so no numeric reproduction criterion exists; the closest
  quantitative statement of the same fact is Clark et al. 2019's "over half
  of BERT's attention in layers 6 to 10 focuses on [SEP]", already covered
  by Target 1's model and data. Replacement pending from the alternates
  list; the choice is recorded here before any battery run.
- Target 4 (Dewage et al. 2026): stated code URL rechecked 2026-09-11,
  still 404. The MP recipe is fully specified in the paper (gamma =
  max/min, sigma^2 = median(s^2)/(1 + gamma), lambda_plus =
  sigma^2 (1 + sqrt(gamma))^2, outlier s^2 > lambda_plus) and is
  reimplemented in `dewage2026/reproduce.py`; Table II gives per-projection
  outlier fractions for Mistral-7B (Q 87.5, K 74.7, V 43.6, O 84.6 percent),
  the reproduction target. Reproduction scheduled on the Mistral-7B weights.
- Target 3 (CHAI 2024): no code released; the redundancy object (last
  token's attention row per head over C4 contexts, Pearson correlation
  across heads, K-means clusters with within-cluster correlation above
  0.95, correlation increasing with depth) is reimplemented in
  `chai/reproduce.py` (complete linkage at 0.95 and 0.90). Development run
  on Qwen2.5-0.5B committed; reproduction scheduled on a 7B model.
- Target 5 (Retrieval Heads 2024): retrieval score reimplemented in
  `retrieval_heads/reproduce.py` at contexts 1K to 4K (paper: 1K to 50K,
  about 600 instances, 3 to 6 percent of heads above 0.1). Development
  smoke run on Qwen2.5-0.5B at 512 tokens retrieved the needle in 4 of 4
  instances with 11.6 percent of heads above 0.1 (short contexts inflate the
  fraction, as expected). Reproduction scheduled on a 7B model.
- Target 4 reproduction (2026-09-11, `dewage2026/reproduce.py` on
  Mistral-7B-v0.1, 128 matrices, 21 minutes on the CPU): outlier counts per
  matrix match the paper's Table II to within 0.3 percent (Q 1511, K 341,
  V 212, O 1450); the paper's percentages are the outliers' share of
  spectral energy (reproduced within 2.1 percent), not the share of
  singular values (21 to 37 percent here). Reproduced; criteria frozen in
  PREREGISTRATION4_DRAFT.md.
- Target 3 reproduction (2026-09-11, `chai/reproduce.py` on Mistral-7B-v0.1,
  32 C4 documents at T = 1024; a first attempt at T = 2048 ran at about a
  minute per document and was stopped): partially reproduced. High
  cross-head correlation and large 0.95-clusters in layers 1 to 6; a
  majority cluster in only 4 of 32 layers; the paper's increase of
  correlation with depth is not reproduced (Spearman with depth -0.38); the
  correlation follows the last row's sink mass. Criteria frozen in
  PREREGISTRATION4_DRAFT.md.
