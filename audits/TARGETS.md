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
