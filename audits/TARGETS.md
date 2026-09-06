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
