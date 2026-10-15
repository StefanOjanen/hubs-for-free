# Preregistration 18: a rule for which attention claims survive null models

Date: 2026-10-15. Frozen by commit and pushed before the confirmatory runs.

## The rule

The audits so far fit one pattern: findings that summarize or compare whole
attention distributions (BERT's layer clustering, CHAI's cross-head
redundancy) are mostly reproduced by nulls that keep the shared columns,
and a finding about where specific heads look (retrieval heads) is not.
Stated as a rule before testing it on new claims:

Let s be the share of a published effect that a null keeping each row's
values and the layer's shared columns (the sink set; [CLS] and [SEP] in BERT)
reproduces, measured from the random-null baseline,
s = (median_colset - median_random) / (real - median_random).
Claims that aggregate attention over many positions or heads into a
distribution-level summary have s >= 0.5. Claims that a head's attention
lands on a token picked out by content or relative position have s <= 0.2.

This rule and the classification of the six claims below were written in
the project plan before any of these claims was computed.

## Six claims, none audited with nulls before

Each statistic follows the source's definition (`audits/rule/rule_round.py`):
- A1, aggregate. H2O (Zhang et al., NeurIPS 2023, Section 3.2 and appendix
  C.8): accumulated attention per position follows a power law. Share of the
  accumulated attention held by the top 10 percent of positions, mean over
  heads and layers. facebook/opt-6.7b (the source's family), 8 WikiText-103
  validation windows of 1,024 tokens.
- A2, aggregate. Scissorhands (Liu et al., NeurIPS 2023, Section 3.2):
  persistence of importance. Pivotal = weight above the row average; t =
  T/2; persistence ratio minus the pivotal share of the first half. Same
  model and windows.
- A3, aggregate. SnapKV (Li et al., NeurIPS 2024, Section 3): the
  observation window's voted positions hit the positions generation attends
  to. Votes over the last 128 prompt rows, top 10 percent of prefix positions
  per head; important = generated-row weight above 1/L_prefix (the source
  does not state its threshold and rate; these are ours). 
  mistralai/Mistral-7B-Instruct-v0.2 (one of the source's models), 6 C4
  prompts of 1,920 tokens and the model's own 128-token greedy continuation.
- L1, located. Induction heads (Olsson et al. 2022): prefix-matching score
  on twice-repeated random tokens (16 sequences of 2 x 128), mean of the
  five best heads. gpt2-medium and EleutherAI/pythia-410m.
- L2, located. Positional heads (Clark et al. 2019, Section 3.1): mean
  attention to the previous token of the four best heads and to the next
  token of the five best heads. bert-base-cased on 200 segments in the
  source's input format (preregistration 17's builder).
- L3, located. Syntactic heads (Clark et al. 2019, Section 4.2): word-level
  maps, [CLS] and [SEP] ignored; per dependency relation the best head's
  accuracy over both directions; mean over twelve relations (UD English EWT
  test, the closest open counterpart of the source's Penn Treebank:
  obj, det, nsubj, amod, advmod, aux, nmod:poss, compound, aux:pass, ccomp,
  mark, compound:prt). bert-base-cased, 500 sentences of 4 to 40 words.

Nulls per map, as in the audits: (a) random softmax rows over the support;
(b) each row permuted over its support with the diagonal fixed; (c) the same
with the shared columns fixed; (d) three untrained initializations. Ten draws
per random family.

## Development (labeled, `audits/rule/dev/`)

Qwen2.5-0.5B for A1 to A3 (2 windows or sequences), gpt2 for L1,
bert-base-uncased for L2 and L3; 3 draws, 1 untrained initialization. All
six effects reproduce. s: A1 0.83, A2 0.04, A3 0.02, L1 0.00, L2 0.01, L3
0.07. The rule as stated holds for 4 of 6 in development; it fails for A2
and A3. A reading suggested by the development data: A1, CHAI's correlation
and Clark's divergence weight each position by its attention mass, which the
sink dominates; A2 and A3 count positions above a threshold, where the sink
is one position among many. The rule and its classification are not changed.

## Registered predictions

- Primary (the rule as stated): in every confirmatory model, s >= 0.5 for
  A1, A2 and A3 and s <= 0.2 for L1, L2 and L3. A claim enters the score only
  if it reproduces: its real value lies above every random draw and every
  untrained initialization. Score: K of the N included claims.
- Secondary, informed by the development run and therefore not a
  prospective test: statistics weighted by attention mass (A1) have s >= 0.5;
  statistics that count positions above a threshold (A2, A3) and located
  claims have s <= 0.2.
- Reported: s for the marginal null and the untrained models, per-layer
  values, Clark et al.'s head counts above one half (L2), per-relation
  accuracies (L3).

## Falsification and reporting

The rule as stated is supported if K = N; each failing claim is reported
with its value. The secondary reading can only be supported or refuted here
on new models, not on new claims; a prospective test of it needs claims not
used in development. `audits/rule/eval_rule.py` writes
`audits/rule_results.json`.
