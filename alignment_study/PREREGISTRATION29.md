# Preregistration 29: does one merged pair switch on a super weight in other models, and does one window catch it?

Date: 2026-10-30. Frozen by the commit that adds this file, before the confirmatory run.

## Question

The exploratory follow-up to round 28 (alignment_study/NOTE.md, last section;
`llama2_followup_*`) found that on Llama-2-7B converted to half its key-value
heads by the weighted fusion, one merged pair in layer 1 (key-value heads 8 and
18), harmless alone and as good as the best pair by every per-layer measure,
switches the super-weight neuron of layer 1's MLP on at ordinary tokens once
layer 0 is converted; on one pairing of layer 1 perplexity went from 16.8 to
over 12,000. A per-layer criterion cannot see this. This round asks, in four
other models and among conversions that are near-optimal by the fusion's own
residual, how often a different choice of pairs in one early layer damages the
converted model severely, whether one calibration window, read after the first
six layers, flags every such conversion before any evaluation and rarely flags
a good one, whether a simple repair recovers it, and whether the weights alone
point to the merges at risk.

## Instrument

`alignment_study/guard_round.py`, evaluated by `alignment_study/eval_guard.py`;
conversion by `fuse_kv.py` (frozen in round 27) with the pair costs and matching
of `kvfuse.py` (checked in round 28 to reproduce the frozen instrument).

Models (not used in development): mistralai/Mistral-7B-Instruct-v0.2 (32 layers,
32 query and 8 key-value heads, bfloat16), openlm-research/open_llama_3b_v2 (26
layers, 32 heads, multi-head, float32), allenai/OLMo-1B-hf (16 layers, 16 heads,
multi-head, float32), Qwen/Qwen2.5-7B (28 layers, 28 query and 4 key-value heads,
projection biases, bfloat16). Every model is converted to half its key-value
heads (pairs).

Calibration: second moments of every layer's attention input over 128 C4
validation documents of 2,048 tokens (round 28's). Reference conversion: every
layer on the minimum-cost perfect matching under the fusion's residual (for
Qwen2.5-7B, whose projections carry biases and which the pair costs do not
cover, neighbouring pairs).

Population: for each early layer l from 0 to 4, conversions identical to the
reference except in layer l, which takes another matching: the minimum-cost
matching forced to contain one given pair, for every pair (Mistral, 8 key-value
heads), for a sample of 24 pairs drawn with seed 29 (OpenLLaMA-3B and OLMo-1B),
or each of the two other perfect matchings (Qwen2.5-7B, 4 key-value heads);
duplicates of the reference are dropped.

Ground truth: WikiText-2 perplexity on the first 4 windows of 2,048 tokens, as
a multiple of the reference conversion's. Severe: at least 5 times. Benign: at
most 2 times. (Development found no member at 10 times or more in this
population; the severe level is a development choice, see below.)

The check: on one C4 window of 2,048 tokens that is neither calibration nor
evaluation text (the first qualifying document after the calibration
documents), the median over positions after the first of each position's
largest absolute residual-stream entry after each of layers 0 to 5, divided
by the unconverted model's; the member is flagged when the largest of the six
ratios exceeds 2.35. Windows carry no beginning-of-sequence token, as in every
earlier round; the check leaves out only the first position, and its median is
robust to the few other positions that carry massive activations in an
unconverted model (two in Llama-2-7B's development window). The threshold and
the population settings are fixed in the instrument (TAU_FROZEN = 2.35); the
registered run refuses any override.

Repair (flagged members): the pair, in layers up to the first flagged one,
whose heads restored to their unmerged weights lower the check's largest ratio
most is forbidden in its layer, and that layer is re-matched at minimum cost
and re-fused; at most 3 times. Without pair costs (Qwen2.5-7B) the layer takes the
remaining perfect matching that avoids every forbidden pair; the instrument
asserts that this happens only with 4 key-value heads, and a layer with no such
matching left is recorded as a repair failure. The instrument also asserts
that the reference matching never appears as a member.

## Development (labeled; Llama-2-7B only, not scored)

`guard_round.py --dev=NousResearch/Llama-2-7b-hf --pairs=16 --extra=[[1,8,18]]`
(alignment_study/guard_dev/Llama-2-7b-hf.json, guard_dev.log; development
threshold 3.0; a smoke test on Llama-160m in guard_dev/llama-160m.json).
Unconverted perplexity 6.20 on the 4 windows, reference conversion 20.02. 78
members: 16 in layers 0, 1, 2 and 4 and 14 in layer 3 after duplicates were
dropped (layer 1's include the follow-up's pair, 8 with 18). Every member of layers 2 to 4 stays within 1.04 times the
reference; layer 0's largest is 2.23 times. In layer 1, 7 members are damaged
(4.50 to 8.93 times), and in all 7 the forced matching pairs head 8 with head 18
(forcing a pair that takes head 8's or head 18's reference partner, 1 or 11,
re-matches 8 with 18); every other layer-1 member, including head 8 with head 16,
stays within 1.09 times. No member reaches 10 times: the catastrophic value of the
follow-up needed CAGA's whole pairing of layer 1, which also moves head 11. The
largest check ratio is at most 1.86 in all 70 benign members and 2.98 to 3.93 in
the 6 members at 5 times or more (the 4.50-times member, which is 8 with 18
forced, reads 1.69); the threshold 2.35 is the geometric midpoint of that gap.
In all 7 damaged members the dimension that spreads is 2533, the unconverted
model's largest first-position entry after layer 1. The massive-activation
layer is 1, its super neuron 7890 (down_proj[2533, 7890], Yu et al. 2024), its
strongest writers heads 8, 11 and 25 (14.2, 11.1 and 7.5 times the median write
strength). Head 8 gets a new partner in 7 of the 7 damaged members and in 1 of
the 9 benign layer-1 members; any of the three writers does in 7 of 7 and 6 of
9, so the registered rule uses the strongest writer. The 5 flagged members
(development threshold 3.0) were repaired to 1.40 to 1.52 times the reference.
The check took 1.4 seconds per member against 24 seconds for the 4-window
evaluation (about 1,000 seconds for a full WikiText-2 evaluation).

## Registered predictions

Gate (stated first): severe members occur in at least 2 of the 4 models. If
not, G1 to G5 are reported as untested, not as passed.
- G1: the check flags every severe member in the population (at least 3 severe
  members pooled over models; otherwise untested).
- G2: it flags at most 5 percent of benign members pooled over models, and at
  most 10 percent in any model.
- G3: after repair, every flagged member is within 2 times the reference
  conversion's perplexity.
- G4: in every severe member, the residual dimension whose median absolute
  value at ordinary positions grows most, after the layer with the check's
  largest ratio (flagged or not), is the dimension of the unconverted model's
  largest first-position entry after that layer (the damage spreads the model's
  own massive activation).
- G5: the weights point to the merges at risk. Call the massive-activation
  layer the first of the check's layers after which the unconverted model's
  largest first-position entry exceeds 20 times its value after layer 0, its
  super neuron the MLP neuron of that layer whose down-projection weight onto
  that entry's dimension is largest in magnitude (the three largest are
  reported), and a key-value head's write strength the norm of its query heads'
  output projections applied to the neuron's gate weights times the norm
  weights before the MLP (query groups are equal in every model; asserted).
  (a) Every severe member varies a layer at or before the massive-activation
  layer. (b) In models whose massive-activation layer has more than 4 key-value
  heads (with 4, every other matching re-pairs every head), every severe member
  that varies that layer gives its strongest writer a new partner, while at
  most half of the benign members varying that layer do. Models without a
  massive-activation layer among the check's layers are not applicable; G5 is
  untested unless severe members occur in at least 2 applicable models. The
  benign members' rate of varying a layer at or before the massive-activation
  layer is reported beside (a), and the counts behind (b) beside its verdict,
  since a few severe members are weak evidence however they fall; the rule with
  every head at least 5 times the median write strength is reported, not scored.
- Reported, not scored: the number of members, severe members and flags per
  model and layer; the separation of the largest check ratio between severe and
  benign members without a threshold (the fraction of severe-benign pairs
  ordered correctly); members between 2 and 5 times and their ratios; the
  check's time against the evaluation's, including the repair's checks; the
  repaired perplexity of flagged members that were not severe.

## Falsification and reporting

A failed gate means the damage found in Llama-2-7B is rare among these models
at this population; G1 failing means a severe member escaped the check; G2
failing, that the check flags too many good conversions to be useful as
written; G3, that the repair does not recover what the check flags; G4, that
some severe damage goes through another route; G5, that the weights alone do
not point to the merges at risk. Every clause is reported whichever way it
falls.
