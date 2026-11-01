# Preregistration 34: does the one-window check choose the early layers' ratio on models it has not seen?

Date: 2026-11-01. Frozen by the commit that adds this file with its population
script and evaluator, before the confirmatory run.

## Question

At a quarter of the key-value heads, Llama-2-7B collapses under either grouping
of four through the hazard of round 28's follow-up, and the one-window check of
round 29 (threshold 2.35, set on that model's pairs) read 576 and 543 on the two
collapses, 0.93 on an averaged conversion that failed without the spike, and at
most 2.1 on Mistral-7B-v0.1's three quarter conversions, none of which collapsed.
On Llama-2-7B the event needs layer 0 quartered: with layer 0 at its pairs and
every other layer quartered, the check reads 0.98 and the model gives 105.5
(development, 2026-11-01). Two claims follow and neither has been tested outside
Llama-2-7B: that the check separates early-layer conversions that collapse
through the event from ones that do not, and that the cheapest setting of the
early layers the check passes is sound.

## Population (round 29's design at a quarter of the heads)

Models: OpenLLaMA-3B v2 (multi-head, 32 heads; round 29's second-strongest
writer onto its super neuron's gate, 5.9 times the median), Mistral-7B-v0.1,
Mistral-7B-Instruct-v0.2 and Qwen2.5-7B, none of which set the threshold or the
rule; Llama-2-7B is the development model of this round
and its members are reported, not scored. For each model, every layer from 2 on
is fixed at the residual's groups of four (Qwen2.5-7B, with 4 key-value heads,
has one group per layer), and layers 0 and 1 each take one of three settings,
unmerged, the residual's pairs, or the residual's groups of four: nine members
per model, 36 scored. Per member, round 29's one-window check on the converted
model (largest ratio over layers 0 to 5) and WikiText-2 perplexity on the first 8
windows of 2,048 tokens (rounds 30 to 32's windows; the cuts below are far
above window noise). Ground truth: a member's damage is its perplexity over the
member with both early layers unmerged, so that the cost of quartering the
other layers, the same in every member, cancels; severe at 5 times or more,
benign at 2 times or less, as in round 29; members in between are reported and
not scored. The check-chosen setting is the unflagged member with the smallest
early-layer cache (unmerged 1, pairs 0.5, groups of four 0.25 of a layer's
heads, summed over the two layers), ties broken by keeping layer 0 the larger
(a fixed order that never sees the outcome). The fixed rule
is "layer 0 at pairs, every other layer at the target ratio", that is the
member (pairs, groups of four), Llama-2-7B's development result, stated here
before any other model's members are seen. Reported, not scored: the half-heads
conversion and the full quartering under neighbouring groups for each model.

## Registered predictions

- R1: the check flags every severe member and no benign member, pooled over the
  four scored models. If no scored model has a severe member, the recall half
  is untestable, not failed, and only the false-alarm half is scored.
- R2: on every scored model the check-chosen setting is not severe.
- R3: on every scored model where the check-chosen setting has a smaller
  early-layer cache than the fixed rule's, it is not severe; if the check
  chooses the fixed rule or dearer on every scored model, R3 is untestable.
- Reported, not scored: Llama-2-7B's nine members; the perplexity of the
  check's choice against the fixed rule's where they differ; the two extra
  conversions per model; run times.

## Prior work and cost

The check detects the super-weight event, not damage in general; the design
above makes the event the only thing that varies between members. Closest
prior work: AsymGQA (Chen et al., arXiv 2406.14963), groups of unequal size
within a layer chosen by an accuracy-guided similarity search with fine-tuning;
what is tested here differs in varying the ratio across layers, choosing it by
an activation hazard check, and using no training. Cost: 11 conversions per
7B model at about 12 minutes each with a 2-minute evaluation, and about 4
minutes each for OpenLLaMA-3B, about 11 GPU hours for the five models; runs
after round 33. If no scored model shows the event, the round says so: that is
evidence about how rare the hazard is. Development test of the instruments
on Llama-160m before the freeze (quad_check_dev/).
