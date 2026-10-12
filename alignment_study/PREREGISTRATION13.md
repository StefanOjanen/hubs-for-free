# Preregistration 13: what the sink does

Date: 2026-10-13. Frozen by commit and pushed before the confirmatory run.

## Question

Round 11 found that the first-token sink, small per position, still
supplies about half of each head's output norm. The literature describes
attention to the sink as heads choosing to do nothing. This round edits the
sink directly and measures the effect on the model's predictions. The four
edits (`sink_common.py`) act after the softmax on query rows 1 to T-1, one
layer at a time and in all layers at once: zero (the first token's weight
set to 0, so each head loses its sink term), mean (that weight replaced by
its mean over rows, so the sink term becomes a constant vector per head),
renorm (the weight set to 0 and the row renormalized, so the head must
spread the sink's share over other tokens), and zero_rand (one random
ordinary position's weight set to 0, for scale). Outcome: round 11's KL.

## Development calibration (labeled)

`sink_round.py --dev` on Qwen2.5-0.5B (`sink_dev/`, `sink_dev_results.json`,
`sink_dev.log`). A first development run used a custom attention function
without causal masking; the built-in equivalence check flagged it (maximum
logit difference 17.5 against eager attention), the run was discarded, the
mask was added, and the rerun matches eager attention exactly (difference
0.0). Results of the rerun: in all layers at once, zero costs 28.7 times
zero_rand (KL 0.225 against 0.008); mean recovers most of it (0.19 of
zero); renorm costs 0.05 of zero in all layers at once although it is the
costliest edit in single layers (up to 0.83). In the 16 high-sink layers
(first-token mass above 0.3), zero exceeds zero_rand in 16, and the median
of mean over zero is 0.14 (interval 0.11 to 0.22).

## Models and plan

Round 11's four models, fixed in `sink_round.py`: every layer of
Qwen2.5-1.5B and TinyLlama, every fourth layer of Mistral-7B and
Qwen2.5-7B, plus all layers at once; 8 evaluation windows of 512 tokens for
the small models and 4 for the 7B models. Each model is checked against an
eager load (identical logits required); the first token's attention mass
per layer (rows 16 and above, window 0) defines high-sink layers.

## Registered predictions

- K1: in all layers at once, zero costs at least 5 times zero_rand, in at
  least three of the four models.
- K2: in all layers at once, mean costs at most half of zero, in at least
  three of the four models.
- K3: zero costs more than zero_rand in at least two thirds of the high-sink
  layers, in at least three of the four models.
- K4: over high-sink layers, the median of mean over zero is at most 0.5,
  with a two-stage cluster bootstrap 95 percent interval whose upper bound is
  below 0.75.
- Reported without thresholds: renorm per layer and in all layers; the
  per-layer profiles; the change in NLL.

## Reading and falsification

K1 and K3 failing means removing the sink's contribution is no worse than
removing an ordinary token's: the sink would be close to a no-op. K2 and K4
failing means the sink term's variation across positions carries most of
its effect: it would act as a gate rather than a constant bias. Each clause
is reported whichever way it falls.

## Artifacts

`sink/<model>.json`, `sink_results.json` from `eval_sink.py`, notes in
RUNLOG.md.
