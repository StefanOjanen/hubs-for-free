# Preregistration 35: does exempting the retrieval heads from key-value head merging restore retrieval from context on models it has not been tuned on?

Date: 2026-11-04. Frozen by the commit that adds this file with its
population script and evaluator, before the confirmatory run; the author
decides whether and when it runs.

## Question

At half the key-value heads, the plain fusion of Mistral-7B-Instruct-v0.2
keeps WikiText-2 perplexity within 2.4 times and retrieves a planted
five-digit code 2 times in 15 (unconverted 15 of 15), in the right form with
the wrong digits (development, 2026-11-01). Keeping unmerged the key-value
heads that serve Wu et al.'s retrieval heads at score 0.3 or more (five of
256, two percent of the cache above half) restores 14 of 15 at perplexity
13.9; five random heads drawn from below score 0.1, with the odd layers'
leftovers held below it, restore 7 at 16.0; the 24 heads whose cheapest
pairing costs most, the fusion's own criterion, restore 5 at 14.0. One model,
fifteen instances per arm, the threshold 0.3 chosen on that model. Two claims
follow and neither has been tested elsewhere: that the exemption restores
retrieval on other models at the same small cost, and that it does so because
the exempt heads are the retrieval heads, not because any small exemption
does.

## Population

Models: Qwen2.5-7B (retrieval scores in audits/retrieval_heads, from the
reproduction of round 4's audit: 33 query heads at 0.1 or more and 12 at 0.3,
serving 17 and 6 of its 112 key-value heads; 4 key-value heads a layer, so a
pair is half the layer and arms (b) and (d) exempt 6 heads, (c) and (e) 17)
and Mistral-7B-v0.1 (scores computed before the freeze by
audits/retrieval_heads/reproduce.py at the audit's Mistral-7B-Instruct
protocol, contexts of 1,024 and 2,048 tokens at five depths, twenty instances
of Wu et al.'s needle, not the codes below, on the hub's default revision at
scoring time; the registered run pins revision 27d67f1b; the result: 29 query
heads at 0.1 or more and 7 at 0.3, serving 21 and 3 of the 256 key-value
heads, the three strongest the same heads as Mistral-7B-Instruct's, layers 12,
18 and 30, so arms (b) and (d) exempt 3 heads and (c) and (e) 21 on this
model); Mistral-7B-Instruct-v0.2 is the development model and is reported, not
scored. Conversion: the paper's fusion at half the key-value heads (round 28's
calibration, 128 C4 validation documents of 2,048 tokens), bfloat16, pairs by
the residual's matching except where exempt. Arms per model: (a) the plain
fusion; (b) exempt the key-value heads serving a query head scored 0.3 or
more; (c) exempt those scored 0.1 or more; (d) the same count as (b) drawn at
random from heads below 0.1, leftovers below 0.1, seeds 0 and 1 (two draws);
(e) the same count as (c), likewise; (f) the same count as (b) chosen by the
fusion's cheapest-pairing cost. The unconverted model is run once. Retrieval:
ten five-digit codes planted at five depths (10, 30, 50, 70, 90 percent) of a
2,048-token C4 haystack after the calibration and check documents, the
question in the Retrieval_Head source's non-chat form, greedy decoding of 24
tokens, a hit when the digits appear: 50 instances per arm. Perplexity on the
first 8 WikiText-2 windows of 2,048 tokens and the cache fraction per arm. If
a scored model's unconverted retrieval is below 45 of 50, the model is
excluded before any converted arm is read (an unconverted model that cannot do
the task cannot test the conversion; 45 is the development model's 15 of 15
scaled with room for tokenization accidents). The population script refuses to
start unless every scored model's retrieval scores exist and match its
configuration, so a model cannot drop out silently.

## Registered predictions

- E1: on both scored models, arm (b) retrieves at least 10 more of 50 than
  the plain fusion (a).
- E2: on both scored models, arm (b) retrieves more than each of its two
  random controls (d).
- E3: on both scored models, arm (b) retrieves more than the residual-chosen
  arm (f).
- E4: on both scored models, arm (b)'s cache fraction is at most 0.55 and its
  perplexity is at most the plain fusion's.
- E5: pooled over the two models, arm (c) retrieves at least as many as arm
  (b) and more than each of its controls (e).
- Reported, not scored: the development model's arms under the same
  protocol (50 instances); per-depth hits; which heads are exempt; run times.

If the plain fusion retrieves 40 or more of 50 on a scored model, that model
cannot show a repair; E1 to E3 are then untestable on it (not failed) and the
round says so, which is itself a finding about where the loss occurs.

## Prior work and cost

Wu et al. (2024) identify retrieval heads and show that pruning them removes
needle retrieval. RazorAttention (Tang et al., 2025), DuoAttention (Xiao et
al., 2025) and HeadKV (Fu et al., 2024) keep full caches for retrieval heads
under token eviction; no conversion or head-merging method we found reports
retrieval. The exemption is the grouping-side counterpart. It is defined on Wu
et al.'s score as audited in this repository; Zhang et al. (arXiv 2506.09944)
show a query-focused score picks heads that matter more per head when masked,
so a different scoring could give a cheaper or better exemption, a follow-up
and not a change to this design. Cost: moments once per model (about 15
minutes), seven converted arms plus the unconverted model, 50 instances each
at about 10 s per instance (the generations in needle_dev took 9 s each; the
longer arm times were the moments and conversions), about 2 GPU hours per
model, 6 for the three. Instruments: retrieval_protect_dev.py in its
registered mode (--registered, the population above, the exclusion rule,
retrieval_round/<model>.json) and eval_retrieval35.py (E1 to E5, checked on a
synthetic population), both committed before this freeze (8edd8eb, 5eddf81);
the development test of the registered mode is the run of the same population
on the development model (--dev, retrieval_round_dev/), done before the freeze
(2026-11-03 to 04, retrieval_round_dev/Mistral-7B-Instruct-v0.2.json: the
unconverted model 50 of 50; the plain fusion 10; (b) 46 at cache 0.520 and
perplexity 13.93 against the fusion's 16.68; (c) 48; the five-head random
draws 7 and 12; the 24-head draws 38 and 6; the residual's five 19; E1 to E5
as written above would all hold on it, and their sizes were not changed after
it) and reported with the round.
