# Preregistration 21: the sink as bias or gate, across architectures, and a sink-free stream

Date: 2026-10-15. Frozen by commit and pushed before the confirmatory runs.

## Question

Round 13 found, in four grouped-query decoders, that removing the first
token's contribution costs 28 to 68 times as much as removing an ordinary
token's, and that replacing its weight by its mean over positions, which
turns its contribution into a constant vector per head, keeps only 13 to 43
percent of that damage: most of what the sink does is a constant bias. Does
that hold across position schemes, attention layouts and encoders? And if
the sink is mostly a bias, can a constant per-head bias stand in for the
sink token when streaming attention evicts it?

## Instrument (`alignment_study/sink_arch.py`)

A custom attention function, checked equal to eager attention in every model,
for multi-head and grouped-query decoders and for bidirectional encoders,
with the sink given as columns: the first token in decoders, both [SEP]
tokens in BERT, the end-of-sentence token in a translation encoder.

Part E, every layer at once, all rows other than the sink rows: zero (the
sink weights set to 0), mean (each replaced by its mean over those rows),
rand (one ordinary position's weight set to 0). Bias share = 1 - KL(mean) /
KL(zero). Outcomes: decoders, next-token KL on 8 WikiText-103 test windows
of 512 tokens; BERT, masked-LM KL at 15 percent masked positions of 200
segments in the source's input format (preregistration 17's builder);
translation encoder, the decoder's next-token KL teacher-forced on the
unchanged model's greedy translation of 60 newstest2013 sentences.

Part W, streaming on 2,048-token WikiText-103 test windows with window W =
256, NLL on positions from W on (4 calibration windows, 8 evaluation
windows): full attention; the window alone; the window plus the first four
tokens (StreamingLLM); the window plus a constant virtual sink, out = w v +
(1 - w) (softmax over the window) V per layer and head, with w the head's
mean first-token weight and v its mean first-token value vector from the
calibration windows, applied to rows whose window no longer holds the first
token. Recovery = (NLL_window - NLL_bias) / (NLL_window - NLL_stream).

## Development (labeled, `alignment_study/sink_arch_dev/`)

Equivalence and restore checks 0 in all five development models. Bias share
and sink-removal damage over ordinary-token removal: Qwen2.5-0.5B 0.85 and
36x (round 13's pattern); gpt2 -0.09 and 19x (removing the sink is costly,
but a constant does not replace it: a gate rather than a bias);
bert-base-uncased [SEP] 0.16 and 0.97x; opus-mt-en-es end-of-sentence 0.60
and 0.51x (in both encoders removing the sink costs no more than an ordinary
token). Streaming on Qwen2.5-0.5B: full 2.70 nats, window 4.99, StreamingLLM
2.91, first token only 2.92, constant virtual sink 3.22, recovery 0.85. The
development suggests a hypothesis this round tests: decoders with rotary
positions have bias-like sinks, decoders with learned absolute positions
gate-like ones.

## Registered predictions

Confirmatory models: gpt2-medium, EleutherAI/pythia-410m, facebook/opt-6.7b,
bert-base-cased, Helsinki-NLP/opus-mt-en-de (Part E); TinyLlama_v1.1,
Qwen2.5-1.5B, Pythia-410m (Part W).
- G1: in the rotary decoder Pythia-410m, the bias share is at least 0.5.
- G2: in the learned-absolute-position decoders GPT-2 medium and OPT-6.7B,
  the bias share is below 0.5.
- G3: in all three decoders, removing the sink costs at least five times as
  much as removing an ordinary token.
- G4: in both encoders, removing the sink costs at most twice as much as
  removing an ordinary token.
- G5: StreamingLLM reproduces: keeping the first four tokens lowers the NLL
  against the window alone in every Part W model.
- G6: the constant virtual sink recovers at least half of StreamingLLM's
  improvement in every Part W model.
- Reported: every KL and NLL, the first-token-only stream, the virtual
  weights, and the checks.

## Falsification and reporting

G1 or G2 failing means the position scheme does not separate bias-like from
gate-like sinks; G3 or G4 failing means the decoder and encoder pictures do
not hold as stated; G6 failing means a constant bias does not stand in for
the sink token in streaming. Every clause is reported whichever way it falls
(`alignment_study/eval_sink_arch.py` writes
`alignment_study/sink_arch_results.json`).
