# Preregistration 16: Voita et al.'s confidence as they define it

Date: 2026-10-13. Frozen by commit and pushed before the confirmatory run.

## Why this round

Preregistrations 12, 14 and 15 registered "the attention confidence of Voita
et al. (2019), the mean largest attention weight" and computed it over all
tokens. Voita et al. define it as the average maximum attention weight
excluding the end-of-sentence token, which they leave out because it is not a
real token (their footnote 2). In a translation encoder that token is where a
head puts attention it does not use, the role the first token plays in a
decoder language model. The registered results of rounds 12, 14 and 15 stand
for the statistic as computed; attributing that statistic to Voita et al. was
wrong. This round tests their definition, in decoder language models and in
the setting they studied.

## Instruments (`voita_definition.py`, `eval_voita.py`)

Decoders: the seven models whose single-head removal damage rounds 12 and 15
stored (Qwen2.5-1.5B, TinyLlama, Mistral-7B and Qwen2.5-7B from round 12, on
WikiText-103 and the layers round 12 ablated; gpt2-medium, Pythia-160m and
Pythia-410m from round 15, every layer, on WikiText-103 and C4). Head
statistics on the same selection windows, model and precision, with the same
sink-column rule: confidence over all tokens (checked against the stored
value), confidence with the sink columns excluded as keys (Voita et al.'s
definition with the sink in place of EOS), with only the first token
excluded, and renormalized by the non-sink mass. Round 15's outcomes were
still being computed at this freeze; no sink-excluded statistic has been
computed for any confirmatory model.

Translation: encoder self-attention heads of Helsinki-NLP/opus-mt-en-ru,
opus-mt-en-de and opus-mt-en-fr, Marian Transformer-base models (6 layers, 8
heads) for Voita et al.'s three language pairs. They are stand-ins: Voita et
al. trained their own models. Statistics on the first 200 English sentences
of newstest2013 with 8 to 40 words, outcomes on the next 100: the head's
slice of the encoder layer's output projection is zeroed and the decoder's
next-token distribution, teacher-forced on the unchanged model's greedy
translation, is compared with the unchanged one (KL over the top 256 tokens
plus remainder). Confidence is averaged over source tokens, EOS excluded as
key and as query.

## Development check (labeled)

`voita_definition.py --dev` on gpt2 and Qwen2.5-0.5B (the development models
of rounds 15 and 12, whose removal outcomes were already stored) and
opus-mt-en-es (`voita_dev/`, `voita_dev.log`, `voita_dev_results.json`).
Recomputed all-token confidence and sink mass equal the stored values
exactly. Within-layer Spearman with removal damage, decoders (36 layers):
sink-excluded confidence +0.28 (interval -0.15 to 0.52; Qwen2.5-0.5B +0.46,
gpt2 -0.08), all-token confidence -0.18, output norm +0.72, partial
correlation of the sink-excluded confidence given sink mass +0.06.
Translation (6 layers): EOS-excluded confidence +0.62, all-token +0.54,
output norm +0.54, negative entropy +0.62, EOS mass -0.12. The thresholds
below were written into `eval_voita.py` before the development run and not
changed; V6 was added after it.

## Registered predictions

Model-clustered two-stage bootstrap intervals (models, then layers; 2,000
draws). Decoder clauses use the WikiText-103 outcomes of all seven models.
- V1: the sink-excluded confidence's mean within-layer Spearman with removal
  damage is at least 0.2, interval above zero.
- V2: it exceeds the all-token confidence's by at least 0.3, interval above
  zero.
- V3: its partial Spearman with removal damage given sink mass has an
  interval above zero.
- V4: in the translation encoders, the EOS-excluded confidence's mean
  within-layer Spearman with removal damage is at least 0.2, interval above
  zero.
- V5: in the translation encoders, it exceeds the all-token confidence's by
  at least 0.1, interval above zero.
- V6 (added after development): in the decoders, output norm's within-layer
  Spearman exceeds the sink-excluded confidence's by at least 0.2, interval
  above zero.
- Reported without thresholds: the C4 outcomes of round 15's models; the
  first-token and renormalized variants; output norm, negative entropy and
  EOS mass in the translation encoders; per-model and per-layer values.

## Falsification and reporting

V1 failing means Voita et al.'s definition does not reliably mark the heads
whose removal matters in decoder language models. V4 failing means it does
not in the setting they studied, with removal damage as the endpoint (they
used layer-wise relevance propagation). V2 and V5 say whether excluding the
no-op token is what matters; V3 whether the statistic carries anything
beyond how much of a head's attention goes to the sink; V6 whether output
norm stays the better predictor. Every clause and per-model value is
reported whichever way it falls, and the text of rounds 12, 14 and 15 is
corrected to describe the statistic they computed.
