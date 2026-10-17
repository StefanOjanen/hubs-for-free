# Preregistration 25: cross-layer KV cache compression under the convention null

Date: 2026-10-17. Frozen by commit and pushed before the confirmatory runs.

## Question

Two published methods compress the key-value cache across layers on the premise
that adjacent layers' cached keys and values are similar. MiniCache (Liu et al.,
NeurIPS 2024) merges the caches of adjacent layers token by token and reports a
cosine similarity between adjacent layers' KV states that rises with depth to
0.8 to 0.9 (LLaMA-3-70B, its Figure 1(a)) and is about 0.5 per token
(LLaMA-2-7B, its Figure 2(b)); the paper does not state how the similarity is
computed, and its code is not released. XQuant (Yang et al., EMNLP 2025)
quantizes the cache to 2 bits and lets a layer reuse the previous layer's codes
with its own scale and zero-point, reporting that over 80 percent of positions
differ by at most one quantization level between adjacent layers, "a previously
overlooked phenomenon". Every attention head's query/key space and value/output
space can be rotated, and a layer's KV heads permuted with their query heads,
without changing the model. A per-token cosine between two layers' caches, a
code-agreement rate, and a merge that pairs head h of one layer with head h of
the next all depend on that arbitrary choice. xKV (Chang et al., ICML 2026)
reported low per-token cosine across layers. This round tests both premises
against nulls, and each method in the trained basis, in random
function-preserving bases and after aligning the heads.

## Instrument

`alignment_study/xlayer_round.py` (evaluated by `alignment_study/eval_xlayer.py`).

Part A (premise), on two held-out WikiText-103 test windows of 1,536 tokens: the
per-token cosine (tokens 1 onward) between adjacent layers of the keys and of
the values (concatenated over KV heads), of the residual stream (layer outputs)
and of the normalized layer input; the key and value cosine again after per-head
alignment fitted on calibration documents (orthogonal Procrustes, rotations
commuting with the rotary embedding for keys), and after the same fit against
token-shuffled targets (the fitting null). Medians over layer pairs.

Part B (MiniCache's merge, as implemented in KVCache-Factory): per head and token,
SLERP (t = 0.6) of the two unit directions, each layer's magnitudes kept, the 5
percent of tokens with the largest angular distance per head kept unmerged;
layer pairs (S, S+1), (S+2, S+3), ... from S = L/2 (MiniCache's setting) and
from S = L/4. The context's cache (1,536 tokens) is merged, then the NLL of the
next 256 tokens is measured; 32 WikiText-103 test windows. Damage = NLL minus
the full cache's (nats per token). Bases for layer l-1 relative to layer l:
trained (the published merge); random0..2 (random function-preserving rotations
and a random KV-head permutation, undone after the merge); aligned (heads
matched by Hungarian assignment on post-alignment similarity and rotated by
orthogonal Procrustes, fitted on 4 C4 validation documents of 1,024 tokens;
equivalent to folding the rotations into W_K, W_Q, W_V, W_O); aligned_nomatch
(the same rotations, heads paired by index).

Part C (XQuant): KIVI 2-bit codes (keys per channel in 32-token groups, values per
token in 32-channel groups, the last 128 context tokens in full precision). The
statistic: the distribution of |code difference| between adjacent layers
(XQuant's Figure 3), against the independence prediction computed from the two
layers' code frequencies, and after alignment. The function: from S = L/4
(rounded up to even), layer l+1 dequantizes layer l's codes with its own scale
and zero-point (XQuant's sharing, its 2-bit form without its calibration);
arms kivi (no sharing), trained, random (layer l quantized in a random
function-preserving basis), aligned (in the basis aligned to layer l+1),
kivi_rot (aligned basis, no sharing); continuation NLL as in Part B.

Models: yaofu/llama-2-7b-80k (a LLaMA-2-7B derivative; LLaMA-2-7B is used by
both papers and is gated), mistralai/Mistral-7B-Instruct-v0.2 (XQuant's model;
MiniCache's Mixtral uses the same attention), Qwen/Qwen2.5-7B, and
microsoft/Phi-3-mini-4k-instruct (MiniCache's). bfloat16 for the 7B models,
float32 for Phi-3-mini, on the M1 Max.

## Development (labeled)

Exploratory probes (scratch scripts, recorded in the run log) on TinyLlama,
Qwen2.5-0.5B, Pythia-410m and the LLaMA-2-7B derivative found adjacent-layer
key and value cosine within 0.07 of zero at every depth of the LLaMA-2-7B
derivative, while its residual stream rose from 0.61 to 0.97 and its normalized
layer input from 0.87 to 0.97 with depth, the shape of MiniCache's Figure 1(a).
A per-head sink-logit correction after round 24's conversion did not reduce its
damage (not pursued).

Development runs of the instrument (`alignment_study/xlayer_dev/`, CPU, float32)
on TinyLlama, Qwen2.5-0.5B and Pythia-410m:
- Part A: median key cosine +0.022, +0.001, +0.018; value +0.001, -0.002,
  -0.002; residual stream 0.875, 0.918, 0.905; aligned value cosine above the
  fitting null by 0.23, 0.28, 0.24.
- Part B, continuation perplexity (full; trained; random median; aligned;
  aligned_nomatch), from L/2: TinyLlama 6.8; 20.8; 19.3; 8.2; 9.1. Qwen2.5-0.5B
  11.7; 53.6; 41.7; 22.3; 25.0. Pythia-410m 14.3; 145.3; 118.2; 43.0; 53.0.
  From L/4: TinyLlama 6.8; 108.1; 169.1; 20.3; 63.8. Qwen 11.7; 238.0; 148.2;
  49.1; 74.5. Pythia 14.3; 1,111.6; 752.8; 49.1; 116.5. From L/2 the trained
  basis does worse than the random median in all three (by 0.08, 0.25 and 0.21
  nats); from L/4 it does better in TinyLlama. Aligned damage is 0.17 to 0.48
  of the trained basis's in all six cells.
- Part C: the share of positions with |code difference| at most 1 is within
  0.006 of the independence prediction for keys and values in all three
  (0.79 to 0.81 for keys, 0.82 for values); after alignment the exact value-code
  agreement exceeds independence by 0.053 to 0.072. Sharing (full; kivi;
  trained; random; aligned): TinyLlama 6.8; 7.3; 212.8; 310.1; 242.4. Qwen
  11.7; 13.3; 356.2; 1,531.4; 282.1. Pythia 14.3; 34.0; 594.2; 654.4; 179.4.
  The trained basis does better than random bases in all three; alignment helps
  in two of three.

## Registered predictions

Four models; two merge settings.
- A1: in every model the median key cosine and the median value cosine between
  adjacent layers are within 0.1 of zero, and the median residual-stream cosine
  exceeds 0.5.
- A2: in every model the median aligned value cosine exceeds the fitting null's
  by at least 0.1.
- B1: MiniCache's merge from L/2 is not helped by the trained basis: its damage
  is at least the median damage over the three random bases minus 0.1 nats, in
  every model.
- B2: aligned merging's damage is at most 0.6 of the trained basis's in at least
  7 of the 8 (model, setting) cells.
- C1: XQuant's statistic (the share of positions with |code difference| at most
  1, median over layer pairs) is within 0.02 of the independence prediction for
  keys and for values in every model.
- C2: after alignment the exact value-code agreement exceeds the independence
  prediction by at least 0.03 in every model.
- C3: XQuant's sharing loses less in the trained basis than in random bases in
  at least 3 of the 4 models.
- Reported, not scored: aligned against trained sharing in Part C, the
  aligned_nomatch arm, kivi_rot, per-layer statistics, and the L/4 comparison
  of trained against random bases.

## Falsification and reporting

A1 failing means adjacent layers' caches are similar in the trained basis after
all; A2 failing means alignment reveals no shared structure beyond fitting; B1
failing means MiniCache's merge does exploit the trained basis; B2 failing means
aligning heads does not make cross-layer merging substantially cheaper; C1
failing means XQuant's statistic reflects more than its code frequencies; C2
failing means aligned codes agree no more than chance; C3 failing means
XQuant's sharing is indifferent to the basis. Every clause is reported whichever
way it falls (`alignment_study/eval_xlayer.py` writes
`alignment_study/xlayer_results.json`).
