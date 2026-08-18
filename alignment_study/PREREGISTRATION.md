# Preregistration: shared-mode structure of attention-head generators

Date: 2026-08-18. Registered before the confirmatory runs below were executed.
Exploratory basis (already run, single layer, archived in `exploratory/`):
on Qwen2.5-0.5B layer 21, the rank-1 correlation of the head-coupling matrix
is -0.04 (real) versus 0.985 (marginal-matched surrogates), and preserving
only the sink column in the surrogate collapses it to 0.10 (blind replica
0.13). The confirmatory question: is this a general, provable property of
trained attention, i.e. are a layer's head generators dominated by one shared
component aligned with the sink, with head individuality second-order?

## Common protocol

Data: Salesforce/wikitext, wikitext-103-raw-v1, validation split; texts with
more than 40 whitespace words, first 12; tokenized with truncation at 64.
Models loaded float32, eager attention, output_attentions. Per (model, layer,
window): heads' attention A (n, T, T) in float64. G_h = (A_h - A_h^T)/2;
gn_h = ||G_h||_F; C_ij = ||[G_i, G_j]||_F; r1 = Pearson correlation of
upper-triangle C against outer(gn, gn); gap = ev_max / |ev_2nd| from eigvalsh.

Surrogate families (8 draws each, rng seed 0), per head, per row i, applied
to causal entries only; all preserve row sums, row IPR, and the diagonal
exactly, hence every gn_h exactly (Proposition 2):
- plain: permute columns 0..i-1 (the family in run_qwen_protocol.py).
- sinkfix: permute columns 1..i-1; column 0 and diagonal fixed.
- altsink (control separating per-head sinkiness from cross-head alignment):
  head h gets target column t_h = h mod 4; per row i, among the causal
  off-diagonal entries (columns 0..i-1), swap the row maximum into column
  t_h when t_h <= i-1, then permute the remaining off-diagonal entries;
  rows with t_h > i-1 are plain-permuted. Every head stays individually
  sink-like (one dominant column) but heads no longer share a column.

Shared-mode decomposition, per (layer, window): stack vec(G_h); S = top
principal component (unit Frobenius norm, sign arbitrary); a_h = <G_h, S>;
E_h = G_h - a_h S. Reported: shared-energy fraction a_h^2 / gn_h^2 per head;
rho = lambda_max / n of the cosine Gram M_hk = <G_h, G_k> / (gn_h gn_k);
cosine of S against the ideal causal sink generator (1 e_0^T - e_0 1^T)/2
normalized. rho is invariant to head relabeling and to the sign of S.

## Registered predictions

- A1 (generality of the breakdown). Qwen2.5-0.5B, all 24 layers: define
  D_l = median plain r1 - median real r1. Predict D_l > 0.3 for at least
  16 of 24 layers.
- A2 (sink sufficiency). Among layers with D_l > 0.3, predict
  |median sinkfix r1 - median real r1| < 0.2 in at least 2/3 of them.
- A3 (shared-mode order parameter). Predict median-over-windows rho(real)
  at least 2x rho(plain surrogate) in at least 2/3 of layers, and Spearman
  correlation across layers between median rho and mean column-0 mass
  greater than 0.5.
- A4 (cross-model). GPT-2 small and Pythia-160m: among layers with mean
  column-0 mass > 0.2, predict at least 2/3 satisfy both D_l > 0.3 and
  |sinkfix - real| < 0.2. If a model has fewer than 4 qualifying layers,
  record that prediction as untestable for that model, not passed.
- A5 (second-order theorem). With G_h = a_h S + E_h, [G_i, G_j] =
  a_i [S, E_j] - a_j [S, E_i] + [E_i, E_j]; the a_i a_j term vanishes
  identically, giving ||[G_i, G_j]||_F <= 2(|a_i| e_j + |a_j| e_i + e_i e_j)
  with e_h = ||E_h||_F. Predict the bound verifies numerically to float64
  precision and median(bound / actual) over head pairs at the three
  highest-sink Qwen layers is below 6 (informative, not vacuous).
- A6 (identity of the shared mode). Predict |cosine(S, ideal sink
  generator)| > 0.7 at every Qwen layer with mean column-0 mass > 0.3.
- A7 (alignment, not sinkiness, is load-bearing). At layers with
  D_l > 0.3, predict the altsink family restores rank-1 behavior
  (median altsink r1 > 0.7) and collapses the shared mode
  (median rho(altsink) < 0.5 x median rho(real)) in at least 2/3 of
  those layers. If A7 holds, per-head sink concentration alone cannot
  explain the phenomenon; cross-head column alignment is required.

## Falsification

If A1 or A2 fails, the shared-sink account does not generalize beyond
layer 21 and the headline claim is rejected in its current form. A3-A6
failures narrow the claim but do not rescue it by reinterpretation; any
post-hoc reading will be labeled as such.

## Artifacts

Confirmatory scripts and JSON outputs land in this directory
(`qwen_multilayer.*`, `crossmodel.*`, `gram_theorem.*`, `replica_subset.*`)
and are committed with this file's history intact.
