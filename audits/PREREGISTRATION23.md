# Preregistration 23: does a statistic's weighting decide whether the sink dominates it?

Date: 2026-10-15. Frozen by commit and pushed before the confirmatory runs.

## Question

Round 18's rule predicted four of six new claims. Its development data
suggested a refinement: statistics that weight positions by attention mass
are reproduced by the null that keeps the shared columns, statistics that
count positions are not. That reading was formed on development data. This
round tests it prospectively by flipping the weighting of five audited
statistics (`audits/rule/flip_round.py`) and predicting how the reproduced
share s moves.

| Pair | Mass-weighted | Counted |
|---|---|---|
| heavy (H2O) | share of accumulated attention on the top 10% of positions (round 18) | share of above-average selections on the top 10% most-selected positions (new) |
| persist (Scissorhands) | share of second-half attention mass on the first half's top 10% positions by mass, minus 0.1 (new) | persistence ratio minus pivotal share (round 18) |
| hit (SnapKV) | share of generation attention mass on the voted positions (new) | hit rate of above-threshold features (round 18) |
| chai (CHAI) | Pearson correlation between heads' last rows (Target 3) | Jaccard overlap of heads' top-5% last-row positions (new) |
| clark (Clark et al.) | Jensen-Shannon contrast, different minus same layer (Target 1) | Jaccard contrast of the top-5 positions per row (new) |

Models and inputs as in the source rounds: OPT-6.7B on 8 WikiText-103 windows
of 1,024 tokens (heavy, persist); Mistral-7B-Instruct-v0.2 on 6 C4 prompts
with its own 128-token continuation (hit); Mistral-7B on Target 3's 32 C4
documents (chai); bert-base-uncased on Target 1's 38 windows (clark). Nulls:
random softmax rows and the shared-columns-kept permutation, 10 draws each;
s = (median_colset - median_random) / (real - median_random).

## Development (labeled, `audits/rule/flip_dev/`)

Qwen2.5-0.5B (heavy, persist, hit, chai) and google/bert_uncased_L-4_H-512_A-8
(clark). s for mass and count: heavy 0.83 and 0.32; persist 0.59 and 0.04;
hit 0.39 and 0.02; chai 0.76 and 0.05; clark 1.00 and 0.75. Flipping the
weighting moves s in the predicted direction in all five pairs, by 0.51,
0.55, 0.37 and 0.71 in the first four, but only by 0.25 for clark. There the
counted statistic takes the top 5 of about 64 positions, and the two separator
columns occupy two of those five slots. The development data therefore point
to a sharper reading than mass against count: the shared columns dominate a
statistic in proportion to the share of its units, attention mass or counted
slots, that they occupy. The plain mass-against-count rule would predict s at
most 0.2 for every counted statistic, including clark's.

## Registered predictions

- F1 (four pairs, one clause each): for heavy, persist, hit and chai, s of
  the mass-weighted statistic exceeds s of the counted statistic by at least
  0.3.
- F2: for clark, where the shared columns occupy a large share of the counted
  slots, the two s values differ by less than 0.3.
- Secondary: the plain rule (mass-weighted s at least 0.5, counted s at most
  0.2) for each of the ten statistics; reported, not scored.

Thresholds were set after the development run above; the five new statistics
have not been computed on any confirmatory model.

## Falsification and reporting

F1 failing for a pair means flipping the weighting does not move the sink's
share there; F2 failing means the occupancy reading does not explain the
small-k case. Every clause is reported whichever way it falls
(`audits/rule/eval_flip.py` writes `audits/rule/flip_results.json`).
