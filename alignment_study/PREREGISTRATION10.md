# Preregistration 10: new attention designs (multi-query, ALiBi, learned positions)

Date: 2026-09-11. Frozen by commit and pushed to the public repository
before any of the four models is downloaded for these runs; the push
timestamp is the ex-ante evidence. Runs locally on Apple MPS (bfloat16 for
all four models under the precision policy).

## Purpose

The thirteen models measured so far use rotary or learned positions with
multi-head or grouped-query attention, and the sink-profile statement of
preregistration 9 holds in eleven of twelve of them. Four open models add
attention designs not yet tested: `tiiuae/falcon-7b` (multi-query
attention, one key/value head for 71 query heads, rotary), `bigscience/bloom-7b1`
(ALiBi position bias, 32 heads, no rotary), `facebook/opt-6.7b` (learned
absolute positions, plain multi-head attention, 32 heads) and `Qwen/Qwen3-8B`
(the newest Qwen generation, 32 query heads, 8 key/value heads, rotary).
None has been analyzed in this project. OPT-6.7B's last-token attention
correlations are being computed for the CHAI reproduction before this run;
no other statistic of these models has been seen.

## Protocol

Exactly preregistration 6 for the interaction statistics (protocol A: 48
windows at T = 64 with column-set surrogates; protocol B: 6 windows at
T = 256 with the wrapped-target control; `rerun_round.py --only=... --out=alignment_study/rerun_pr10`)
and exactly preregistration 9 for the generative model (`sinkprofile_round.py
--only=... --out=alignment_study/sinkprofile_pr10`). Evaluation by the
committed `eval_rerun.py` and `eval_sinkprofile.py` on those directories.
A model is testable for a clause under the same rules as before (at least
3 high-sink layers where the clause concerns high-sink layers). Falcon's
71 heads make the surrogate statistics slow; the protocol is not reduced
for it.

## Registered predictions

- N1: S1' (shared sink operator) holds in at least 3 of the 4 models.
- N2: S2' (column-set sufficiency) holds in at least 3 of 4.
- N3: S3' (dissociation, wrapped control at T = 256) holds in at least 3
  of 4.
- N4: pooled over the high-sink layers of the four models with cv(gn) >=
  0.1, Spearman(shared energy, r1) <= -0.5.
- N5: the sink-profile rebuild reproduces the per-layer z at Spearman >= 0.7
  with median error <= 3 z-units (P1 and P2 of preregistration 9) in at
  least 3 of 4 models.
- Reported without threshold: which models form a first-token sink at all
  (the number of high-sink layers), since ALiBi and learned positions may
  place or spread the sink differently; a model with fewer than 3
  high-sink layers is recorded as untestable, not failed, and that is
  itself the finding for that design.

## Falsification

Any of N1, N2, N3 or N5 failing in 2 or more testable models restricts the
corresponding statement to the rotary and learned-position designs tested
before, and the paper says so. Every value is reported whichever way it
falls; post-hoc analyses are labeled.

## Artifacts

`rerun_pr10/<model>.json`, `rerun_pr10_results.json`,
`sinkprofile_pr10/<model>.json`, `sinkprofile_pr10_results.json`; run notes
in RUNLOG.md.
