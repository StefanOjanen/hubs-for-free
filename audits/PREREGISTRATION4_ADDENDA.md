# Preregistration 4: addenda

Dated corrections and clarifications to `PREREGISTRATION4.md`, which is not
edited after its freeze (commit `ee4e267`, pushed 2026-09-14 23:24 UTC;
SHA-256 of the file at that commit
`f078f7e43ea166f4f3877084bb424b547b064fdb3107c775b6ca07d2ee702b70`;
permalink
https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md).
Nothing below changes a statistic, a null family, a threshold or an
outcome label. Each addendum was committed before the run it concerns.

## Addendum 1 (2026-09-15): title line

The first line of the frozen file still reads "(DRAFT, not yet frozen)".
The Status line directly below it, written at the freeze, governs. The
title was left as it is so that the file's digest stays the one recorded in
`alignment_study/RUNLOG.md`.

## Addendum 2 (2026-09-15): registration of record

The header says the file is "frozen by commit and pushed to the public
repository, then registered on OSF". The author deferred the OSF
registration on 2026-09-15 (a draft was prepared on OSF and left
unregistered). The registration of record is therefore the public freeze
commit, the same discipline as preregistrations 1 to 3 and 5 to 10 of this
project. The battery scripts' `--registered=` flag takes the freeze
permalink and writes it into each result file under the key `registered`.
No battery ran on real data before this addendum was committed.

## Addendum 3 (2026-09-15): precision of the untrained model in Target 5

The retrieval battery instantiates the config-initialized Qwen2.5-7B in the
same dtype as the trained model under the platform policy
(`hubsfree/adapters.py`: bfloat16 for a 7B on the 32 GB Apple MPS working
set). The committed script had omitted the dtype argument, which would have
requested float32 (about 30 GB of weights) and failed on this machine. The
frozen text fixes no precision for the untrained model, and the base result
of Target 5 was reproduced in bfloat16 as well. Changed before the run.

## Addendum 4 (2026-09-15): readings fixed by the evaluation script

`audits/eval_batteries.py`, committed before any battery result exists,
fixes four readings of the frozen text that the text leaves open. (i)
"Shrinkage" is the share of the real effect a null reproduces, the frozen
file's own definition (null median over real; for Target 3 the excess over
the random-null median), so "shrinks by X percent" and "shrinkage under X
percent" both refer to that share; the raw shares are printed so a reader
who prefers the other reading of Target 3's clause on the marginal-matched
surrogate ("shrink by more than 50 percent against (b)") can apply it. (ii)
A clause the file states per layer (Target 3) holds for a model when it
holds in at least two thirds of the layers, the fraction the file names for
the column-set clause. (iii) A "matched" statistic lies between the 5th and
95th percentile of the null; for the cluster share, which ties, the
mid-rank percentile is used. (iv) With five or three untrained
initializations, a 95th or 99th percentile clause holds only if the real
statistic exceeds every initialization. Committed before the batteries of
Targets 1, 3 and 4 finished and before the Target 3 battery started.

## Addendum 5 (2026-09-15): model flags and reporting fields

The frozen file runs the Target 3 battery on Mistral-7B-v0.1 and
OPT-6.7B and the Target 5 battery on Qwen2.5-7B and
Mistral-7B-Instruct-v0.2; the committed scripts had the first model of each
pair hard-coded. They now take `--model=` and name the result file after the
model. The Target 3 battery additionally records the 5th and 95th
percentiles and the mid-rank percentile of every null statistic, and the
Target 5 battery records the per-initialization values of the untrained
null instead of medians alone. No statistic, null or threshold changed. The
Qwen2.5-7B run of Target 5 started before this addendum under the old
output name and is renamed to the per-model name when it finishes (noted in
RUNLOG.md).

## Addendum 6 (2026-09-15): Target 4 battery in parallel worker processes

At the measured 26 seconds per 4096 x 4096 Gram eigenvalue solve under the
concurrent load, the single-process battery (generator seed 0) would have
needed more than a day for the 64 square matrices. It was stopped after
the six matrices of layers 0 and 1 (their log lines are kept in
`audits/dewage2026/battery.log` and were seen before the stop) and
restarted as six single-threaded workers over interleaved layers: worker i
takes the layers with l mod 6 = i and seeds its generator with 1000 + i;
`--aggregate` combines the per-matrix rows with the same aggregation code
as the single-process form. Statistic, null families, draw counts and
thresholds are unchanged; only the random draws differ from what seed 0
would have produced.
