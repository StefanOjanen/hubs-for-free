# Scale round run log

Session 1 (2026-09-01, free Colab T4): scale_round.py ran as registered.
Qwen/Qwen2.5-3B completed all 12 windows; its per-layer block was captured
from the cell output and stored as scale_partial/qwen2.5-3b_capture.json.
The runtime was reclaimed by Colab about 65 minutes in, during window 4 of
microsoft/Phi-3-mini-4k-instruct; results for the remaining four models
were lost with the VM (scale_round.json was never retrievable).

Deviation for session 2: scale_round_v2.py reimplements the identical
registered protocol with vectorized surrogate generation (numpy
Generator.permuted over draws x heads per row) and GPU commutator norms
in float32. Surrogate distributions are unchanged; the concrete random
draws differ from v1's stream because the RNG call order differs; the
commutator norms move from float64 to float32, far below draw noise.
Validation against the development model (layers 2, 11, 20, stride-40
windows): real, plain, sinkfix, altsink-v2, and zmed statistics match
tier1_robust.json within window-subset variation, e.g. layer 20 alt2
0.971 (v2) vs 0.974 (tier1), layer 11 alt2 0.985 vs 0.986. A direct
same-window comparison of the two altsink implementations agreed to
three decimals. Model order for session 2 puts the 7B models first so
the highest-value results bank earliest if the session is reclaimed;
per-model results print as MODEL_RESULT blocks for progressive capture.
Evaluation of S1-S4 merges the captured session-1 Qwen2.5-3B block with
the session-2 models.

Session 3 (same day): scale_round_v2.py with the two remaining models
completed both (Phi-3-mini, OLMo-2-7B) in about 18 minutes. All five
registered models are therefore complete across three sessions. Per-model
blocks were captured from cell output into scale_partial/ (the script
rounds aggregates to 4 decimals before printing, so the captured blocks
are the complete output, not truncations). Session 2 was reclaimed during
OLMo's 29 GB download; no computed results were lost across the three
sessions. Evaluation: eval_scale_round.py -> scale_round_results.json.

Local platform (2026-09-10). From this date runs execute on the author's
Apple M1 Max (32 GB unified memory, MPS working set 21.3 GB) through torch
MPS, with the device and precision policy in hubsfree/adapters.py: float32
whenever 4 x parameters <= 0.75 x working set (about 4B parameters on this
machine), bfloat16 above; float16 is excluded because Qwen2.5 attention
overflows to NaN in float16 on MPS. Fidelity (platform_fidelity.py ->
platform_fidelity.json; Qwen2.5-1.5B, 12 windows, T = 64, 28 layers, 20
high-sink): MPS float32 reproduces the CPU float32 layer medians of sink
mass, r1, shared energy and cos(S, sink) to four decimals with 0 of 336
sink-column changes, at 68 ms per window against 382 ms on CPU. MPS
bfloat16 deviates by at most 0.055 (r1), 0.019 (shared energy) and 0.020
(cos) over all layers, 0.048, 0.007 and 0.003 over high-sink layers, with
6 of 336 sink-column changes, at 89 ms per window. The registered anchor
check (scale_round_v2.py --validate, Qwen2.5-0.5B, layers 2/11/20, six
windows) agrees between CPU and MPS float32 on every printed statistic to
four decimals except zmed at layer 20 (10.9165 against 10.9166).
Consequence for the 3B to 7B set: Qwen2.5-3B and Phi-3-mini run in float32
locally; Mistral-7B, Qwen2.5-7B and OLMo-2-7B run in bfloat16 as on the T4.

Rerun round (PREREGISTRATION6.md, frozen and pushed as 3d5b337 at
2026-09-10 20:13 UTC). Launched 2026-09-10 20:15 UTC on the M1 Max with
`--exclude=allenai/OLMo-2-1124-7B --purge-large`. OLMo-2 needs a 27 GB
download that does not fit the 23 GB free while the author's photo archive
uploads to Drive; it runs later under the same frozen file and its date is
recorded here when it does. `--purge-large` removes the Hugging Face cache
of models at or above 3B parameters after their run so the two 7B models
fit the disk in sequence. Downloads run at about 1 MB/s behind the upload,
so the round is expected to take many hours; results appear per model in
`rerun/` and the evaluation is `eval_rerun.py`.
The twelve queued models completed at 2026-09-10 22:33 UTC (2 h 18 min;
the download rate recovered once the photo archive had left the disk).
Phi-3-mini ran in float32 under the precision policy (3.8B parameters);
Mistral-7B and Qwen2.5-7B in bfloat16. OLMo-2-7B launched at 22:35 UTC
under the same frozen file (`rerun_round.py --purge-large`, log
`rerun_round_olmo.log`), 129 GB free.
OLMo-2-7B completed at 23:25 UTC (52 min, bfloat16, 19 high-sink layers);
its cache was purged. All thirteen models are in `rerun/`; scorecard in
`rerun_results.json` and NOTE.md.

Head-merging round (PREREGISTRATION7.md, frozen and pushed as 6b8f8ea at
2026-09-11 08:19 UTC). Launched 08:21 UTC on the M1 Max with
`merge_round.py --purge-large` (log `merge_round.log`; per-model results in
`merge/`). Qwen2.5-1.5B from cache; 3B and 7B re-downloaded since the
rerun purged them.

Derivation round (PREREGISTRATION8.md, frozen and pushed as b9b8ef0 at
2026-09-11 08:24 UTC). Launched 08:26 UTC on the CPU (the GPU is running
the merge round) for the seven models cached locally: gpt2, gpt2-medium,
Pythia-160m, Pythia-410m, TinyLlama, Qwen2.5-1.5B base and Instruct. The
five larger models run under the same frozen file once their weights are
downloaded again (the merge round purges them after use); dates below.

Audit reproductions (2026-09-11). Target 4 (Dewage 2026) reproduced on the
Mistral-7B-v0.1 weights on the CPU while the GPU ran the merge round
(`audits/dewage2026/reproduce.log`, 21 minutes, float32 SVD of 128
matrices). CHAI (Target 3) and retrieval heads (Target 5) reproductions
queued on the GPU behind the merge round (`alignment_study/mps_pipeline.log`
records their start and end times).
Merge round completed 13:25 UTC: Qwen2.5-1.5B 77 minutes including its
pair sweep, Qwen2.5-3B 74 minutes, Qwen2.5-7B 154 minutes (bfloat16). The
GPU pipeline (derivation for the five remaining models, CHAI reproduction
on Mistral-7B, retrieval-heads reproduction on Qwen2.5-7B) started at
13:25 UTC.
Derivation round completed 13:29 UTC (GPU part: Qwen2.5-3B and Phi-3-mini
in float32, Mistral-7B, Qwen2.5-7B and OLMo-2-7B in bfloat16). CHAI
reproduction on Mistral-7B started 13:29 UTC.
Audit reproductions on the GPU (2026-09-11): CHAI at 64 documents of 2,048
tokens ran at about a minute per document and was stopped after five;
rerun at 32 documents of 1,024 tokens, 13:42 to 13:45 UTC. Retrieval heads
at 1K, 2K and 4K contexts with nine depths ran at over a minute per 2K
instance and was stopped; rerun at 1K and 2K with five depths (20
instances), 13:45 to 14:02 UTC. Both reductions relative to the sources
are recorded in PREREGISTRATION4.md. All work of the day ran on the
M1 Max; no cloud compute was used.

Sink-profile round (PREREGISTRATION9.md, frozen and pushed as 6e6b253 at
2026-09-11 14:32 UTC). Launched 14:32 UTC on the M1 Max
(`sinkprofile_round.py`, log `sinkprofile_round.log`, results per model in
`sinkprofile/`); all twelve models' weights were cached from the earlier
rounds of the day.
Sink-profile round completed 15:33 UTC (61 minutes for twelve models; the
32-head models took 8 to 12 minutes each, the statistics being CPU-bound
numpy over 21 coupling computations per layer and window).

Audit reproductions on the sources' own models (2026-09-11, from 16:40 UTC):
after the Llama-2 gate turned out to be avoidable, Target 5 reruns on
`yaofu/llama-2-7b-80k` (the retrieval-heads paper's primary model) and
`mistralai/Mistral-7B-Instruct-v0.2` (another of its models), and Target 3
reruns on `facebook/opt-6.7b` (the CHAI paper's OPT family); same reduced
protocols as the earlier reproductions. Caches of OLMo-2-7B, Phi-3-mini and
Qwen2.5-3B were removed to make room (all their runs are complete and
committed).

New-architecture round (PREREGISTRATION10.md, frozen and pushed as de18d52
at 16:40 UTC before any of Falcon-7B, BLOOM-7b1 or Qwen3-8B was downloaded;
OPT-6.7B was downloading for the CHAI reproduction at that time). Runs
follow the audit reproductions on the GPU; per-model results in
`rerun_pr10/` and `sinkprofile_pr10/`.

Llama-2-7B-80K (2026-09-11, 17:55 to 19:50 UTC). Two failed attempts before
the recorded run: (1) with the checkpoint's saved rope configuration
("dynamic", factor 10) the model generated degenerate text beyond a few
hundred tokens under every precision and attention implementation tried;
loading with linear position interpolation, factor 10, restored fluent
generation at 900 and 2,000 tokens (`--rope=linear:10`, deviation recorded
in the script header). (2) The generate-with-attentions path exhausted the
GPU's memory at 2,048 tokens on this 32-KV-head model (36 GB requested);
the reproduction and battery scripts now decode step by step and keep only
each head's argmax position (reproduction) or the rows of needle-token steps
(battery), which reproduces the earlier Qwen2.5-0.5B numbers exactly. The
orchestrator was paused (SIGSTOP) during this run so the PR10 rounds would
not share the GPU, and resumed after it.
Preregistration 10 rounds completed 22:29 UTC on 2026-09-11 (rerun protocol
19:38 to 21:04, sink-profile protocol 21:04 to 22:29; Falcon's 71 heads set
the pace at 51 minutes for the sink-profile cell). The orchestrator's log is
`alignment_study/audit_pipeline2.log` and `pr10_pipeline.log`.

Preregistration 4 freeze (2026-09-15). Amendments 1 to 5 (draw counts for
the weight-level nulls, last-row surrogates for the last-token statistic,
three untrained initializations for Target 5, the initializer-based
untrained null for Target 4, and the inclusion rule for Llama-2-7B-80K) were
written into the file and the battery scripts before the freeze; the
Gram-eigenvalue recipe was checked equal in outlier count and energy share
to the SVD recipe on three random matrices and four Mistral-7B projection
matrices. Incident, recorded for completeness: while verifying the scripts,
a command intended to print the retrieval battery's help text started the
script without --dry-run at about 23:20 UTC on 2026-09-14; it was killed about four
minutes later while loading the model and the first instances, before
writing any output, and no result was seen. Every battery script now
refuses to run on real data without an explicit --registered=<OSF URL>
flag. The file was renamed from PREREGISTRATION4_DRAFT.md to
PREREGISTRATION4.md at the freeze; the freeze commit hash and the OSF
registration URL are recorded below when they exist.

Freeze commit: ee4e267, pushed 2026-09-14 23:24 UTC (the freeze date in the
file, 2026-09-15, is the author's local date, UTC+2). SHA-256 of
audits/PREREGISTRATION4.md at that commit:
f078f7e43ea166f4f3877084bb424b547b064fdb3107c775b6ca07d2ee702b70. OSF
registration URL: pending (added below when the registration exists).

Registration decision (2026-09-15). The author deferred the OSF
registration ("not needed yet"); an OSF draft with the same summary and a
copy of the frozen file was prepared and left unregistered. The registration
of record for preregistration 4 is the public freeze commit ee4e267
(permalink
https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md),
the discipline of preregistrations 1 to 3 and 5 to 10. Before any battery
started: the four battery scripts' guard now takes this permalink as
--registered=<URL> and writes it into the result file; the retrieval
battery's untrained Qwen2.5-7B is instantiated in the trained model's dtype
(bfloat16) instead of the script's implicit float32, which does not fit the
32 GB working set (PREREGISTRATION4_ADDENDA.md, addenda 1 to 3); all four
dry runs passed after the edits. Launch times, order and outcomes are
logged below.

Battery launch (2026-09-15 00:03 UTC, head 3289070). Clark (bert-base,
CPU float32) and Dewage (numpy, 6 Accelerate threads) run concurrently with
the MPS queue: retrieval on Qwen2.5-7B (bfloat16), then CHAI on
Mistral-7B-v0.1 (bfloat16); two 7B models do not fit the 32 GB working set
together. Logs: audits/battery_pipeline.log and audits/*/battery.log.
Queued after the pipeline: CHAI on OPT-6.7B and retrieval on
Mistral-7B-Instruct-v0.2 (the frozen file names both models for Targets 3
and 5; the scripts gained --model= before those runs, addendum 5). The
evaluation script audits/eval_batteries.py and addenda 4 and 5 were
committed while the first batteries ran and before any result file
existed. Deviation: the frozen inclusion rule for Llama-2-7B-80K (Target 5)
says its outcome is recorded here before any battery runs; the batteries
above started before that test ran. The test's thresholds are fixed in the
frozen file (needle in at least 18 of 20 instances, 2 to 8 percent of heads
above 0.1, with the source repository's own haystack and prompt) and its
outcome does not depend on the other batteries; it runs when the MPS queue
is free and its result is recorded here before a Llama battery, if any,
runs.

Sleep incident (2026-09-15). The machine was on battery and entered Low
Power Sleep at 00:06 UTC, three minutes after the launch, and woke at 09:25
UTC when the lid was opened; the batteries paused with it, so the runtimes
recorded in the result files include about nine hours and twenty minutes
of sleep. From 09:34 UTC a caffeinate assertion holds the machine awake
while the pipeline shell exists and a watcher pauses the batteries
(SIGSTOP) below 25 percent battery charge and resumes them on AC power.
Power pause (2026-09-15): the watcher paused the batteries at 13:04 UTC at
24 percent charge and resumed them at 14:42 UTC when the charger was
connected; the retrieval battery on Qwen2.5-7B finished at 14:44 UTC
(runtime_s in its result file includes the sleep and the pause).

Dewage restart (2026-09-15, about 14:55 UTC). The single-process Target 4
battery had finished 6 of 128 matrices in about two hours of awake time
(26 s per 4096 x 4096 eigenvalue solve under load, measured); it was
stopped and relaunched as six single-threaded workers with aggregation
(addendum 6). Worker logs: audits/dewage2026/battery_part*of6.log;
aggregation log: battery_aggregate.log.

Llama-2-7B-80K inclusion test (2026-09-15): audits/retrieval_heads/inclusion_test.py
reimplements the source repository's protocol (nightdessert/Retrieval_Head at
commit 3ac171a: its haystack parts and needles, sentence-boundary insertion
with the 200-token buffer, the non-chat prompt, 50-step decoding stopping at
a newline, per-step top-1 scoring, ROUGE-1 recall above 50 as the success
test) and applies the frozen rule; committed before its run and queued on the
MPS after the CHAI battery, ahead of the second-model batteries. Dry run on
Qwen2.5-0.5B retrieved 2 of 2 instances.

Clustered inference (2026-09-15). alignment_study/clustered_stats.py
recomputes every pooled correlation of the manuscript from the per-model
result files with a two-stage bootstrap (models resampled, then layers
within a model; checkpoints for the dynamics round; 2,000 draws, seed 0)
and a random-effects average; clustered_stats.json keeps the per-layer
intervals beside the clustered ones. The intervals quoted in manuscript.md,
README.md and NOTE.md were replaced by the clustered ones (S4' -0.78 to
-0.57 instead of -0.76 to -0.62; G1 R^2 0.24 to 0.93 instead of 0.42 to
0.69; M2 -0.10 to +0.35 instead of -0.10 to +0.32). Every law statement
keeps a negative upper bound; the PR10 law on four models widens to -0.92
to -0.37.

Clark restart (2026-09-15, 15:05 UTC). The Target 1 battery was killed by
the system (exit 137, memory) at 14:52 UTC while generating the surrogate
draws; fixed to draw-by-draw generation (addendum 7) and relaunched from
the beginning; the killed run's log is kept as battery_killed.log.

Batteries completed so far (2026-09-15). Retrieval on Qwen2.5-7B (finished
14:44 UTC): 4.2 percent of heads above 0.1, every null at the 100th
percentile, top-10 shrinkage 0.011 or less; E3 holds on this model. CHAI on
Mistral-7B-v0.1 (finished 16:42 UTC, 7,080 s): real cross-head correlation
0.60 to 0.99 across layers; the sink-set surrogate (c) reproduces 0.90 to
0.98 of the correlation excess over the random null but the real value sits
above every one of its 200 draws in all 32 layers, so the registered
"matched" clause (percentile between 5 and 95 in two thirds of the layers)
fails 0 of 32; the marginal-matched surrogate (b) reproduces about zero, so
the clause "reproduces more than half" fails 0 of 32 under the reading of
addendum 4; (a) and (d) survived in 32 of 32. Per-layer label: shrinks in
32 of 32. The registered expectation for this model fails as written; the
numbers are in audits/RESULTS.md. Pipeline 2 started 16:43 UTC with the
Llama-2-7B-80K inclusion test on the MPS; Clark (relaunched 15:05) and the
six Dewage workers continue on the CPU.

Llama-2-7B-80K inclusion test (2026-09-15, 16:43 to 17:01 UTC, 1,051 s,
bfloat16, linear rope factor 10). With the source repository's haystack
parts, needles, sentence-boundary insertion, non-chat prompt and ROUGE
success test, the needle was retrieved in 20 of 20 instances (recall 90 to
100), so the earlier failure under our WikiText haystack and prompt was a
haystack and prompt effect. The fraction of heads above 0.1 is 1.27 percent
(13 of 1,024; strongest L11H15 0.63, L16H19 0.60), below the 2 percent floor
of the frozen rule, so the model stays excluded from the battery; result
in audits/retrieval_heads/llama-2-7b-80k_inclusion_test.json. The model
cache was removed afterwards at the author's request. The second-model
batteries (retrieval on Mistral-7B-Instruct-v0.2, CHAI on OPT-6.7B) exited
at startup at 17:01 UTC on an argument-parsing clash (addendum 8) and are
relaunched after the fix.

Batteries completed (2026-09-15, continued). Retrieval on
Mistral-7B-Instruct-v0.2 (pipeline 3, finished 18:41 UTC, 5,813 s): 3.4
percent of heads above 0.1, mean top-10 score 0.42; every null at the 100th
percentile, top-10 shrinkage 0.007 to 0.011, untrained initializations
zero; E3 holds on both control models, so no verdict is withheld. The power
watcher paused pipeline 3 from 19:12 to 19:55 UTC (battery at 25 percent,
resumed on AC power). Clark (relaunched run, finished 21:06 UTC, 21,745 s
under CPU contention with the six Dewage workers): D 0.1265, nearest
neighbor in own layer 39.6 percent; random maps and the five untrained
models give D of order 1e-5 (percentile 100, shrinkage 0); the
marginal-matched surrogate reproduces 0.095 of D (percentile 100), the
separator-column surrogate 0.602 (percentile 100; nearest-neighbor fraction
34.7 percent against the real 39.6). Label: shrinks. The registered
expectation (shrinkage between 0.2 and 0.7 against both surrogates) fails
on the marginal-matched clause: the layer clustering is not a marginals
effect, it is 60 percent the shared separator columns and 40 percent
something the separator columns do not carry.

CHAI on OPT-6.7B (2026-09-15, finished 21:14 UTC, 9,143 s). Real cross-head
correlation 0.34 to 0.83 (against 0.55 to 0.99 on Mistral-7B); the sink-set
surrogate reproduces a median 0.68 of the correlation excess (0.03 to 0.92,
rising with depth as the sink forms) and the marginal-matched surrogate a
median 0.011; the real value lies above all 200 draws of every family in
all 32 layers, so the registered "matched" clause fails 0 of 32 as on
Mistral. Majority label: shrinks. With both models in, E1 fails as
written and E2 fails on the marginal clause of Target 1; E3 holds on both
control models; E5 waits for the Target 4 battery, whose six workers
finished their real matrices on 2026-09-16 and are drawing the initializer
null (parts 3, 4 and 5 exited 11:11 to 11:12 UTC).

Target 4 battery (2026-09-16, six workers, last exit 12:04 UTC, aggregate
of 128 matrices). All four projection types: the real outlier count lies
above all 20 Gaussian norm-matched draws (percentile 100), so the first
clause holds everywhere; but the Gaussian null itself produces a median
1288 of 4096 outliers for the square types and 138 of 1024 for K and V, so
it reproduces 85 to 89 percent of the reported count for Q and O, 65
percent for V and 40 percent for K. The within-matrix permutation matches
the Gaussian null to three digits (the entry distribution is not the
mechanism), so the clause "at least half the count is matched by (c')"
holds for Q, V and O and fails for K. Row-norm-matched weights reproduce 71
to 88 percent of the energy share, so the clause written as "the energy
share shrinks by less than 50 percent against (b')" fails in all four
types. Types passing all three clauses: 0 of 4; E5 fails as written.
Label: shrinks in all four types. The whole battery of preregistration 4 is
now complete: E1, E2 and E5 fail as written, E3 holds on both control
models, E4 was dropped before the freeze, and the falsification clause is
not triggered.

Post hoc for Target 4 (2026-09-16, labeled, not registered):
audits/dewage2026/posthoc_edge.py compares the recipe's edge with two
oracle-free calibrations (scale matched to the Marchenko-Pastur median and
to its mean) on four layers. Both calibrations put a Gaussian matrix at
exactly zero outliers, against the recipe's 1289 of 4096; the real counts
fall from 1583 to 828 and 182 (Q), 354 to 276 and 105 (K), 240 to 121 and
67 (V), 1463 to 513 and 202 (O). Author letters for the four targets are
drafted in private/letters/ (not committed; the repository holds no
correspondence).

Functional round (PREREGISTRATION11.md, 2026-10-12). Development
calibration on Qwen2.5-0.5B (labeled; functional_dev.py, 497 s,
functional_dev_calibration.json and .log), then the dry run of the
confirmatory script on the development model into a scratch directory to
test the evaluation code (not kept). The preregistration, the instruments
(functional_common.py), the runner with its fixed merge plan
(functional_round.py) and the evaluation (eval_functional.py) are frozen
in this commit, before any confirmatory model is run. Development changed
the hypothesis, as recorded in the preregistration: removing the sink
columns did not improve the map comparison; the positive predictions
concern the first-order output perturbation.

Functional round results (2026-10-12, 19:00 to 21:37 UTC, head e45b49f).
Qwen2.5-1.5B (966 s, float32, 8 layers x 30 pairs, 4,096 evaluation
tokens), TinyLlama_v1.1 (1,055 s, float32, 8 x 48), Mistral-7B-v0.1
(4,498 s, bfloat16, 6 x 48, 2,048 tokens), Qwen2.5-7B (2,852 s, bfloat16,
5 x 48, 2,048 tokens): 1,152 merges in 27 layers, weights restored exactly
after every merge (restore-check KL 0.0 in all four). The keep-awake held
and no pause was needed (charger connected at 45 percent). Evaluation
(eval_functional.py, unchanged since the freeze): gate passes (KL ranking
reliability 0.88; Mistral-7B alone 0.69, with four evaluation windows);
F1, F2, F3a, F3c and F3d pass; F3b fails (map distance minus generator
cosine 0.07, interval -0.03 to 0.18, against 0.10 with an interval above
zero); F4 fails (the output-perturbation pick and the generator-cosine
pick coincide in 10 of 27 layers, the former is cheaper in 11 and costlier
in 6; median cost ratio 1.0). Results in functional/, functional_results.json
and eval_functional.log.

Standard-statistics round (PREREGISTRATION12.md, 2026-10-13). Development
calibration on Qwen2.5-0.5B (labeled; standard_round.py --dev, 440 s,
standard_dev/ and standard_dev_results.json). The preregistration, the
instruments (standard_common.py, reusing round 11's frozen code), the
runner with its fixed plan (standard_round.py) and the evaluation
(eval_standard.py) are frozen in this commit before any confirmatory model
runs. In parallel on the CPU: the full rerun of the Target 1 battery after
addendum 9.

Corrections from the verification of 2026-10-12 (private report), applied
2026-10-13. Public text that overstated or misreported audit numbers is
corrected in manuscript.md, README.md, NOTE.md and audits/TARGETS.md: the
sink-set null on Mistral-7B reproduces 77 to 99.8 percent of the CHAI
statistic (the earlier "90 to 99.8" and, in this log on 2026-09-15, "0.90
to 0.98" were wrong); on OPT-6.7B a median 68 percent; the calibrated
Marchenko-Pastur counts fall by a factor of 1.3 to 9 (not 2 to 9); the
spectral paper's code is public, defines its percentages as energy
ratios and claims a dominant, not exclusive, component; CHAI's code
exists, unlinked from the paper, and its statistic is our reconstruction;
the Retrieval Heads paper names no primary model; the expectations
synthesis no longer says all three failures share one cause. The Target 1
number follows the rerun after addendum 9. Earlier entries of this log are
left as written.

What-the-sink-does round (PREREGISTRATION13.md, 2026-10-13). The first
development run of sink_round.py used a custom attention function without
causal masking (a registered custom attention receives no mask in
transformers 5); its equivalence check against eager attention flagged a
maximum logit difference of 17.5, the run was discarded, causal masking
was added inside the function, and the rerun matches eager attention
exactly. The preregistration, sink_common.py, sink_round.py, eval_sink.py
and the development outputs are frozen here, before the confirmatory run,
which is queued after the standard-statistics round. Also committed before
its run: inclusion_test.py --span=answer, the answer-span scoring of the
source code (addendum 10), to be run on Llama-2-7B-80K after the sink round.

Standard-statistics round results (2026-10-13, 22:34 to 00:20 UTC, head
fc6ae80). Qwen2.5-1.5B 993 s, TinyLlama 1,204 s, Mistral-7B 2,198 s,
Qwen2.5-7B 1,919 s; weights restored exactly in all four. The battery fell
to 23 percent at 23:03 UTC; the runner paused the process (state T) and
resumed it on the charger at 23:08 UTC; the runner's exit code 145 reflects
that stop, the program finished normally. Evaluation unchanged since the
freeze: 8 of 8 primary clauses pass (standard_results.json,
eval_standard.log).

What-the-sink-does results (2026-10-13, 00:21 to 00:40 UTC, head 97d2837).
Qwen2.5-1.5B 321 s, TinyLlama 165 s, Mistral-7B 359 s, Qwen2.5-7B 306 s;
sinkint attention identical to eager attention in all four (maximum logit
difference 0.0), restore checks 0. Evaluation unchanged since the freeze:
4 of 4 primary clauses pass (sink_results.json, eval_sink.log). TinyLlama
has no layer whose first-token mass exceeds 0.3, so K3 rests on the other
three models; its sink is not concentrated on the first token.

Long-context round results (PREREGISTRATION14.md, 2026-10-13, 00:41 to 01:20
UTC, frozen at 18901f6): Qwen2.5-1.5B 997 s and TinyLlama 1,357 s at 1,024
tokens, restore checks 0. Evaluation (eval_long.py, unchanged since the
freeze): 6 of 7 clauses pass; S5 fails narrowly (merges chosen by output
perturbation cost 0.83 and 0.82 of the generator-cosine choice, both below 1,
geometric mean 0.82 against the registered 0.8). The Llama answer-span test
queued after the sink round did not run at first: its queue script passed an
undefined variable as the source path and the test stopped at its own guard
(log kept in the scratchpad); it was relaunched after this round with the
absolute path.

Llama-2-7B-80K answer-span inclusion test (2026-10-13, 01:20 to 01:31 UTC):
20 of 20 retrieved, 20 of 1,024 heads above 0.1 (1.95 percent), below the 2
percent floor by one head; excluded as the rule is written (addendum 11).

Target 1 battery rerun after addendum 9 (2026-10-12 22:23 to 2026-10-13
01:31 UTC, 11,251 s at low priority beside the GPU rounds): separator-column
null reproduces 0.638 of the contrast (0.602 before the fix) and 35.4
percent same-layer nearest neighbors (34.7 before); marginal null 0.095;
random and untrained nulls about zero. Label shrinks; the registered
expectation still fails on the marginal clause. audits/RESULTS.md, Figure 13
and the public text regenerated or updated.

Multi-head importance round (PREREGISTRATION15.md, 2026-10-13, 16:06:50 to
16:27:51 UTC, head 46fd50c): gpt2-medium 591 s, Pythia-160m 100 s,
Pythia-410m 564 s; restore checks 0 on both corpora. Evaluation
(eval_importance_mha.py, unchanged since the freeze): 7 of 7 clauses pass
(importance_mha_results.json, eval_importance_mha.log). The labeled
development run of preregistration 16 shared the GPU for about five minutes
of this round (16:16 to 16:22 UTC); it reproduced the stored statistics of its two decoders
exactly while sharing.

Correction (2026-10-13): Voita et al. (2019) define a head's confidence as
its average maximum attention weight excluding the end-of-sentence token
(their footnote 2). Preregistrations 12, 14 and 15 computed it over all
tokens, sink included, and called it their statistic. Found while writing up
round 15. The registered results stand for the statistic as computed;
preregistration 16, frozen at f5a9539 before its run, tests the definition as
written. Public text is corrected with that round's results.

Voita-definition round (PREREGISTRATION16.md, frozen at f5a9539,
2026-10-13). The first attempt (16:28:07 to 16:34:27 UTC) stalled while
downloading opus-mt-en-ru: the Hugging Face transfer client stopped at 67 MB.
No statistic had been computed and no output written; it was stopped. The
three translation models were then downloaded separately over plain HTTP
(HF_HUB_DISABLE_XET=1, for the downloader only). Decoders ran 16:46:15 to
16:50:16 UTC (seven models, 9 to 75 s each; recomputed all-token confidence
and sink mass equal the stored values exactly), opus-mt-en-ru and
opus-mt-en-fr 16:57:09 to 17:06:49 (251 s, 240 s), opus-mt-en-de 17:09:38
to 17:13:38 (234 s) once its download finished; translation restore checks
0. The run scripts ran at head 54d6def; voita_definition.py and eval_voita.py
are unchanged since the freeze (empty git diff against f5a9539). Evaluation:
5 of 6 clauses pass, V5 fails (voita_results.json, eval_voita.log).

Readings round (audits/PREREGISTRATION17.md, frozen at 94b7734). Both halves
launched 2026-10-13 20:20 UTC with --registered=<freeze permalink>,
concurrently on the GPU. The laptop ran on battery: the guard paused both
runs at 24 percent at 21:02:51 UTC and resumed them on AC at 03:24:36 UTC on
2026-10-14. A second pause at 11 percent (03:38:58 to 03:39:18) came while
both runs drew more than the 65 W adapter supplies. A hysteresis watcher
(hold at 6 percent while discharging, release at 20) was added for the CHAI
half and never triggered. Clark finished 03:52:39 UTC, Mistral-7B 04:00:00,
OPT-6.7B 04:35:24 (2,101 s). The guards logged exit status 145 for the two
runs they had stopped and resumed (zsh's record of a stopped job); both runs
printed their completion lines and wrote complete result files. The scripts
are unchanged since the freeze (empty git diff against 94b7734). Evaluation
(eval_readings.py, unchanged): 8 of 9 clauses pass, C2 fails
(readings_results.json, eval_readings.log).

ViT attention-distance round (audits/PREREGISTRATION19.md, frozen at 8455f04,
2026-10-15 00:12:00 to 00:17:38 UTC): DINOv2-L, DINOv2-L with registers,
ViT-L/16, ViT-B/16 on 128 images, ran beside preregistration 18's OPT-6.7B
run. transformers 5's image processors require torchvision; each model's own
preprocessor_config.json is applied with PIL and numpy instead (frozen code).
Evaluation (eval_vit.py, unchanged): 3 of 5 clauses pass, V1 and V4 fail
(vit_results.json, eval_vit.log).

Merge benchmark (alignment_study/PREREGISTRATION20.md, frozen at 283e5b5,
2026-10-15 00:21:32 UTC to about 01:08 UTC): gpt2-medium 1,329 s, then
Pythia-160m and Pythia-410m, beside preregistration 18's 7B runs. Evaluation
(eval_merge_bench.py, unchanged since the freeze): 5 of 5 clauses pass
(merge_bench_results.json, eval_merge_bench.log).

Rule round (audits/PREREGISTRATION18.md, frozen at cdda0f1, 2026-10-15
00:03:48 to 01:46:06 UTC): L1, L2, L3 in minutes; A1 and A2 on OPT-6.7B
(1,897 s) and A3 on Mistral-7B-Instruct-v0.2 (3,955 s), most of it the
untrained 7B initializations, which are built on the CPU and copied to the
GPU and ran slowly under memory pressure (swap up to 37 GB, disk free down
to 23 GB at its lowest; nothing was paused or lost). Evaluation (eval_rule.py,
unchanged since the freeze): K = 4 of N = 6 (A2, A3 fail); the
development-informed secondary reading fits 6 of 6 (rule_results.json,
eval_rule.log).

Sink-architectures round (alignment_study/PREREGISTRATION21.md, frozen at
8a15d0a, 2026-10-15 01:46:27 to 01:53:48 UTC, run at head fbf5f37 with the
round's scripts unchanged since the freeze): Part E (five models) and Part W
(three models). OPT-6.7B's equivalence check is not exact in bfloat16
(maximum logit difference 0.44; recorded in the result file and the note).
Evaluation (eval_sink_arch.py, unchanged): 3 of 6 clauses pass, G2, G3 and G6
fail (sink_arch_results.json, eval_sink_arch.log).

Merge benchmark at scale (alignment_study/PREREGISTRATION22.md, frozen at
f5d3b5e, 2026-10-15 21:13:30 to 2026-10-16 04:21:47 UTC): Part S Pythia-1.4B
3,975 s, OPT-1.3B 13,802 s (11,904 merges), GPT-2 large 5,225 s; Part D six
models, 2,662 s. Evaluation (eval_merge_bench2.py, unchanged since the
freeze): 6 of 7 clauses pass, D2 fails (merge_bench2_results.json,
eval_merge_bench2.log).

Weighting-flip round (audits/PREREGISTRATION23.md, frozen at 50fdbc4,
2026-10-16 04:21:57 to 04:53:59 UTC): Clark 19 s, CHAI 479 s, OPT-6.7B 428 s,
SnapKV 922 s. Evaluation (eval_flip.py, unchanged since the freeze): 4 of 5
clauses pass, F2 fails (flip_results.json, eval_flip.log).

MHA-to-GQA conversion round (alignment_study/PREREGISTRATION24.md),
development, 2026-10-16. The zero-shot harness (zeroshot.py) first ran more
than 25 minutes on Pythia-410m without finishing: MPS rebuilt its kernels for
nearly every batch, each of a new shape; bucketing lengths (multiples of 32,
then 64) and padding the output-head rows to multiples of 512 brought the
35,551 requests to 204 s. Its base numbers on Pythia-410m and GPT-2 medium
match EleutherAI's published lm-evaluation-harness values within 0.3 points.
The first conversion code spent 463 s on OPT-125m's partitions, most of it in
generalized Procrustes with SVDs falling back to the CPU one at a time; the
rewrite on head-dimension Gram blocks gives the same 144 partitions in 168 s.
Casting to float64 on the MPS device fails; the arrays now move to the host
first. Code-path check (OPT-125m, Llama-160m, Phi-1.5) 3,427 s; development
runs GPT-2 medium 3,165 s and Pythia-410m 2,573 s; convention null 354 s.
Frozen at 7f12472; confirmatory runs started 2026-10-16 11:57 UTC, the five
models in sequence and then the convention null, under caffeinate with a
battery guard that pauses below 20 percent when discharging.

MHA-to-GQA conversion round, first confirmatory attempt (2026-10-16 11:57 to
14:03 UTC): OPT-1.3B's partitions took about an hour and its unconverted
evaluation 3,410 s, about four times the expected time, because the machine
was swapping at about 250 MB/s in each direction. The process held 29 GB of
GPU memory for 8 GB of live tensors: each 12,288-token scoring batch returned
a key-value cache for every layer (4.8 GB on OPT-1.3B), and the MPS allocator
kept the freed blocks. The run was stopped. zeroshot.py now passes
use_cache=False when scoring and empties the allocator cache after every
batch and every evaluation; neither changes a computed value (on the first 300
examples of each task, the per-example results with and without the change,
and the first attempt's full-run results on the same examples, agree in all
2,100 cases), and the footprint fell to about 9 GB. The first attempt's
unconverted OPT-1.3B evaluation is kept (alignment_study/gqa/opt-1.3b.json.partial).
The models are now run one invocation each (--only), fastest first; the order
does not affect any result.

Cross-layer KV compression, exploration and development (2026-10-17, while
round 24 ran on the GPU; preregistration 25). Exploratory scratch probes, not
registered: per-token cosine between adjacent layers' keys and values, trained
basis, held-out, near zero in TinyLlama, Qwen2.5-0.5B, Pythia-410m and
yaofu/llama-2-7b-80k (every layer pair within 0.07 of zero in the last, run in
bfloat16 on the CPU); the residual stream of the last rose from 0.61 to 0.97
with depth. A MiniCache-style merge on 16 documents was no better in the
trained basis than in random function-preserving bases and much better after
alignment in all three small models. XQuant's code-difference statistic matched
the independence prediction. A per-head sink-logit correction after round 24's
GPT-2 medium conversion removed the sink drift but not the damage (396 to 405,
82 to 79). GPT-2 medium returns NaN logits on the CPU in float32 with this
transformers version (the MPS runs are unaffected); the CPU probes used other
models. The development runs of xlayer_round.py ran on the CPU (XLAYER_DEVICE)
to leave the GPU to round 24: TinyLlama 2,901 s, Qwen2.5-0.5B 1,277 s,
Pythia-410m 1,482 s. A first development attempt on the GPU alongside round 24
drained the battery while plugged in (68 to 25 percent): the adapter cannot
feed both, so it was stopped and the guard now pauses at 15 percent whatever
the reported charging state.

MHA-to-GQA conversion round, confirmatory (alignment_study/PREREGISTRATION24.md,
frozen at 7f12472; run from 2026-10-16 14:30 UTC to 2026-10-18 04:26 UTC with
the round's scripts unchanged since the freeze except zeroshot.py's memory
fix, 5556656): GPT-2 large 14,437 s, Pythia-1.4B 22,260 s, OLMo-1B 17,877 s,
OPT-1.3B 20,689 s (its unconverted evaluation reused from the first attempt),
Phi-2 56,417 s (including a pause of 1 h 21 min by the battery guard), then
the convention null, 4,920 s for the five models. Evaluation (eval_gqa.py,
unchanged since the freeze): 9 of 10 clauses pass, P6 fails
(gqa_results.json, eval_gqa.log).

Closed-form fusion for grouped-query conversion (preregistration 26),
exploration and development, 2026-10-18, while round 24's convention null ran on
the GPU. Exploratory scratch probes on the CPU, not registered: a least-squares
refit of the output projections after round 24's mean pooling (Pythia-410m,
g = 2: 171 to 90 with sim_align's groups, 2,737 to 53,208 without alignment, 182
with a target that also corrected the residual-stream drift); the closed-form
fusion with and without query and output weighting on Pythia-410m, and on
neighbouring groups of Pythia-160m, OPT-125m, Llama-160m and GPT-2; a
sequential variant of the fusion (slightly worse, not used); a key-and-value
split probe was written but not run. The first code-path check on OPT-125m
returned non-finite weights for groups of one (an eigenvalue of zero in a
direction without energy); such directions are now left out. Round 25's
confirmatory pipeline was stopped before it started and is queued after round
26's confirmatory runs, its preregistration and scripts unchanged. Development
runs (fuse_round.py --dev, GPU): GPT-2 medium 1,499 s, Pythia-410m 1,427 s;
architecture checks for Phi-1.5 (871 s) and OLMo-1B (705 s) on the CPU.

Closed-form fusion round, confirmatory (alignment_study/PREREGISTRATION26.md,
frozen at 16404cf; run from 2026-10-18 05:35 UTC to 2026-10-19 09:22 UTC with
the round's scripts unchanged since the freeze): GPT-2 large 11,029 s,
Pythia-1.4B 13,769 s, OLMo-1B 12,649 s, OPT-1.3B 12,794 s, Phi-2 34,726 s. The
GPT-2 large run was restarted twice (07:18 and 09:46 UTC): evaluations ran 5
to 8 times slower than round 24's (the unconverted model's took 4,838 s against
555 s), and its fusion and check phases varied from 43 s to 991 s. The restart
notes in fuse.log guessed memory compression, after CPU probes run alongside
had filled memory; the cause found afterwards was heat. With the laptop's lid
closed the GPU sustained 1.1 to 1.5 TFLOPS in a matrix-multiply loop against
8.8 in a short burst, and with the lid opened (2026-10-18 about 11:20 UTC) it
sustained 7.8 to 8.7 and evaluations returned to about 590 s. No result was
written before the second restart except GPT-2 large's unconverted
evaluation, which the partial file kept (it equals round 24's exactly).
Evaluation (eval_fuse.py, unchanged since the freeze): 6 of 6 clauses pass
(fuse_results.json, eval_fuse.log). Round 25's confirmatory pipeline started
on completion (2026-10-19 09:22 UTC).

Cross-layer round (alignment_study/PREREGISTRATION25.md), first confirmatory
attempt (2026-10-19 09:22 to 10:14 UTC): LLaMA-2-7B-80K's Parts A and B
completed (their lines are in xlayer.log) and Part C stopped at its first
quantized arm, because the KIVI quantization converted the cache to float64 on
the GPU, which MPS does not support (the development runs were on the CPU).
The fix moves that conversion to the host and returns the result to the
cache's device; on CPU tensors it changes nothing (outputs and codes
bit-identical for float32 and float64 inputs), and on the GPU it reproduces the
CPU result exactly; the whole instrument then ran end to end on the GPU on
llama-68m. No result file had been written; the run restarts from the first
model with the fix.

Cross-layer round, confirmatory (second attempt, with the device fix 82cadd2;
2026-10-19 10:16 to 14:17 UTC): LLaMA-2-7B-80K 4,226 s (Parts A and B
identical to the first attempt's), Mistral-7B-Instruct-v0.2 3,735 s,
Qwen2.5-7B 3,783 s, Phi-3-mini 2,705 s. Evaluation (eval_xlayer.py, unchanged
since the freeze): 5 of 7 clauses pass, B2 and C3 fail (xlayer_results.json,
eval_xlayer.log).

Fusion in models that already share key-value heads (preregistration 27),
exploration and development, 2026-10-19. Exploratory CPU probe: Qwen2.5-0.5B
with its two key-value heads merged into one, perplexity 9,506 with averaging,
256 with the unweighted fit and 164 with the weighted fusion (16 windows). A
dry run on Mistral-7B-v0.1 in bfloat16 that held the float64 second moments of
all 32 layers (4.3 GB) and the weight snapshot (2.7 GB) in memory, alongside
the user's applications, hung the machine at 17:07 local time (kernel panic,
watchdog timeout); nothing in the repository was lost. The instrument now keeps
second moments in float32, promoted per layer when fused (perplexities
unchanged to four decimals on the development models), and the snapshot on
disk; GPU jobs run under a memory watchdog that kills them if free memory falls
below 12 percent or swap grows by 3 GB. The rotary step was vectorized
(converted weights equal the per-plane loop's to 1e-9 relative) and two
products reassociated, which brought a 7B conversion from over 17 minutes to
about 10. The second dry run (fusekv_dryrun.py; log fusekv_dev_mistral_dryrun.log)
held 15 to 16 GB on the GPU with the system at least 67 percent free.
Development runs with the final code (fusekv_round.py --dev, GPU): Qwen2.5-0.5B
636 s, Llama-160m 396 s; they reproduce the first development run's
perplexities to the printed precision.

Head-to-head with CAGA on Llama-2-7B (preregistration 28), development,
2026-10-28. Llama-2-7B comes from the NousResearch/Llama-2-7b-hf copy at
revision 8efe6c9b; both safetensors files have the SHA-256 hashes listed in the
official meta-llama/Llama-2-7b-hf metadata. Development on Llama-160m (GPU,
261 s) and OpenLLaMA-3B v2 with CAGA's calibration of 128 documents of 2,048
tokens (second moments 543 s). On OpenLLaMA-3B the groups-of-one check gave a
maximum logit difference of 12.5 at the first position (median over positions
0.5), perplexity 7.6921 against 7.6865 on 16 windows: the bfloat16 rounding of
the rewritten weights. The pass was stopped after its base evaluation to look
at this (llama2_numerics_check.py). The first version of the check moved the
weights from the GPU to the host with .to("cpu", torch.float64); in torch 2.13
on MPS that call returns corrupted values for large tensors (errors of 5 to 11
on unit-scale weights, from float32 sources as well) and raises an error for
small ones, while .cpu().double() is exact, as is every host-to-GPU conversion
tested. No committed script uses the one-step form (all scripts searched); the
check now uses .cpu().double(). On Llama-160m the check found that bfloat16
rounding costs the weighted fusion 0.07 percent of perplexity with groups of one
and nothing measurable with pairs (62.17 in bfloat16, 62.14 in float32, against
22.91 and 22.90 unconverted); a change of gauge that makes groups of one exact
in bfloat16 leaves the function unchanged (identical perplexity in float32) but
makes pairs slightly worse in bfloat16 (62.61 against 62.17 weighted, 91.1
against 85.8 unweighted), so the round keeps round 27's conversion. The
OpenLLaMA-3B pass was resumed from its partial result with three arms.

Head-to-head with CAGA, confirmatory (frozen at 0abdd89; 2026-10-28 23:00 to
2026-10-29 14:45 UTC, 56,700 s, one job on the GPU under the memory watchdog and
the battery guard). On the 45 W adapter the job drained the battery even while
charging, and the guard paused it at 15 percent and resumed it at 30 nine times
(13 to 26 minutes each); three evaluations include pauses (base 11,458 s,
fuse/paired 10,253 s, caga/neighbour 7,376 s), the other three ran without
them (6,588 to 6,625 s).
Second moments 929 s, groups-of-one check 705 s, pairing 116 s, each fused or
CAGA conversion 654 to 710 s. Evaluation (eval_llama2.py, unchanged since the
freeze): 6 of 6 clauses pass (llama2_results.json, eval_llama2.log). During the
pauses, two checks of tools that are not part of the round ran on the CPU: the
export to a real grouped-query checkpoint (gqa_export_check.py; Llama-160m, the
exported checkpoint with 6 key-value heads reproduces the simulated conversion
to 2e-4 in logits) and the conversion command (kvfuse_check.py; its pairing
costs equal the frozen round's exactly and choose the same pairs in all 12
layers, and its saved checkpoint reproduces the simulated conversion to 4e-4).
Round 27's last model, Qwen2.5-7B, was then rerun from the start with the same
instrument (the first run ended without output after its groups-of-one check).

Round 27's Qwen2.5-7B, second attempt (2026-10-29 14:45 UTC): the memory watchdog
killed the job at 14:58 UTC, after the groups-of-one check (maximum logit
difference 0.97, as in the first attempt) and at the start of the base
evaluation, with free memory at 6 percent and 12.3 GB of swap. A second 7B job
was running alongside it: a parallel session of the assistant, opened by the
author for questions, had queued the exploratory follow-up to round 28
(llama2_caga_followup.py, Llama-2-7B) to start when round 27 finished, keyed on a
line that round 27's first attempt had already written, so it started at 14:47
UTC; the watchdog killed both. Two 7B models do not fit in 32 GB, so this
explains the second attempt's end; the first attempt's watchdog log was lost with
the reboot and its cause is not recorded. The sessions now coordinate before
using the GPU. One further memory peak was found and removed: the zero-shot
scorer projected all continuation rows of a batch onto the vocabulary at once,
which with Qwen2.5's 152,064-token vocabulary comes to several GB of float32
log-probabilities for a long-answer batch (by calculation), five times what
Mistral's 32,000-token vocabulary needs. The scorer now projects the rows in
chunks of 512 (zeroshot.py, HEAD_CHUNK). Checked on Qwen2.5-0.5B in bfloat16 on
the GPU (zeroshot_head_chunk_check.py, 300 examples of each of the seven tasks):
log-likelihoods differ by at most 3e-5, greedy matches and every per-example
decision are identical, and the averages agree to every digit. Round 28's
evaluations ran before this change. The pipeline's evaluator ran on the four
finished models again (not the registered evaluation; it is regenerated when
Qwen2.5-7B finishes). Qwen2.5-7B restarts from the beginning, alone.

Round 27's Qwen2.5-7B, third attempt (2026-10-29 17:22 to 2026-10-30 02:30 UTC,
alone on the GPU, scorer chunked): completed. Evaluation (eval_fusekv.py, unchanged
since the freeze) on all five models: 3 of 4 clauses pass, K2 fails
(fusekv_results.json, eval_fusekv.log). The exploratory follow-up to round 28
(llama2_caga_followup.py, started by the parallel session and run alone between
the second and third attempts) gave perplexity above 9,000 for both CAGA's
construction and the weighted fusion on pairs matched by CAGA's residual; it is
not reported until a diagnostic (llama2_followup_diag.py) locates the damage.

Exploratory follow-up to round 28 (not preregistered; 2026-10-29 to 30, every
job alone on the GPU under the memory watchdog and the battery guard). The
parallel session's follow-up (llama2_caga_followup.py, which kept the second
moments on disk to save memory) ran between round 27's second and third
Qwen2.5-7B attempts: second moments 2,563 s, CAGA's construction and the fusion
on pairs matched by CAGA's residual 2,660 and 2,621 s (llama2_followup.json;
log llama2_followup.log, which also holds the next two runs). A diagnostic
(llama2_followup_diag.py, 2026-10-30 02:31 UTC, 7,972 s with battery pauses)
measured both pairings layer by layer. A clean rerun (llama2_followup_rerun.py,
04:44 UTC, 8,475 s with one pause) repeated the two full conversions with the
second moments in memory through round 28's convert function and reproduced
both perplexities to every printed digit, so the on-disk second moments were
not the cause. The bisection (llama2_followup_bisect.py, 07:12 to 08:45 UTC,
5,580 s) converted layers cumulatively and swapped single layers, restoring
unconverted weights from the checkpoint files (checked identical to the loaded
model; restored perplexity equal to the unconverted value both times). The
mechanism check (llama2_followup_mechanism.py, 08:45 to 09:07, 1,319 s;
second moments of layers 0 and 1 only, the forward pass stopped after layer
1), the first-layers test (llama2_followup_firstlayers.py, 09:07 to 10:51,
6,219 s; layers converted one at a time through round 28's frozen convert
function on a one-layer view of the model), the neuron check
(llama2_followup_neuron.py, 10:51 to 10:57, 306 s), the quantization probe
(llama2_quant_sink_probe.py, 10:57 to 11:15, 1,043 s) and the first-token
check (llama2_quant_firsttoken.py, 11:15 to 11:18, 140 s) followed, each
started by a waiter keyed on the previous job's process ID. The parallel
session also ran a CPU probe on Llama-160m (llama2_dev/prefix160m.py,
prefix160m.json) while the GPU was free; it may have overlapped the
bisection's start by a few minutes, which does not affect an exploratory run.

Sequential conversion, development (exploratory, labelled; 2026-10-30;
fuse_seq.py, seqfuse_dev.py, alignment_study/seqfuse_dev/). The weighted fusion
fitted layer by layer on the inputs the converted layers below produce, with and
without a QEP-style correction of each layer's projections toward the
unconverted model's inputs. The layer-by-layer stream reproduces the model
exactly (largest difference 0). On Llama-160m (float32, 128 C4 documents of
1,024 tokens, 40 WikiText-2 windows of 2,048 tokens, unconverted 73.98) the
one-shot fit gives 113.1 on matched pairs and 123.3 on neighbouring pairs; the
sequential fit 120.9 and 130.9 (119.8 on the one-shot pairs), the correction
123.9 (alpha 0.5) and 136.2 (alpha 1.0): worse in every arm
(seqfuse_dev_160m.log). On OpenLLaMA-3B (float32, 128 documents of 2,048
tokens, unconverted 7.60) the one-shot fit gives 30.28 on matched pairs
(seqfuse_dev_3b.log). The machine then panicked (watchdog timeout, 11:56 UTC,
the same failure as on 2026-10-19): between arms the script dropped its
reference to the model but its key-value view still held it, so the next copy
was loaded beside the old one (two float32 3B models, 27 GB, plus the 3.4 GB
calibration stream), and the script's name did not match the memory
watchdog's pattern. Nothing in the repository was lost. The development
scripts now free the model and its views before loading another (also
uptrain_round.py and guard_round.py), the sequential stream is updated in
place, and both guards live in private/guards/ (gitignored, so a reboot no
longer deletes them) and match every alignment_study/*.py process. The
OpenLLaMA-3B sequential arms, rerun after the reboot, give 33.27 on pairs
re-matched on the converted stream and 33.68 on the one-shot pairs, against
30.28 one-shot: worse here too. The sequential fit is not pursued.

Spurious massive activations after key-value head merging (preregistration 29).
Development on Llama-2-7B (guard_round.py --dev, 2026-10-30): the first
attempt was paused by the battery guard at 20 percent with the laptop off AC,
then stopped by hand at 8 percent before it had written anything, so that the
machine could sleep without a 20 GB job in memory; a waiter in private/guards/
restarted it on AC at 30 percent. That run failed at the reference check: the
writers' gate vector reused the name of the check's window (gw), so the window
became a float tensor; renamed and checked end to end on Llama-160m, the
survey ran 15:54 to 17:06 UTC (4,317 s, 78 members). Frozen at 4d02534 with the
development values (threshold 2.35, severe at 5 times, the strongest-writer
rule for G5). Confirmatory run 17:08 to 19:49 UTC (9,636 s, one job on the GPU
on AC power under both guards, no pauses): Mistral-7B-Instruct-v0.2 4,356 s,
OpenLLaMA-3B 2,662 s, OLMo-1B 754 s, Qwen2.5-7B 1,861 s. Evaluation
(eval_guard.py, unchanged since the freeze): the gate fails, G1 to G5 untested,
0 of 6 (guard_results.json, eval_guard.log).

Energy-preserving fusion, development (exploratory, labelled, not pursued;
2026-10-30; energy_fuse_dev.py, energy_fuse_dev/, energy_fuse_dev_160m.log).
The fused heads' least-squares fit keeps less output energy than the original
heads (median 0.885, at least 0.524, of E|O_h C_h x|^2 under the calibration
second moment on Llama-160m), and the converted Llama-2-7B's residual stream
runs 4 to 11 percent below the unconverted model's through layers 2 to 30. On
Llama-160m (float32, 128 C4 documents of 1,024 tokens, 40 windows of 2,048,
unconverted 73.98, the fusion on matched pairs 113.13) rescaling every fused
query head's output slice to restore the energy gives 122.73, and restoring half
of it in log scale 116.62: worse both ways. Not pursued. Rescaling merged
units' statistics after merging is prior work (REPAIR, Jordan et al. 2023; bias
correction after quantization, Nagel et al. 2019); the test here was only
whether a closed-form per-head rescale inside the weighted fusion helps.

Uptraining and rebalancing, development (exploratory, labelled; 2026-10-30;
uptrain_round.py --dev, fuse_balance.py, balance_dev.py; uptrain_dev/,
balance_dev/, uptrain_dev_*.log). Uptraining trains the attention projections
on C4 text after document 10,000 (AdamW, 16 sequences of 1,024 tokens a step).
On Llama-160m at learning rate 1e-4 and 0.2M tokens: the unconverted model
23.18 to 23.90 on WikiText-2, averaging 727 to 285, the weighted fusion 50.6 to
38.4. On TinyLlama (4 to 2 key-value heads) the fusion diverged at 1e-4 with a
one-step warmup (55.4 to 39,009 in two steps); with at least 10 warmup steps at
2e-5 the unconverted model stayed at 8.38 while the fusion fell to 38.8 and then
rose to 63.8 over 0.33M tokens. The fused weights are badly scaled: the fusion
builds the shared value in a whitened basis and the output projection absorbs
the inverse, so in TinyLlama the value rows grow to median norm 2.72 (largest
49.6) and the output columns shrink to 0.106 (smallest 0.0008), against 0.46
and 0.51 before, and the query planes reach 31.6. fuse_balance.py rescales,
without changing the function, every shared value row against the matching
output columns and every rotary plane of the shared key against the same plane
of the queries, to equal norms (the minimum-norm factorization of each product;
the same kind of function-preserving rescaling as cross-layer equalization in
data-free quantization, Nagel et al. 2019). Logits change by at most 5.6e-6
(Llama-160m) and 4.9e-6 (TinyLlama) of the largest logit. With the attention
projections rounded to 4 bits per output channel, perplexity is 138.3 against
144.1 unbalanced on Llama-160m (fused 113.7) and 62.0 against 71.8 on TinyLlama
(fused 54.5); at 3 bits 189.6 against 204.1 and 217.4 against 226.7; bfloat16 and
8 bits change little either way. Trained at 2e-5, the rebalanced fused TinyLlama
falls steadily from 55.4 to 21.2 in 0.33M tokens (zero-shot 0.377) where the
unbalanced one ends at 63.8 (0.329). Round 30's draft is redesigned around this.

Rebalancing fused key-value heads (preregistration 30, frozen at ab9827e). The
first registered attempt (21:51 UTC) finished Part Q for OLMo-1B and
h2o-danube-1.8b and was killed by the memory watchdog at 22:21 UTC during
OpenLLaMA-3B (free memory 12 percent): the rounding part held two copies of the
attention weights at once (4.3 GB each in float32) beside the model. The
instrument now keeps one copy at a time and computes the same quantities in the
same way (balance_round.py, commit after ab9827e; the two finished models are
kept). The evaluator's output from the killed pipeline, written on incomplete
data, was deleted; the registered evaluation runs once every model is done.
Resumed at 22:21 UTC with that fix; Part Q finished 00:13 UTC (OpenLLaMA-3B
923 s, Llama-2-7B 1,993 s, Mistral-7B-Instruct 1,910 s, Qwen2.5-7B 1,865 s) and
Part T ran 00:13 to 06:38 UTC (each OLMo-1B run about 607 s, each
h2o-danube-1.8b run about 990 s; the fused starts recompute the second moments
in every run), on AC power with no pauses. Evaluation (eval_balance.py,
unchanged since the freeze): 4 of 6 (balance_results.json, eval_balance.log).
After the round, kvfuse.py fuses and rebalances each layer in float32 and
rounds to the model's dtype once (--no-balance skips the rebalancing;
fuse_balance.balance takes a list of layers). Checks: kvfuse_check.py on
Llama-160m (CPU, float32): the rebalanced checkpoint reproduces the simulated
unbalanced fusion to 3.5e-4 in logits of up to 22.7, pairing unchanged
(llama2_dev/kvfuse_check_llama-160m_balanced.json); kvfuse_balance_check.py on
TinyLlama (bfloat16 checkpoints, 8 windows): 58.93 rebalanced against 58.63
unbalanced as saved, 66.4 against 76.8 with the attention projections at 4 bits
(balance_dev/kvfuse_balance_check_TinyLlama_v1.1.json).

Rebalancing under GPTQ and for the key-value cache, development (exploratory,
labelled; quant_balance_dev.py, balance_dev/quant_*.json; 32 C4 documents of
1,024 tokens). Fused model on matched pairs, unbalanced against rebalanced.
Llama-160m (20 windows; fused 105.0): attention projections at 4 bits by round
to nearest 130.7 against 125.5, by GPTQ 114.0 against 110.3; at 3 bits 182.1
against 171.0 and 155.5 against 132.4; key-value cache at 4 bits per token and
head 112.0 against 108.9, at 2 bits 268.5 against 279.3. TinyLlama (8 windows;
58.5): 4 bits 77.1 against 67.6 (round to nearest) and 71.0 against 67.2
(GPTQ); 3 bits 235.8 against 251.3 and 101.2 against 80.7; cache at 4 bits 58.0
against 53.3, at 2 bits 312 against 425. Rebalancing keeps helping under GPTQ
and for a 4-bit cache, and hurts a 2-bit cache.

Cross-layer sharing and SVD-LLM rebalancing, development (exploratory,
labelled; xlayer_fuse_probe.py, xlayer_cost_probe.py, svdllm_balance_dev.py;
xlayer_fuse_dev/, balance_dev/svdllm_*.json). A layer's keys are 93 to 98
percent (Llama-160m) and 98 to 99 percent (TinyLlama) linearly predictable from
the previous layer's attention input, its values (output weighted) 73 to 87 and
81 to 96 percent. In the fusion's own weighted residual, sharing one cache
between adjacent layers of TinyLlama costs 0.008 to 0.028 of key energy and 0.14
to 0.32 of value energy, against 0.006 to 0.025 and 0.15 to 0.28 for pairing
heads within each layer at the same saving; the prediction residual alone is
0.03 to 0.15 of the value energy. Not pursued. Rebalancing SVD-LLM's factors (the
MLP projections of Llama-160m at half their parameters, perplexity 2,356 against
67.0 unconverted) makes rounding worse, 3,334 against 1,983 at 4 bits and 26,097
against 3,320 at 3 bits: SVD-LLM's split already scales each component by the
square root of its singular value, which gives the important components the
most rounding precision, and equal norms undo that. The rebalancing of round 30
is specific to the fusion's whitened construction, not a rule for whitened
factorizations in general.

Attention-weighted calibration for the fusion, development (exploratory,
labelled; attnw_fuse_dev.py, attnw_fuse_dev/; 32 C4 documents of 1,024 tokens,
pairs fixed at the fusion's matching under the plain second moment). The fusion
fits its shared heads in the metric of S = E[x x^T], every token alike; here S is
mixed with S_att, which weights each token by the attention it receives
(averaged over the layer's heads and query positions, scaled so that uniform
attention gives S), as (1 - lambda) S + lambda S_att, used for keys, values and
query energies alike. WikiText-2 perplexity at lambda 0, 0.25, 0.5 and 1:
Llama-160m (20 windows, unconverted 67.0) 105.00, 103.51, 104.24, 107.60;
TinyLlama (8 windows, 7.24) 58.51, 47.17, 40.53, 36.46; OpenLLaMA-3B (8 windows,
6.77) 28.40, -, 25.62, 24.83. Controls on TinyLlama at lambda 1: the first
position left out of the weighting 36.56; a position-only weighting (the
attention uniform causal attention would give) 59.89 at 0.5 and 61.26 at 1;
query energies from the plain moment 39.39, and pairs re-matched under the mixed
moment as well 39.44; the received attention estimated from every 8th query
position 36.40. The rule for round 31's lambda (minimax relative loss over the
three development models on {0, 0.5, 1}) was committed before OpenLLaMA-3B's
lambda-1 value was read (5f061e1) and picks lambda 1 (worst loss 3.2 percent,
Llama-160m, against 11.2 percent for 0.5). Closest prior work: the
attention-aware Hessians of BoA (Kim et al., ICML 2025) for quantization.
Equalization on the original models (cle_original_dev.py; fuse_balance.py with
every KV head alone): no gain, Llama-160m 76.6 against 76.9 at 4 bits and 157
against 160 at 3, TinyLlama 7.54 against 7.53 and 10.84 against 10.86, so round
30's rebalancing gain comes from the fusion's imbalance.

Fitting the shared heads where attention looks (preregistration 31, frozen at
207a5ec and pushed before the run). Ran 07:40 to 10:24 UTC on AC power, one GPU
job under both guards, no pauses or kills (attnw_round.py; attnw.log): OLMo-1B
284 s, h2o-danube-1.8b 488 s, Mistral-7B-Instruct-v0.2 2,519 s, Llama-2-7B 4,705 s
(with the whole test set twice), Qwen2.5-7B 1,835 s. Evaluation (eval_attnw.py,
unchanged since the freeze): 0 of 3 (attnw_results.json, eval_attnw.log). The
plain arm reproduced round 30's fused perplexity to four decimals in OLMo-1B,
h2o-danube-1.8b, Mistral-7B-Instruct and Llama-2-7B (54.7016, 38.8145, 16.6787,
15.9997) and round 28's whole-test 16.78 on Llama-2-7B (16.7796), but gave
39,380.6 for Qwen2.5-7B against round 30's 23.8729 on the same pairs, calibration
and windows. Cause: attnw_round.moments accumulates S and S_att of x alone,
while fuse_kv.second_moments appends a ones column for layers whose projections
carry biases and fuse_kv.fuse_kv_group reads the bias only when the moment has
that column; with a d x d moment it fitted Qwen2.5-7B's weights and left every
query, key and value bias in the old basis. Both Qwen2.5-7B arms are invalid as
computed; the evaluator ran as frozen and its output stands. Post hoc
correction, not preregistered and not scored (attnw_biasfix.py, which reuses
attnw_round.run_model with the column appended to both moments, the attention
weight multiplying it as every other entry; attnw_correction/,
attnw_biasfix.log), 10:31 to 11:11 UTC (2,394 s): plain 23.8729 (round 30:
23.8729), weighted 27.7764. Any routine that builds second moments for the fusion must append the
ones column when kv.has_bias(l), as fuse_kv.second_moments does.

Attention-derived metrics for the fusion, development (exploratory, labelled;
attnw_mix_dev.py, committed with the rule that reads it at 9c17ad1;
attnw_mix_dev/, attnw_mix_dev.log; 32 C4 documents of 1,024 tokens, queries every
8th position, pairs fixed at the fusion's matching under S). Arms (key metric /
query energies / value metric) and WikiText-2 perplexity on Llama-160m (20
windows), TinyLlama and OpenLLaMA-3B (8 windows): plain S/S/S 105.00, 58.51,
28.40; round 31's S_att throughout 107.34, 36.40, 24.85; S_att/S_att/S_bar 108.62,
38.87, 26.09; BoA's metrics S/S/S_bar 103.83, 55.82, 28.60; sensitivity
S_w/S_u/S_bar 102.78, 40.56, 26.56; S_w/S_u/S 104.03, 41.25, 25.68. The rule
(attnw_mix_rule.py; attnw_mix_dev/rule_result.json) picks the sensitivity arm,
smallest improvement over plain 2.12 percent (Llama-160m); every other arm but
S_w/S_u/S (0.92) loses on at least one model. In TinyLlama and OpenLLaMA-3B most
of the gain comes from the key side; the exact value metric S_bar alone helps
little and loses to plain on OpenLLaMA-3B. Run times: Llama-160m 53 s, TinyLlama
751 s, OpenLLaMA-3B 2,844 s (each split arm fuses twice). Instrument check
(attnw2_check.py; attnw2_check/llama-160m.json): attnw2_common's metrics, with the
query energies computed per head and rotary plane instead of a d x d S_u, and its
per-layer fusion reproduce the development values on Llama-160m (plain identical,
sensitivity 102.7752 against 102.7752, relative 2.4e-8). The check's runner first
waited on the parallel session's waiter, whose command line contains the pattern
both used with pgrep -f, and was started by hand; waiters now key on process IDs.

Quantization-aware value scaling for the fused model, development (exploratory,
labelled, negative; qscale_dev.py, qscale_dev.log, balance_dev/qscale_*.json).
After the fusion and rebalancing, a row scale of q_proj, k_proj or v_proj is
absorbed by per-output-channel rounding, so only the value-row factors act, through
the o_proj columns they divide. Each value row was scaled by its root mean square
o_proj input to the power alpha, alpha chosen per layer from {-0.5, ..., 1} to
minimize the o_proj rounding error weighted by its input second moment (AWQ's scale
search, Lin et al. 2024, applied to the fused value/output pair). Perplexity,
rebalanced against scaled: Llama-160m 4 bits RTN 137.7 / 134.8, GPTQ 119.8 / 121.0;
3 bits RTN 186.7 / 177.4, GPTQ 143.5 / 143.2. TinyLlama 4 bits RTN 67.6 / 65.0,
GPTQ 67.2 / 66.7; 3 bits RTN 251.3 / 150.0, GPTQ 80.7 / 81.9. The scaling helps
round-to-nearest only; GPTQ already absorbs what it buys. The search picks alpha 0
to -0.5, the opposite sign to AWQ: the fused value rows with high activation energy
already carry their scale in the whitened basis, so shrinking their o_proj columns
is what equalizes o_proj's rows. Not pursued.

Writer heads kept in high precision under 3-bit RTN, development (exploratory,
labelled, negative; writer_guard_dev.py, writer_guard_dev.json/.log; Llama-2-7B,
8 WikiText-2 windows, unquantized 5.027). Every decoder linear layer at 3 bits per
output channel; kept entries excluded from their row's range and left in bfloat16.
The weight-only writer statistic of round 29 again gives KV heads 8, 11 and 25
(14.2, 11.1, 7.5 times the median) in layer 1. Perplexity: nothing kept 399.2;
super weight down_proj[2533, 7890] kept (Yu et al. 2024) 389.3; writer heads'
o_proj columns 410.1; writers and super weight 401.0; three random head triples
409.8, 414.7, 406.7, with the super weight 400.2, 406.4, 394.4. Neither the writers
nor the super weight alone rescues per-channel 3-bit RTN, and the writers are no
better than random heads; the 3-bit failure is diffuse, as the earlier probe's
layer ablations suggested (llama2_quant_sink_probe.json). Not pursued.

Attention-derived Hessians for GPTQ, development (exploratory, labelled, not
pursued; attnw_gptq_dev.py, attnw_gptq_dev/, attnw_gptq_dev.log; 32 C4 documents
of 1,024 tokens). The unconverted model's attention projections quantized per
output row by GPTQ (quant_balance_dev.gptq; Hessians from the unquantized model),
with the input Hessians of q_proj / k_proj / v_proj as plain S/S/S, BoA's
S/S/S_bar, sensitivity S_u/S_w/S_bar, and S_u/S_w/S (o_proj always its own input's
second moment). Llama-160m (20 windows, unquantized 66.99), 4 bits: 68.50, 69.50,
69.09, 68.18; 3 bits: 88.18, 73.73, 73.69, 88.92. TinyLlama (8 windows, 7.24), 4
bits: 7.78, 7.50, 7.52, 7.80; 3 bits: 10.76, 8.63, 8.78, 11.20. The value Hessian
S_bar (Kim et al.'s, ICML 2025) removes most of the 3-bit cost in both models, a
replication of their result in plain GPTQ; the key and query sensitivity weights
add nothing to it and alone are slightly worse than plain. For quantization the
gain sits on the value side; for the fusion (round 32's development) it sits on the
key side. Not pursued as a quantization method: the part that works is BoA's.

Conversion command extended (2026-10-31; kvfuse.py, gqa_export.py): matched
groups of four (quad_groups: the minimum-cost pairs, then the minimum-cost
matching of those pairs by the four-head residual, group_cost, which
generalizes pair_costs to any group size) and projections with biases (the
Qwen2 layout: pair costs on the second moment of [x, 1], as fuse_kv builds it;
the export keeps q, k and v biases with their rows). Unit check on two-layer
random models (kvfuse_tiny_check.py, llama2_dev/kvfuse_tiny_check.json):
group_cost equals pair_costs on every pair to 5.6e-17; quads and pairs export
and reload with logits within 2.4e-7 of the simulated conversion (scale 0.7);
the Qwen2 model with biases within 1.8e-7. Real-model checks queued after the
GPU chain: kvfuse_bias_check.py (Qwen2.5-0.5B) and kvfuse_quad_check.py
(Llama-160m, matched against neighbouring quads).

Fitting the fusion in the metrics the attention computation implies
(preregistration 32, frozen at bcdcd9d and pushed before the run). Ran 12:52 to
19:00 UTC on AC power, one GPU job under both guards, no pauses or kills
(attnw2_round.py; attnw2.log): SmolLM2-1.7B 736 s, Llama-3.2-1B 446 s,
Mistral-7B-v0.1 4,081 s, OLMo-1B 466 s, h2o-danube-1.8b 837 s,
Mistral-7B-Instruct 4,052 s, Llama-2-7B 6,242 s (whole test set twice),
Qwen2.5-7B 5,177 s; 22,037 s in all. Evaluation (eval_attnw2.py, unchanged since
the freeze): gate passes, 2 of 4 (attnw2_results.json, eval_attnw2.log). The
plain arms reproduce round 31's values to four decimals in all five shared
models, Qwen2.5-7B's the corrected 23.873 (attnw2_common.metrics carries the
bias column). The 7B models' metrics pass takes 1,100 to 2,300 s and the split
fusion, two closed-form passes per group, 2 to 5 times the plain fusion's time;
Qwen2.5-7B's sensitivity arm ran 3,539 s against an estimate of 2,300, which is
why its log was quiet for an hour.

Round 32's Llama-2-7B diagnostic (attnw2_llama2_diag.py, post hoc; attnw2_diag/,
attnw2_diag.log) ran 19:01 to 19:53 UTC and was killed by the memory watchdog at
its third variant: swap had grown 4,111 MB over the baseline the watchdog took at
14:30 while free memory stood at 29 percent, and the watchdog had been restarted
at 14:30 with a 4,000 MB swap-growth limit instead of the usual 8,000 (round 32
held the same per-layer metrics under the usual limit without trouble). The two
finished variants are kept: the plain fusion everywhere reproduces 16.000 with
the one-window check at 0.98, and the sensitivity fit in layers 0 and 1 only
gives 16.307 with the check at 1.01, so the early layers carry a small part of
the 16.0 to 18.4 loss and the super-weight check sees nothing. The watchdog is
back at the 8,000 MB limit with a fresh baseline; the script resumes from its
saved variants and the two missing ones are queued at the end of the gap chain.


Uptraining at 5M tokens, development (labelled; uptrain_round.py --dev on
Llama-160m, learning rate 1e-4, warmup 10 steps, 16 sequences of 1,024 tokens a
step, 305 steps per arm; uptrain_dev/llama-160m_5M.json, uptrain_dev_5M.log; ran
19:54 to 21:50 UTC; the battery guard paused it from 20:27 to 20:38 UTC while the
laptop was off AC power at 20 percent, a stop and continue that leaves the
computation unchanged; no kills). Arms: the unconverted model
trained the same way, averaging, the fusion rebalanced, round 32's fusion
rebalanced (all at half the key-value heads, 12 to 6); WikiText-2 perplexity on
40 windows of 1,024 tokens. Final perplexity base 23.76 (from 23.18), averaging 51.61 (from 727.2), fusion rebalanced 30.78 (from 50.63), round 32 fusion rebalanced 30.36 (from 47.35); zero-shot averages after training base 0.410, mean 0.359, fuse 0.383, fuse_sens 0.370; the fusion's lead over averaging in log perplexity 2.66 at the start and 0.52 at the end, retained 0.194; checkpoints at 0, 0.5, 1, 2, 3, 4 and 5M tokens: averaging 727.2, 163.1, 92.0, 64.4, 55.2, 53.5, 51.6; fusion 50.6, 33.6, 32.4, 31.2, 30.8, 30.9, 30.8; round 32 fusion 47.4, 33.1, 31.9, 30.9, 30.5, 30.5, 30.4. Averaging catches up
most of the way, from 14 times the fused start to 1.7 times after 5M tokens, and
is still falling where the fused curves are flat from 3M tokens on; the round 32
start stays below the plain fused start at every checkpoint, by 1 to 3 percent.
The rule committed at e2c9852 (uptrain_rule.py, uptrain_dev/rule_result.json)
gives F = 0.1 for round 33's U3 (retained lead 0.194, rounded down to a tenth).

Bias-path check and benchmark of the conversion command (2026-10-31, 21:50 to
22:00 UTC, the gap chain under the guards). kvfuse_bias_check.py on Qwen2.5-0.5B
(CPU, float32; llama2_dev/kvfuse_bias_check_Qwen2.5-0.5B.json): the pair costs on
the moment of [x, 1] are finite, the command's pairs equal the check's, the saved
checkpoint keeps the key and value biases, loads as a 1-KV-head Qwen2 model and
gives the simulated conversion's logits exactly (difference 0.0 of scale 22.5).
kvfuse_bench.py on TinyLlama (4 to 2 KV heads, bfloat16, MPS, batch 1, a
2,048-token prefill then 128 decode steps, median of 3; kvfuse_bench/
TinyLlama_v1.1.json): cache 47 to 23 MiB at 2,176 tokens (ratio 0.500, by
construction), decode 78.0 to 84.3 tokens per second (1.08 times), prefill 1.0 s
both. One laptop's numbers, not a serving benchmark; a 1.1B model's cache is
not its bottleneck at batch 1.

Checks of the conversion command's extensions on Llama-160m (2026-10-31, 22:00
to 22:15 UTC, CPU, float32, in the gap chain). kvfuse_check.py (llama2_dev/
kvfuse_check_llama-160m_balanced.json): after the extension, the command's pair
costs still equal the frozen round 28 instrument's (largest difference 0.0),
its pairs equal round 28's in every layer, and the saved checkpoint reproduces
the simulated conversion to 3.5e-4 in logits of 22.7. kvfuse_quad_check.py
(llama2_dev/kvfuse_quad_check_llama-160m.json): group_cost on every pair equals
pair_costs' entry to 2.5e-16; the two-level matching's groups of four (12 to 3
KV heads) give WikiText-2 perplexity 265.6 on 8 windows of 2,048 tokens against
339.7 for neighbouring groups of four (unconverted 67.8), the command's
--group 4 --pairing matched groups equal the check's, and its checkpoint
reproduces the simulation to 1.7e-4. Groups of four cost this small model 3.9
times the unconverted perplexity against 1.55 for pairs (round 26's protocol
differs; the ratio is indicative).

Single-weighting ablation of the fusion, development (labelled; 2026-10-31 22:00
to 2026-11-01 01:25 UTC in the gap chain; fuse_ablation_dev.py, which runs round
26's instrument with two added arms, perplexity only; fuse_ablation_dev/*.json,
summary.json; round 26's base, unweighted and both-weighted rows copied). On
round 24's simulated-merge groups, WikiText-2 perplexity unweighted / queries
only / outputs only / both: g = 2, OPT-1.3B 29.59 / 29.63 / 28.84 / 29.13,
Pythia-1.4B 29.95 / 26.93 / 26.97 / 27.30, GPT-2 large 31.48 / 29.84 / 31.82 /
30.41, OLMo-1B 169.14 / 140.56 / 106.00 / 87.36, Phi-2 23.32 / 19.29 / 23.00 /
18.92; g = 4, OPT-1.3B 105.43 / 92.89 / 105.27 / 93.66, Pythia-1.4B 116.37 /
79.73 / 101.12 / 71.21, GPT-2 large 177.95 / 154.72 / 134.22 / 116.79, OLMo-1B
989.59 / 911.33 / 648.40 / 601.45, Phi-2 83.56 / 63.23 / 88.66 / 65.73. The query
weighting alone is below the unweighted fit in 9 of 10 cells, the output
weighting alone in 8 of 10, and both together are lowest in 5 of 10: by a wide
margin where the conversion is hard (OLMo-1B at both sizes, Pythia-1.4B and
GPT-2 large at g = 4) and within 0.8 to 3.8 percent of the better single
weighting where it is easy (OPT-1.3B at both sizes, Pythia-1.4B and GPT-2 large
at g = 2, Phi-2 at g = 4). Neither weighting is the whole effect and the two add
where it matters; the query weighting is the more reliable one on its own.

Round 32's Llama-2-7B diagnostic, resumed (post hoc; attnw2_llama2_diag.py from
its saved variants; 01:26 to 02:48 UTC in the gap chain, 4,858 s, under the
memory, battery and disk guards, no kills; attnw2_diag/Llama-2-7b-hf.json,
attnw2_diag.log). The four variants, WikiText-2 perplexity on 8 windows of
2,048 tokens and round 29's one-window check (largest ratio over layers 0 to
5): the plain fusion everywhere 16.000 (0.98); the sensitivity fit in layers 0
and 1 only 16.307 (1.01); in layers 2 and up only 17.868 (1.01); everywhere
18.373 (1.06), which reproduces round 32's registered value. In log perplexity
the early layers add 0.019 and the deeper layers 0.110 of the 0.138 total; the
check never approaches its threshold of 2.35. The loss is spread through the
deeper layers, and the super-weight mechanism of round 28's follow-up does not
explain it.

Llama-2-7B at a quarter of the key-value heads, development (labelled;
quad_dev.py, 02:50 to 06:58 UTC under the three guards; quad_dev/
Llama-2-7b-hf.json, quad_dev.log; round 28's calibration, whole-test WikiText-2
perplexity at 2,048 tokens, ZS-7 on 500 examples per task, round 29's one-window
check over layers 0 to 5; bfloat16; the model reloaded between arms). Unconverted
5.470, ZS-7 0.636, check 1.00. Groups of four by the two-level matching
(summed four-head residual 9.61 against 10.31 for neighbouring groups): 14,918,
ZS-7 0.405, check 576 at layer 1 (then 466, 314, 297, 290 through layer 5);
neighbouring groups of four: 18,953, ZS-7 0.402, check 543; averaging over
neighbouring groups: 4,066, ZS-7 0.363, check 0.93. Both fused groupings
collapse through layer 1 although heads 8 and 18 of that layer are in different
groups ([1, 8, 15, 20] and [7, 11, 18, 26] in the matching), so a four-head
merge switches the layer on by another combination; averaging collapses without
it, as at half the heads in round 28. The check flagged both fused collapses,
its first severe cases outside the development population it was set on, and
read 0.93 on the one that fails the other way. Run times: moments 855 s, groups
435 s, each conversion 644 to 678 s, each evaluation about 3,050 s; restore
check 5.027 on 8 windows. The repair at the four-head level (quad_repair_dev.py)
is queued after Mistral-7B-v0.1's arms.

Mistral-7B-v0.1 at half and a quarter of the key-value heads, development
(labelled; quad_dev.py, 06:59 to 13:35 UTC under the three guards; paused by the
battery guard from 11:44 to 12:59 UTC while the laptop was off AC power, a stop
and continue; quad_dev/Mistral-7B-v0.1.json, quad_dev.log; the protocol of the
Llama-2-7B entry). Unconverted 5.254, ZS-7 0.681, check 1.00. Half the heads
(8 to 4) with pairs by the residual: 24.581, ZS-7 0.481, check 1.02 (round 32's
8-window value 28.68). A quarter (8 to 2) with the two-level matching (summed
residual 10.63 against 11.25 neighbouring): 146.4, ZS-7 0.387, check 1.11;
neighbouring groups of four 176.1, ZS-7 0.385, check 2.12; averaging 812.9,
ZS-7 0.355, check 1.45. No collapse and no spike on this model: the fusion at a
quarter loses 28 times the unconverted perplexity where averaging loses 155,
and the hazard that collapses Llama-2-7B at the same ratio does not appear.
Run times: moments 902 s, groups 100 s, conversions 599 to 630 s, evaluations
3,180 to 3,210 s (the last one 7,974 s including the pause); restore check 4.859
on 8 windows.

Repair of the quarter-heads conversion of Llama-2-7B, development (labelled;
quad_repair_dev.py, 13:36 to 15:10 UTC under the three guards, no pauses;
quad_dev/Llama-2-7b-hf_repair.json, quad_repair.log). With layers 0 to 5
converted by the matched groups of four, the one-window check reproduces the
full conversion's reading at layer 1 (576). Leaving each group of layer 1
unmerged in turn: the check falls to 0.71 without [1, 8, 15, 20] and to 0.83
without [7, 11, 18, 26], and stays at 426 to 586 without any other group;
[1, 8, 15, 20] merged alone reads 0.83. The event needs both merges, head 8's
group and the group of heads 11 and 18, as round 28's follow-up found for the
pair 8-18 with head 11 as the counterweight. Round 29's repair at the four-head
level fails three times: forbidding the group holding head 8 re-matches it with
other heads ([1, 8, 14, 24], then [1, 3, 8, 28], then [1, 8, 9, 17]) and the
check reads 491, 422 and 386. The fallback, layer 1 at its matched pairs (the
groups that gave 16.78 at half the heads) with every other layer quartered,
reads 360 on the check and the full conversion gives 5,344 (ZS-7 0.413): once
layer 0 is quartered, layer 1's safe pairs fire too. Every reading above the
threshold went with a super-weight collapse and every reading below with its
absence; the check does not see collapses of the other kind (averaging at
4,066 reads 0.93, Mistral's fused quarterings at 146 and 176 read 1.11 and
2.12). A
sweep of how many early layers must stay at pairs (quad_fallback_dev.py) runs
next.

A per-layer ratio for Llama-2-7B chosen by the one-window check, development
(labelled; quad_fallback_dev.py, 15:10 to 16:44 UTC under the three guards, no
pauses; quad_dev/Llama-2-7b-hf_fallback.json, quad_fallback.log). With layers 0
to 5 converted, the first k layers at the matched pairs and the rest at the
matched groups of four: k = 0 reproduces the full quartering's check (576 at
layer 1); k = 1 reads at most 0.98, and k = 2 to 6 at most 0.98 as well. Layer
0's quartering is what makes layer 1 fire; with layer 0 at half its heads, layer
1 takes its groups of four without the spike. The smallest passing k, 1, was
converted in full (layer 0 at 16 key-value heads, layers 1 to 31 at 8; cache
fraction 0.258 against 0.25 for the full quartering): whole-test perplexity
105.5 at 2,048 tokens, ZS-7 0.408 (unconverted 5.47 and 0.636; half the heads
16.78 and 0.504), the check 0.98; the full quartering gave 14,918 and averaging
4,066 (ZS-7 0.363). Run times: moments 867 s, each sweep point about 130 s,
conversion 686 s, evaluation about 3,100 s. A checkpoint with one layer at a
different ratio is not expressible in a Hugging Face config, which holds one
num_key_value_heads; exporting it needs a per-layer head count in the loader.

Calibration size, development (labelled; calib_sens_dev.py, 16:44 to 17:20 UTC
in chain 2; calib_dev/OLMo-1B-hf.json, calib_dev/h2o-danube-1.8b-base.json,
plan_chain2.log). The fusion at half the key-value heads from the first 128,
64, 32 and 16 C4 validation documents of 2,048 tokens, perplexity on 8
WikiText-2 windows with the pairs matched on the same moment and with the
128-document pairs. OLMo-1B (unconverted 12.02): 54.70, 54.06 (the same pairs in
all 16 layers), 56.60 with its own pairs and 57.98 with the 128-document pairs
(12 of 16 layers agree), 57.87 and 59.50 (11 of 16). h2o-danube-1.8b (6.76):
38.82, 34.43 and 36.92 (21 of 24), 36.50 and 39.14 (21 of 24), 40.37 and 39.95
(23 of 24). The 128-document values reproduce rounds 30 to 32's to three
decimals. Sixty-four documents (131,072 tokens) are as good as 128 on both
models; sixteen cost 4 to 6 percent; on h2o-danube the values move by about 10
percent either way between sizes, which is the variance of the calibration
sample itself, so an absolute fused perplexity at this scale carries that
uncertainty while the paired comparisons of rounds 30 to 32, which share the
calibration between arms, do not.

Qwen2.5-7B with pairs matched by the residual, development (labelled;
qwen_pairs_dev.py, 17:16 to 18:01 UTC in chain 2 under the three guards, no
pauses; qwen_pairs_dev.json, plan_chain2.log). Rounds 27 to 32 converted
Qwen2.5-7B (4 key-value heads a layer) with neighbouring pairs because the pair
cost did not accept biased projections; it does now (the ones column of the
bias fix). Under the 128-document moment the matching lowers the summed
residual over the 28 layers from 4.718 to 4.362 and differs from the
neighbouring pairs in 15 of 28 layers (each layer has three pairings of its
four heads); perplexity on the 8 WikiText-2 windows is 70.10 with the matched
pairs against 23.87 with the neighbouring pairs (unconverted 5.40; the
neighbouring arm reproduces round 30's 23.873). On this model the residual
chooses the worse pairing, as in round 27's K2, where its weighting lost to the
unweighted fit in the two Qwen2.5 cells that merge every key-value head into
one. Which of the 15 layers carry the difference was not measured. The
neighbouring pairs remain the reported setting for Qwen2.5-7B; the paper
states the exception.

The fusion's residual against round 29's merge damage, development (labelled;
residual_damage_dev.py, 18:01 to 18:51 UTC in chain 2 under the three guards,
no pauses; residual_damage/Mistral-7B-Instruct-v0.2.json, open_llama_3b_v2.json,
OLMo-1B-hf.json; plan_chain2.log). For each of round 29's forced-pair
conversions (one of layers 0 to 4 forced to contain a given pair, every other
layer at the fusion's matching) the pair costs of those layers were recomputed
from the same calibration and each member given its matching's cost increase
over the reference matching and its forced pair's own cost, against its damage
(round 29's perplexity on 4 windows of 2,048 tokens over the reference
conversion's). Pooled Spearman with damage: Mistral-7B-Instruct, 98 members,
cost increase -0.21, forced pair's cost -0.12, one-window check -0.14;
OpenLLaMA-3B, 117 members, -0.14, 0.39, -0.16; OLMo-1B, 117 members, 0.16,
0.18, 0.15. The populations hold almost no damage: relative perplexity has sd
0.014 across Mistral's layers 1 to 4 (0.98 to 1.04), at most 0.005 in
OpenLLaMA's layers 2 to 4 and 0.02 to 0.05 in OLMo's five layers; only
Mistral's layer 0 spans more (0.87 to 1.39), and there the cost increase
correlates at -0.38 over 20 pairs. OpenLLaMA's pooled 0.39 is a between-layer
artefact (the forced pair's cost rises with the layer at 0.73 and damage at
0.51; within layers -0.12 to 0.27). In Mistral and OpenLLaMA several forced
pairs in layer 0 give 10 to 16 percent lower perplexity than the residual's own
matching. The script failed on Qwen2.5-7B, whose members are labelled by their
whole matching (four heads) rather than a forced pair, before reaching the
Llama-2-7B development population; the parser now reads both labels, and the
two populations rerun after chain 2 (run_a10_rerun.sh, residual_damage_rerun.log),
before round 33's pipeline. Read: this population cannot test whether the
residual orders merge damage; a within-layer test over every matching of one
layer with KL as ground truth is drafted (residual_select_dev.py, unchecked;
PLAN_kvfusion_gaps.md, A10b).

Guards, 2026-11-01 evening (operational). The whole-test script (fulltest_dev.py)
ran under paging from about 18:50 UTC: swap 6.8 of 8 GB, then 9.5 of 10 GB, the
7B model's 16 to 18 GB footprint beside the author's applications, the process
inside MPS linear ops with the GPU at 98 percent; Mistral-7B-v0.1 took 64
minutes and Mistral-7B-Instruct 73 (the script prints cumulative seconds; a
first reading of this entry said 137). The battery guard paused the job at 20
percent (20:45 UTC; the adapter had reported "not charging" since 19:48) and
resumed it at 30 percent (20:53 UTC) once power was back. At 21:20 UTC the
memory watchdog, a day-old instance with its swap baseline at 3,549 MB against
9,490 MB in use (growth 5,941 MB, free memory 39 percent and falling toward the
30 percent half of its kill rule), was restarted with a fresh baseline (9,563
MB; watchdog.log), as CLAUDE.md prescribes before a long chain; the 15 percent
floor is unchanged.

Whole-test perplexity at 2,048 tokens for the 7B entries, development
(labelled; fulltest_dev.py, 2026-11-01 18:46 to 22:08 UTC in
chain 2 under the three guards, one battery pause of 8 minutes, the run under
paging throughout; fulltest_dev/Mistral-7B-v0.1.json,
Mistral-7B-Instruct-v0.2.json, Qwen2.5-7B.json; plan_chain2.log). The plain
fusion at half the key-value heads (round 28's calibration; the residual's
matching for the Mistral models, neighbouring pairs for Qwen2.5-7B as in the
registered rounds) on the whole WikiText-2 test set beside the registered 8
windows: Mistral-7B-v0.1 24.581 against 28.683 (unconverted 5.254 against
4.859; 163 windows; 64 minutes), Mistral-7B-Instruct 14.507 against 16.679
(5.950 against 5.540; 163; 73 minutes), Qwen2.5-7B 27.835 against 23.873
(6.850 against 5.396; 146; 64 minutes). The 8 windows are harder than the test
set's average for the Mistral models and easier for Qwen; in all three the
fused-to-unconverted ratio is lower on the whole test than on the 8 windows
(4.68 against 5.90, 2.44 against 3.01, 4.06 against 4.42), so the registered
8-window ratios overstate the relative damage by a quarter, a quarter and a
tenth. The whole-test values are the ones to quote beside CAGA's and Llama-2-7B's
(16.78, the same protocol).

Retrieval of a planted fact, development (labelled; needle_dev.py, 22:08 to
22:42 UTC in chain 2 under the three guards, no pauses;
needle_dev/Mistral-7B-Instruct-v0.2.json, plan_chain2.log). Five codes at
three depths in a 2,048-token C4 haystack, greedy decoding of 24 tokens.
Unconverted: 15 of 15 (133 s). The plain fusion at half the key-value heads
(round 28's calibration, the residual's pairs): 2 of 15 (0, 1 and 1 by depth),
every answer in the right form with the wrong digits (1111, 10110101, 2736 for
27365). The same conversion's perplexity is 16.68 on 8 windows and 14.51 on the
whole test against 5.54 and 5.95: the loss of copying from context is far
larger than the perplexity ratio suggests. The repository's retrieval-head
scores for this model (audits/retrieval_heads, Wu et al.'s protocol) give 35
query heads at 0.1 or more and 15 at 0.2, on 24 and 9 of the 256 key-value
heads. A repair is drafted, retrieval_protect_dev.py (helpers checked on a
synthetic layer; not yet run on a model): those key-value heads stay unmerged
and the rest are matched by the residual, against the same count protected at
random and by the fusion's cheapest-pairing cost; per arm the 15 instances,
8-window perplexity and the cache fraction.

Queue, 2026-11-01 22:48 UTC (operational). On the author's decision the
retrieval-protection development run (A11, retrieval_protect_dev.py) is
inserted before round 33: after the A10 rerun come round 34's development test
and the Llama-2-7B export and benchmark (run_pr33a.sh), then A11 (run_a11.sh),
then round 33 and its evaluator (run_pr33b.sh); one GPU job at a time under the
three guards.

Where the residual pairs the retrieval heads, CPU check from committed
artifacts (retrieval_pairs_check.py, retrieval_pairs_Mistral-7B-Instruct-v0.2.json;
2026-11-01 22:52 UTC; a parallel session found the same pattern independently).
With a key-value head's score the largest over the four query heads it serves,
the five heads at 0.3 or more (layer 2 head 5 at 0.44, 12/1 at 0.61, 18/0 at
0.60, 19/2 at 0.33, 30/2 at 0.35) are each paired by round 29's reference
matching (the fusion's residual at half the heads) with a head scoring at most
0.21, four of them at 0.00; at 0.1 or more, 20 of 24 are paired with a head
below the threshold. The residual does not see retrieval function.
retrieval_protect_dev.py now runs four arms by default (protect at 0.1 and at
0.3, random and residual-chosen protection at the 0.1 count), the base arms
on request, after needle_dev's fused arm took about 27 minutes under paging.
