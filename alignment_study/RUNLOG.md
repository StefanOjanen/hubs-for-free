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
