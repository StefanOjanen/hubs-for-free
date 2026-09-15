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
