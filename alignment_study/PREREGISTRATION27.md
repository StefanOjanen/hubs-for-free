# Preregistration 27: fusing key-value heads in models that already share them

Date: 2026-10-19. Frozen by commit and pushed before the confirmatory runs.

## Question

Round 26 converted multi-head models to grouped-query attention without training
by fitting each group's shared key and value head in closed form to what the
heads compute, weighting keys by each head's queries and values by its output
projection; it beat averaging and the unweighted fit in every model. Most current
models already share key-value heads between several query heads. Their cache
can be halved again by merging neighbouring key-value heads, so that every query
head attached to either reads one shared head. This round asks whether the
weighted fusion, generalized to key-value heads that serve several query heads,
still beats averaging and the unweighted fit in such models, up to 7B parameters,
and in a 3B multi-head model.

## Instrument

`alignment_study/fuse_kv.py` and `alignment_study/fusekv_round.py`, evaluated by
`alignment_study/eval_fusekv.py`.

Fusion: round 26's (`fuse_common.py`), with each key-value head's value weighting
the sum of O_h^T O_h over the query heads that read it and its key weighting the
sum of their query second moments; each query head absorbs its key-value head's
key transform (on rotary planes a complex scalar per plane) and each output slice
the value transform. Groups of one key-value head reproduce the model. The second
moments of the attention input are kept in float32 and promoted to float64 one
layer at a time, and the weight snapshot is kept on disk: a dry run that held both
in memory with a 7B model exhausted the machine's memory (recorded in the run
log). The rotary planes are processed together in real arithmetic, and the output
products are associated right to left; on the development models this changes
no perplexity in the fourth decimal.

Arms (neighbouring key-value heads merged in groups of g = 2, and g = 4 where the
count allows): `mean` (projections averaged, as in Ainslie et al. 2023), `fuse_u`
(the unweighted fit) and `fuse` (the weighted fusion).

Calibration: the second moment of every layer's attention input on round 24's 8
C4 validation documents of 1,024 tokens. Evaluation: WikiText-2 perplexity over
1,024-token windows and the seven tasks of `zeroshot.py` on the first 1,000
examples of each (OpenBookQA has 500), the same for every model. dlog(a) =
log(perplexity of a / perplexity of the unconverted model).

Models: TinyLlama/TinyLlama_v1.1 (32 query and 4 key-value heads; g = 2, 4),
Qwen/Qwen2.5-1.5B (12 and 2; g = 2), openlm-research/open_llama_3b_v2 (32 and 32,
multi-head; g = 2, 4), mistralai/Mistral-7B-Instruct-v0.2 (32 and 8; g = 2, 4) and
Qwen/Qwen2.5-7B (28 and 4; g = 2, 4): nine cells. float32 for the first two,
bfloat16 for the others, on the M1 Max, one model at a time under a memory
watchdog.

## Development (labeled)

Development runs (`alignment_study/fusekv_dev/`, GPU, the first 500 examples of
each task): Qwen2.5-0.5B (14 query and 2 key-value heads, g = 2) and Llama-160m (12
and 12, multi-head, g = 2 and 4). Perplexity with averaging, the unweighted fit and
the weighted fusion (unconverted 14.65 and 22.57): Qwen2.5-0.5B at g = 2 11,167,
254 and 166; Llama-160m at g = 2 789, 84.7 and 60.8, at g = 4 17,436, 276 and 219.
Every registered direction holds: the weighted fusion is below averaging and below
the unweighted fit in all three cells; at g = 2 its dlog is 0.37 and 0.28 of
averaging's (geometric mean 0.32) and its accuracy is higher by +0.029 and +0.049
(mean +0.039, interval 0.026 to 0.051). Groups of one reproduce both models to
1.2e-4 in any logit. A first development run, before the memory and speed changes
above, gave the same perplexities to the printed precision.

A dry run on mistralai/Mistral-7B-v0.1 (`alignment_study/fusekv_dryrun.py`,
`fusekv_dev_mistral_dryrun.log`; 8 WikiText-2 windows) checked memory and time and
gave, for g = 2, 520 with averaging, 49.7 with the unweighted fit and 40.9 with the
weighted fusion, against 6.95 unconverted (dlog ratio 0.41); groups of one
reproduced the model to 0.31 in logits of up to 28, the rounding of rewritten
bfloat16 weights. Because this dry run measured outcomes, Mistral-7B-v0.1 is a
development model, and the confirmatory 7B model of its family is
Mistral-7B-Instruct-v0.2. The dry run held 15 to 16 GB on the GPU with the system
at least 67 percent free; each 7B conversion took about 10 minutes, mostly one
float64 eigendecomposition per layer. An earlier dry run, which held the float64
second moments of all layers and the weight snapshot in memory, hung the machine
(kernel watchdog panic, 2026-10-19 17:07).

## Registered predictions

Nine cells (model, g); five models at g = 2.
- K1: `fuse` has a lower perplexity than `mean` in at least 8 of 9 cells.
- K2: `fuse` has a lower perplexity than `fuse_u` in at least 8 of 9 cells.
- K3 (size, g = 2): the geometric mean over the five models of
  dlog(`fuse`) / dlog(`mean`) is at most 0.5.
- K4 (accuracy, g = 2): `fuse`'s average accuracy exceeds `mean`'s in at least 4
  of 5 models, and the mean over models of the difference has a paired bootstrap
  95% interval above zero (examples resampled within tasks within models, 2,000
  draws).
- Reported, not scored: every arm's perplexity and accuracy, perplexity as a
  multiple of the unconverted model's, the accuracy differences at g = 4 and
  against `fuse_u`, and the check that groups of one reproduce the model.

## Falsification and reporting

K1 or K3 failing means the weighted fusion does not carry its advantage over
averaging to models that already share key-value heads; K2 failing means the
weighting stops mattering there; K4 failing means the perplexity gains do not
reach the tasks. Every clause is reported whichever way it falls
(`alignment_study/eval_fusekv.py` writes `alignment_study/fusekv_results.json`).
