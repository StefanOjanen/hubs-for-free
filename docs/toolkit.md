# hubsfree: the toolkit

`hubsfree` runs any statistic computed on attention maps against the null
families this repository found necessary, and reports where the real value
sits in each null and how much of the effect each null already produces.
It needs numpy and scipy; the Hugging Face adapter needs the `models`
extra (torch, transformers, datasets).

## Install

```
pip install -e .              # from a clone
pip install -e ".[models]"    # with the model adapter
```

## Three commands

```
hubsfree demo                                   # the coordinator random matrices produce, one second
hubsfree extract --model Qwen/Qwen2.5-0.5B --layer 11 --out maps.npy
hubsfree audit maps.npy --draws 100 --json report.json
```

`extract` saves one layer's attention maps for a text (default text built
in, `--text` to change, `--max-length` for the window) as an
(n_heads, T, T) float64 array. `audit` runs the built-in statistics against
the default families and prints the report.

## Ten lines in Python

```python
import numpy as np, hubsfree
A = np.load("maps.npy")                              # (n_heads, T, T), rows sum to one
def hub_score(A):
    C = hubsfree.coupling(hubsfree.generators(A))
    return float(C.sum(1).max() / C.sum(1).mean())
res = hubsfree.run_battery(A, stats={"hub_score": hub_score}, draws=100)
hubsfree.percentile_report(res)
```

## What the nulls keep

| Family | Keeps | Destroys | Use |
|---|---|---|---|
| `random` | causal (or bidirectional) shape, row-stochasticity, a lognormal spread of head sharpness | everything learned | base rate of the statistic on noise |
| `plain` | every row's sorted values (so each row's sharpness and self-mass) and every generator norm | which column each row attends to | is the statistic more than per-head sharpness? |
| `colfix` | the above plus the sink column set (`sink_columns`; for bidirectional maps the first and last columns) | the remaining placement | is the statistic more than sharpness plus the shared column? |
| `wrapped` | every row's sorted values; each head stays as concentrated as before, on its own target column | the shared column | dissociation control: alignment versus concentration (primary control of preregistration 6) |
| `shift` | each head's exact row geometry, cyclically shifted by a head-specific offset | the shared column | matched-geometry dissociation control (secondary) |
| `altsink` | as `wrapped` with fixed targets t_h = h + 1 and rows above the target plain-permuted | the shared column | the earlier control, kept for continuity; loses power at 32 heads |

All permutation families preserve row sums, row inverse participation
ratios, the diagonal, and every generator norm exactly (tests hold this to
1e-12). `wrapped`, `shift` and `altsink` are defined for causal maps and
skipped for bidirectional ones. Restoring the rank-one geometry with a
dissociation control needs the causal width: at 32 heads use windows of
256 tokens or more (alignment_study/NOTE.md, instrument fixes).

## Built-in statistics

`rank1_corr` (correlation of the coupling matrix with the head-norm
products), `eigengap` (lambda_1 / |lambda_2| of the coupling matrix), `rho`
(top eigenvalue share of the cosine Gram of the generators), `shared_energy`
(mean over heads of the energy on the layer's shared operator),
`max_gnorm_share`, and for causal maps `sink_mass`, `cos_to_sink` (cosine
of the shared operator to the ideal sink-set operator) and
`sink_floor_of_shared_energy` (the part of the shared energy that each
head's own sink profile accounts for; `derived_shared_energy` in the API). Any function
`f(A) -> float` can be added through `stats=`.

## Reading the report

`percentile` is the percentage of null draws below the real value. When
the `random` family is present, `reproduced` is (null median minus random
median) over (real minus random median): the share of the real effect,
measured from the noise baseline, that the constrained null already
produces. Near 1 means the null carries the effect; near 0 means the
effect lies beyond what the null keeps; negative values mean the null moves
the statistic the other way. The audit outcome labels used in this
repository (survives, shrinks, matched) are defined in
`audits/PREREGISTRATION4_DRAFT.md` in terms of these two numbers.

## Base rates and matched-noise floors

`two_sigma_flags` and `ward_sizes` are the rules whose base rates the
synthetic battery (`experiments.py`) tabulates: how often noise produces a
"special" head or the cluster pattern at a given head count. Use
`random_causal_softmax` with your own n and T to compute the base rate of
any outlier rule before reporting a component as special.

## Model adapter

`hubsfree.adapters.load_model(name, dtype="auto", device="auto")` loads a
Hugging Face model with eager attention under the repository's precision
policy (float32 when the weights fit the device, bfloat16 otherwise;
never float16). `attentions(tok, model, text, max_length)` returns the
per-layer (n_heads, T, T) float64 arrays. The measured fidelity of Apple
MPS against the CPU float32 protocol is in
`alignment_study/platform_fidelity.json`.
