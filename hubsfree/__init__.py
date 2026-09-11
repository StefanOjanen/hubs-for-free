"""hubsfree: null models and constrained surrogates for attention-structure claims.

Given attention maps A of shape (n_heads, T, T) (causal, row-stochastic),
the package computes the statistics that structure claims are built on and
their distributions under three families of nulls:

- random causal softmax matrices (no learned structure at all),
- per-row marginal-matched permutations (keep each row's sharpness and
  self-mass, destroy cross-head alignment),
- column-preserving permutations (additionally keep the shared sink column
  set),
- dissociation controls (every head stays as concentrated, heads no longer
  share a column).

A statistic that looks the same under these nulls is measuring the
container, not the model. See paper.md in the source repository.
"""
from .stats import (generators, gnorms, coupling, rank1_corr, eigengap, rho,
                    shared_mode, sink_column, sink_columns, column_masses, sink_generator,
                    sink_set_generator, cos_to_sink, derived_shared_energy, two_sigma_flags, ward_sizes)
from .nulls import (random_causal_softmax, surrogate_plain, surrogate_colfix,
                    surrogate_altsink, surrogate_shift, surrogate_wrapped, is_causal)
from .battery import run_battery, percentile_report

__version__ = "0.1.0.dev0"
__all__ = ["generators", "gnorms", "coupling", "rank1_corr", "eigengap", "rho",
           "shared_mode", "sink_column", "sink_columns", "column_masses", "sink_generator", "two_sigma_flags",
           "ward_sizes", "random_causal_softmax", "surrogate_plain",
           "surrogate_colfix", "surrogate_altsink", "surrogate_shift", "surrogate_wrapped",
           "sink_set_generator", "cos_to_sink", "derived_shared_energy", "run_battery",
           "percentile_report"]
