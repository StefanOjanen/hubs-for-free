"""Run statistics against the null battery and report percentiles and the
fraction of each effect that a constrained null reproduces."""
import numpy as np

from .nulls import (is_causal, random_causal_softmax, surrogate_altsink, surrogate_colfix,
                    surrogate_plain, surrogate_shift, surrogate_wrapped)
from .stats import (coupling, cos_to_sink, derived_shared_energy, eigengap, generators, gnorms,
                    rank1_corr, rho, shared_mode, sink_column, sink_columns)


def _sink_mass(A):
    n, T, _ = A.shape
    c = sink_column(A)
    rows = np.arange(max(1, c + 1), T)
    return float(A[:, rows, c].mean()) if len(rows) else float("nan")


BUILTIN_STATS = {
    "rank1_corr": lambda A: rank1_corr(coupling(generators(A)), gnorms(generators(A))),
    "eigengap": lambda A: eigengap(coupling(generators(A))),
    "rho": lambda A: rho(generators(A)),
    "shared_energy": lambda A: shared_mode(generators(A))[3],
    "max_gnorm_share": lambda A: float(gnorms(generators(A)).max() / gnorms(generators(A)).sum()),
}
CAUSAL_STATS = {
    "sink_mass": _sink_mass,
    "cos_to_sink": lambda A: cos_to_sink(A),
    "sink_floor_of_shared_energy": lambda A: derived_shared_energy(A)[0],
}
DEFAULT_NULLS = ("random", "plain", "colfix", "wrapped")


def default_columns(A):
    """Columns the column-preserving surrogate keeps: the sink set for causal
    maps (hubsfree.stats.sink_columns), the first and last columns for
    bidirectional maps (BERT's [CLS] and final [SEP] positions)."""
    T = A.shape[-1]
    return tuple(sink_columns(A)) if is_causal(A) else (0, T - 1)


def run_battery(A, stats=None, nulls=DEFAULT_NULLS, draws=50, seed=0, cols=None):
    """A: (n, T, T) row-stochastic attention maps (causal or bidirectional).
    stats: dict name -> f(A) -> float; defaults to the built-ins (plus the
    sink statistics for causal maps). nulls: any of random, plain, colfix,
    wrapped, shift, altsink (the last three are causal-only and skipped
    otherwise). cols: columns kept by colfix (default: default_columns).
    Returns {stat: {"real": x, "columns": cols, null: [values]}}."""
    rng = np.random.default_rng(seed)
    causal = is_causal(A)
    if stats is None:
        stats = dict(BUILTIN_STATS, **(CAUSAL_STATS if causal else {}))
    n, T, _ = A.shape
    cols = tuple(default_columns(A) if cols is None else cols)
    out = {name: {"real": float(f(A))} for name, f in stats.items()}
    fams = {}
    if "random" in nulls:
        fams["random"] = [random_causal_softmax(rng, n, T, causal=causal) for _ in range(draws)]
    if "plain" in nulls:
        fams["plain"] = list(surrogate_plain(A, rng, draws))
    if "colfix" in nulls:
        fams["colfix"] = list(surrogate_colfix(A, rng, cols=cols, draws=draws))
    if causal:
        if "wrapped" in nulls:
            fams["wrapped"] = list(surrogate_wrapped(A, rng, draws))
        if "shift" in nulls:
            fams["shift"] = list(surrogate_shift(A, rng, draws))
        if "altsink" in nulls:
            fams["altsink"] = list(surrogate_altsink(A, rng, draws))
    for fam, mats in fams.items():
        for name, f in stats.items():
            out[name][fam] = [float(f(M)) for M in mats]
    for name in out:
        out[name]["columns"] = list(cols)
    return out


def percentile_report(res, quiet=False):
    """Per statistic and null: percentile of the real value in the null
    distribution and, when a random-softmax family is present, the fraction
    of the real excess over the random null that the constrained null
    reproduces (median of null minus median of random, over real minus
    median of random). A fraction near 1 means the null already produces
    the effect; near 0 means the effect is beyond what the null keeps."""
    rows = []
    for stat, d in res.items():
        base = float(np.median(d["random"])) if "random" in d else None
        for fam, vals in d.items():
            if fam in ("real", "columns"):
                continue
            v = np.asarray(vals, dtype=float)
            pct = float((v < d["real"]).mean() * 100)
            row = {"statistic": stat, "null": fam, "real": d["real"], "null_median": float(np.median(v)),
                   "percentile": pct, "reproduced": None}
            if base is not None and fam != "random" and abs(d["real"] - base) > 1e-9:
                row["reproduced"] = float((row["null_median"] - base) / (d["real"] - base))
            rows.append(row)
    if not quiet:
        for r in rows:
            rep = "" if r["reproduced"] is None else f"  reproduced {r['reproduced']:+.2f}"
            print(f"{r['statistic']:>16} vs {r['null']:<8} real {r['real']:+.3f}  "
                  f"null median {r['null_median']:+.3f}  percentile {r['percentile']:5.1f}{rep}")
    return rows
