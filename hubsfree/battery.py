"""Run a statistic against the null battery and report percentiles."""
import numpy as np

from .nulls import random_causal_softmax, surrogate_altsink, surrogate_colfix, surrogate_plain
from .stats import coupling, eigengap, generators, gnorms, rank1_corr, rho, sink_column

BUILTIN_STATS = {
    "rank1_corr": lambda A: rank1_corr(coupling(generators(A)), gnorms(generators(A))),
    "eigengap": lambda A: eigengap(coupling(generators(A))),
    "rho": lambda A: rho(generators(A)),
    "max_gnorm_share": lambda A: float(gnorms(generators(A)).max() / gnorms(generators(A)).sum()),
}


def run_battery(A, stats=None, nulls=("random", "plain", "colfix"), draws=50, seed=0):
    """A: (n, T, T) causal attention maps. stats: dict name -> f(A) -> float
    (defaults to BUILTIN_STATS). Returns {stat: {"real": x, null: [values]}}."""
    rng = np.random.default_rng(seed)
    stats = BUILTIN_STATS if stats is None else stats
    n, T, _ = A.shape
    out = {name: {"real": float(f(A))} for name, f in stats.items()}
    fams = {}
    if "random" in nulls:
        fams["random"] = [random_causal_softmax(rng, n, T) for _ in range(draws)]
    if "plain" in nulls:
        fams["plain"] = list(surrogate_plain(A, rng, draws))
    if "colfix" in nulls:
        fams["colfix"] = list(surrogate_colfix(A, rng, cols=(sink_column(A),), draws=draws))
    if "altsink" in nulls:
        fams["altsink"] = list(surrogate_altsink(A, rng, draws))
    for fam, mats in fams.items():
        for name, f in stats.items():
            out[name][fam] = [float(f(M)) for M in mats]
    return out


def percentile_report(res):
    """Percentile of the real statistic within each null distribution and the
    shrinkage of its excess over the null median. Prints and returns rows."""
    rows = []
    for stat, d in res.items():
        for fam, vals in d.items():
            if fam == "real":
                continue
            v = np.array(vals)
            pct = float((v < d["real"]).mean() * 100)
            rows.append({"statistic": stat, "null": fam, "real": d["real"],
                         "null_median": float(np.median(v)), "percentile": pct})
    for r in rows:
        print(f"{r['statistic']:>16} vs {r['null']:<8} real {r['real']:+.3f}  "
              f"null median {r['null_median']:+.3f}  percentile {r['percentile']:5.1f}")
    return rows
