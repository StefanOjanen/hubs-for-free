"""Command line: `hubsfree demo` (the coordinator random matrices produce)
and `hubsfree audit <file.npy>` (run the battery on saved attention maps)."""
import argparse
import sys

import numpy as np

from .battery import percentile_report, run_battery
from .nulls import random_causal_softmax
from .stats import coupling, eigengap, generators, gnorms, two_sigma_flags, ward_sizes


def demo(seed=147, n=14, T=26):
    A = random_causal_softmax(np.random.default_rng(seed), n, T)
    G = generators(A); gn = gnorms(G); C = coupling(G)
    rs = C.sum(1)
    print(f"Random causal softmax matrices, n = {n} heads, T = {T}, seed {seed}. No training.")
    print(f"  2-sigma outliers in generator norm:   {sorted(two_sigma_flags(gn))}")
    print(f"  2-sigma outliers in coupling row sum: {sorted(two_sigma_flags(rs))}")
    print(f"  eigengap lambda_1/|lambda_2|:         {eigengap(C):.1f}")
    print(f"  Ward clustering into 4 groups:        {ward_sizes(C)}")
    print("A 'coordinator dimension', a spectral gap, and a three-singletons-plus-one-cluster")
    print("pattern, from matrices with no structure to find. Run the battery before believing them.")


def audit(path, draws):
    A = np.load(path)
    if A.ndim != 3:
        sys.exit("expected an array of shape (n_heads, T, T)")
    res = run_battery(A.astype(np.float64), draws=draws)
    percentile_report(res)


def main():
    p = argparse.ArgumentParser(prog="hubsfree")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--seed", type=int, default=147)
    a = sub.add_parser("audit"); a.add_argument("path"); a.add_argument("--draws", type=int, default=50)
    args = p.parse_args()
    if args.cmd == "demo":
        demo(args.seed)
    else:
        audit(args.path, args.draws)


if __name__ == "__main__":
    main()
