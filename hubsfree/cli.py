"""Command line.
  hubsfree demo                       the coordinator random matrices produce
  hubsfree audit maps.npy [options]   run the battery on saved (n, T, T) maps
  hubsfree extract --model M --layer L [--text ...] --out maps.npy
                                      save one layer's attention maps from a
                                      Hugging Face model (needs the models extra)
"""
import argparse
import json
import sys

import numpy as np

from .battery import DEFAULT_NULLS, percentile_report, run_battery
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


def load_maps(path):
    A = np.load(path)
    if isinstance(A, np.lib.npyio.NpzFile):
        A = A[A.files[0]]
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        sys.exit("expected an array of shape (n_heads, T, T)")
    return A


def audit(path, draws, nulls, cols, out):
    A = load_maps(path)
    cols = None if cols is None else tuple(int(c) for c in cols.split(","))
    res = run_battery(A, nulls=tuple(nulls.split(",")), draws=draws, cols=cols)
    print(f"{path}: {A.shape[0]} heads, T = {A.shape[1]}, columns kept by colfix: {res[next(iter(res))]['columns']}")
    rows = percentile_report(res)
    if out:
        json.dump({"source": path, "draws": draws, "results": res, "report": rows}, open(out, "w"), indent=1)
        print("wrote", out)


def extract(model, layer, text, max_length, out, device):
    from .adapters import attentions, load_model
    tok, m = load_model(model, dtype="auto", device=device)
    mats, ids = attentions(tok, m, text, max_length)
    if layer < 0 or layer >= len(mats):
        sys.exit(f"layer must be in 0..{len(mats) - 1}")
    np.save(out, mats[layer])
    print(f"{model} layer {layer}: saved {mats[layer].shape} maps for {len(ids)} tokens to {out}")


def main():
    p = argparse.ArgumentParser(prog="hubsfree")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--seed", type=int, default=147)
    a = sub.add_parser("audit"); a.add_argument("path"); a.add_argument("--draws", type=int, default=50)
    a.add_argument("--nulls", default=",".join(DEFAULT_NULLS)); a.add_argument("--cols", default=None)
    a.add_argument("--json", dest="out", default=None)
    e = sub.add_parser("extract"); e.add_argument("--model", required=True); e.add_argument("--layer", type=int, required=True)
    e.add_argument("--text", default="The committee reviewed the proposal in detail and asked for a revised budget before the next meeting. " * 8)
    e.add_argument("--max-length", type=int, default=64); e.add_argument("--out", default="maps.npy"); e.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.cmd == "demo":
        demo(args.seed)
    elif args.cmd == "audit":
        audit(args.path, args.draws, args.nulls, args.cols, args.out)
    else:
        extract(args.model, args.layer, args.text, args.max_length, args.out, args.device)


if __name__ == "__main__":
    main()
