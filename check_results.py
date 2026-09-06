# CI regeneration check: compares a freshly generated results.json against
# the committed baseline within tolerances that absorb platform floating
# point differences but not changes in any reported quantity.
import json
import sys

new = json.load(open(sys.argv[1]))
base = json.load(open(sys.argv[2]))
ATOL, RTOL = 0.05, 0.10
IGNORE = {"runtime_seconds", "date", "note"}
failures = []

def walk(a, b, path=""):
    if isinstance(b, dict):
        for k, v in b.items():
            if k in IGNORE: continue
            if k not in a: failures.append(f"{path}/{k}: missing in new"); continue
            walk(a[k], v, f"{path}/{k}")
    elif isinstance(b, list):
        if len(a) != len(b): failures.append(f"{path}: length {len(a)} vs {len(b)}"); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]")
    elif isinstance(b, bool) or b is None or isinstance(b, str):
        if a != b: failures.append(f"{path}: {a!r} vs {b!r}")
    else:
        if abs(a - b) > ATOL + RTOL * abs(b):
            failures.append(f"{path}: {a} vs baseline {b}")

walk(new, base)
if failures:
    print("REGENERATION DRIFT:"); print("\n".join(failures)); sys.exit(1)
print("results.json regenerates within tolerance (atol 0.05, rtol 0.10).")
