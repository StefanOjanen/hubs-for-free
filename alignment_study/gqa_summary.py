# Tables for PREREGISTRATION24 result files: WikiText-2 perplexity and average zero-shot accuracy per
# arm, and paired bootstrap intervals (examples resampled within each task) for accuracy differences.
#   python alignment_study/gqa_summary.py alignment_study/gqa_dev
import glob
import json
import sys
import numpy as np

sys.path.insert(0, "alignment_study")
from zeroshot import AVG_METRIC

ARMS = ("contiguous", "random0", "wse", "act", "sim", "contiguous_align", "random0_align", "jin", "sim_align")
ALIGNED = {"contiguous_align", "random0_align", "jin", "sim_align"}
PUBLISHED = ("contiguous", "wse", "act", "jin")


def correct(ev, task):
    key = "correct_norm" if AVG_METRIC[task] == "acc_norm" else "correct"
    return np.frombuffer(ev[task][key].encode(), dtype=np.uint8) - 48


def avg_diff_ci(ea, eb, B=2000, seed=0):
    """Paired bootstrap of the difference in average accuracy (a - b), examples resampled per task."""
    rng = np.random.default_rng(seed)
    diffs = {t: correct(ea, t).astype(float) - correct(eb, t) for t in AVG_METRIC}
    point = float(np.mean([d.mean() for d in diffs.values()]))
    boots = np.zeros(B)
    for b in range(B):
        boots[b] = np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs.values()])
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def table(path):
    r = json.load(open(path))
    ev = r["evals"]; base = ev["base"]
    print(f"\n{r['model']}  base ppl {base['wikitext2_ppl']:.2f}  avg {base['average']:.4f}  "
          f"(align diff {r.get('alignment_max_logit_diff', float('nan')):.1e}, restore ppl {r.get('restore_check_ppl', float('nan')):.2f})")
    for g in (2, 4):
        rows = [(a, ev.get(f"g{g}/{a}")) for a in ARMS if f"g{g}/{a}" in ev]
        if not rows:
            continue
        print(f"  g={g}:  " + "  ".join(f"{a} {e['wikitext2_ppl']:.1f}/{e['average']:.3f}" for a, e in rows))
        if f"g{g}/sim" in ev:
            for ref in PUBLISHED:
                if f"g{g}/{ref}" in ev:
                    p, lo, hi = avg_diff_ci(ev[f"g{g}/sim"], ev[f"g{g}/{ref}"])
                    print(f"     sim - {ref}: acc {p:+.4f} [{lo:+.4f}, {hi:+.4f}]  ppl ratio {ev[f'g{g}/sim']['wikitext2_ppl'] / ev[f'g{g}/{ref}']['wikitext2_ppl']:.3f}")
    return r


if __name__ == "__main__":
    for p in sorted(glob.glob(f"{sys.argv[1]}/*.json")) + sorted(glob.glob(f"{sys.argv[1]}/*.json.partial")):
        table(p)
