# Sink-pair candidates from the committed single-pair artifacts (no GPU). pair_start_diag.py found that the pairs of
# Mistral-7B-Instruct's layer 0 whose merge is catastrophic on the WikiText-2 ranking windows and quiet on held-out C4
# windows, (1, 5) and (0, 1), serve the attention sink and fail only on inputs without the beginning-of-sequence token.
# residual_pairs_dev.py measured every pair of layers 0, 1, 2 and a middle layer fused alone on both texts for three
# models (residual_select_dev/<model>_pairs.json on the ranking windows; _pairs_c4.json and _pairs_c4_w8.json on C4),
# so the signature can be read off: a pair whose KL on the ranking windows is at least MIN_KL and at least RATIO times
# its KL on C4. Lists the candidates per model and layer, the heads they involve, and writes
# alignment_study/sink_pair_candidates.json, the input to round 36's sink-pair sets (PREREGISTRATION36_draft.md).
#   python alignment_study/sink_pair_candidates.py [--min-kl=0.05] [--ratio=10]
import glob
import json
import os
import sys

arg = lambda n, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{n}=")), d)
MIN_KL, RATIO = float(arg("min-kl", 0.05)), float(arg("ratio", 10))
OUT = "alignment_study/sink_pair_candidates.json"
res = {"min_kl": MIN_KL, "ratio": RATIO, "models": {}}
for fp in sorted(glob.glob("alignment_study/residual_select_dev/*_pairs.json")):
    m = os.path.basename(fp)[:-len("_pairs.json")]
    c4 = next((f for f in (fp[:-5] + "_c4_w8.json", fp[:-5] + "_c4.json") if os.path.exists(f)), None)
    if not c4:
        continue
    W, C = json.load(open(fp))["per_layer"], json.load(open(c4))["per_layer"]
    entry = {"ranking_windows_file": fp, "c4_file": c4, "layers": {}}
    for l in W:
        if l not in C:
            continue
        rows = []
        for pair, kw in W[l]["pair_kl"].items():
            kc = C[l]["pair_kl"].get(pair)
            if kc is None:
                continue
            rows.append({"pair": pair, "kl_ranking": kw, "kl_c4": kc, "ratio": kw / max(kc, 1e-6), "candidate": kw >= MIN_KL and kw / max(kc, 1e-6) >= RATIO})
        rows.sort(key=lambda r: -r["ratio"])
        cands = [r for r in rows if r["candidate"]]
        heads = sorted({int(h) for r in cands for h in r["pair"].split("-")})
        entry["layers"][l] = {"candidates": cands, "heads": heads, "top_by_ratio": rows[:5]}
    res["models"][m] = entry
    print(f"== {m}")
    for l, e in entry["layers"].items():
        c = ", ".join(f"({r['pair'].replace('-', ', ')}) {r['kl_ranking']:.3f} against {r['kl_c4']:.4f} ({r['ratio']:.0f}x)" for r in e["candidates"]) or "none"
        print(f"   layer {l}: candidates {c}; heads {e['heads']}; largest ratios " + ", ".join(f"{r['pair']} {r['ratio']:.0f}x" for r in e["top_by_ratio"][:3]))
json.dump(res, open(OUT, "w"), indent=1)
print(f"SINK_PAIR_CANDIDATES_DONE {OUT}")
