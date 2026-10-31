# Renders every table of the KV-fusion paper from the committed result files (nothing typed in by hand) as
# Markdown, to private/drafts/paper_kvfusion/tables.md (the directory is created; the path can be changed with
# --out=). Tables that depend on results not yet present are skipped with a note.
#   python alignment_study/make_paper_tables.py [--out=path]
import glob
import json
import os
import sys

OUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), "private/drafts/paper_kvfusion/tables.md")
SHORT = {"allenai/OLMo-1B-hf": "OLMo-1B", "gpt2-large": "GPT-2 large", "facebook/opt-1.3b": "OPT-1.3B", "microsoft/phi-2": "Phi-2",
         "EleutherAI/pythia-1.4b": "Pythia-1.4B", "TinyLlama/TinyLlama_v1.1": "TinyLlama", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
         "openlm-research/open_llama_3b_v2": "OpenLLaMA-3B", "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct", "Qwen/Qwen2.5-7B": "Qwen2.5-7B",
         "OLMo-1B-hf": "OLMo-1B", "h2o-danube-1.8b-base": "h2o-danube-1.8b", "open_llama_3b_v2": "OpenLLaMA-3B", "Llama-2-7b-hf": "Llama-2-7B",
         "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct", "Qwen2.5-7B": "Qwen2.5-7B", "SmolLM2-1.7B": "SmolLM2-1.7B", "Llama-3.2-1B": "Llama-3.2-1B",
         "Mistral-7B-v0.1": "Mistral-7B-v0.1"}
short = lambda k: SHORT.get(k, k)
L = []
w = L.append


def table(header, rows):
    w("| " + " | ".join(header) + " |"); w("|" + "---|" * len(header))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w("")


def f(x, d=2):
    return "" if x is None else f"{x:.{d}f}"


# ---- Table 1: five multi-head models, half and a quarter of the KV heads (rounds 24 and 26) ----
T = json.load(open("alignment_study/fuse_results.json"))["table"]
w("## Table 1. Multi-head models converted without training (rounds 24 and 26)\n")
w("WikiText-2 perplexity and seven-task average accuracy. Aligned pooling: Jin et al.'s rotations folded into the weights and mean pooling, on the groups the simulated merge chose (round 24's best of nine recipes). Unweighted fit: the closed form without the query and output weightings. Fusion: this paper. Same groups in the three converted columns.\n")
rows = []
for g in (2, 4):
    for k in sorted(T, key=lambda s: short(s.rsplit(" ", 1)[0])):
        if not k.endswith(f" g{g}"):
            continue
        v = T[k]; m = k.rsplit(" ", 1)[0]
        rows.append([short(m), g, f(v["base"]["ppl"]), f(v["sim_align"]["ppl"]), f(v["fuse_u/sim_align"]["ppl"]), f(v["fuse/sim_align"]["ppl"]),
                     f(v["base"]["acc"], 3), f(v["sim_align"]["acc"], 3), f(v["fuse/sim_align"]["acc"], 3)])
table(["Model", "g", "ppl unconverted", "aligned pooling", "unweighted fit", "fusion", "acc unconverted", "aligned pooling", "fusion"], rows)

# ---- Table 2: Llama-2-7B against CAGA (round 28) ----
R = json.load(open("alignment_study/llama2/Llama-2-7b-hf.json"))["evals"]
w("## Table 2. Llama-2-7B, 32 to 16 KV heads, no training (round 28)\n")
w("WikiText-2 perplexity at 2,048 and 4,096 tokens on the whole test set; ZS-7 as CAGA defines it (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA). CAGA's published row is from its paper; its construction as specified in that paper is run here on the same pairs.\n")
lab = {"base": "unconverted", "fuse/paired": "fusion, pairs by its residual", "fuse/neighbour": "fusion, neighbouring pairs",
       "caga/paired": "CAGA's construction (ours), same pairs", "caga/neighbour": "CAGA's construction (ours), neighbouring pairs", "mean/neighbour": "averaging, neighbouring pairs"}
rows = [[lab[k], f(R[k]["ppl_2048"]), f(R[k]["ppl_4096"]), f(R[k]["zs7"], 3)] for k in lab if k in R]
rows.insert(2, ["CAGA, published (Table 2 of its paper)", "25.1", "25.1", "0.443"])
table(["Conversion", "ppl 2,048", "ppl 4,096", "ZS-7"], rows)

# ---- Table 3: models that already share KV heads (round 27) ----
T = json.load(open("alignment_study/fusekv_results.json"))["table"]
w("## Table 3. Models that already share key-value heads, converted to fewer (round 27)\n")
w("WikiText-2 perplexity and seven-task accuracy; g is the number of KV heads merged into one.\n")
rows = [[short(k.rsplit(" ", 1)[0]), k.rsplit(" ", 1)[1], f(v["base"]["ppl"]), f(v["mean"]["ppl"]), f(v["fuse_u"]["ppl"]), f(v["fuse"]["ppl"]), f(v["base"]["acc"], 3), f(v["mean"]["acc"], 3), f(v["fuse"]["acc"], 3)]
        for k, v in sorted(T.items(), key=lambda kv: (short(kv[0].rsplit(" ", 1)[0]), kv[0]))]
table(["Model", "g", "ppl unconverted", "averaging", "unweighted fit", "fusion", "acc unconverted", "averaging", "fusion"], rows)

# ---- Table 4: rebalancing and rounding (round 30, part Q) ----
w("## Table 4. The fused model rounded to 4 and 3 bits per output channel, as computed and rebalanced (round 30)\n")
w("WikiText-2 perplexity on 8 windows of 2,048 tokens; the attention projections rounded, round to nearest; six models at half their KV heads.\n")
rows = []
for fpath in sorted(glob.glob("alignment_study/balance/Q_*.json"), key=lambda p: short(os.path.basename(p)[2:-5])):
    r = json.load(open(fpath)); m = short(os.path.basename(fpath)[2:-5])
    rows.append([m, f(r["base"]), f(r["fused"]), f(r["balanced"]), f(r["fused_w4"]), f(r["balanced_w4"]), f(r["fused_w3"]), f(r["balanced_w3"])])
table(["Model", "unconverted", "fused", "rebalanced", "4 bits, fused", "4 bits, rebalanced", "3 bits, fused", "3 bits, rebalanced"], rows)

# ---- Table 5: uptraining (round 30, part T) ----
w("## Table 5. Uptraining from three starts, final perplexity after 327,680 tokens (round 30)\n")
w("WikiText-2 perplexity on 40 windows of 1,024 tokens at the end of training, by learning rate.\n")
for fpath in sorted(glob.glob("alignment_study/balance/T_*.json")):
    r = json.load(open(fpath)); m = short(os.path.basename(fpath)[2:-5])
    lrs = r["lrs"]
    w(f"{m} (unconverted {r['base']:.1f}):\n")
    rows = []
    for start, name in (("mean", "averaging"), ("fuse", "fused as computed"), ("fuse_balanced", "fused, rebalanced")):
        rows.append([name] + [f(next(x["final"] for x in r["runs"] if x["start"] == start and x["lr"] == lr), 1) for lr in lrs])
    table(["start"] + [f"lr {lr:g}" for lr in lrs], rows)

# ---- Table 6: calibration metrics (rounds 31 and 32) ----
w("## Table 6. The fit's metric: every token alike, by received attention (round 31), by first-order sensitivity (round 32)\n")
w("WikiText-2 perplexity on 8 windows of 2,048 tokens at half the KV heads; Qwen2.5-7B's round 31 values are the post hoc correction (the registered run had an instrument defect).\n")
P31 = json.load(open("alignment_study/attnw_results.json"))["per_model"]
qc = json.load(open("alignment_study/attnw_correction/summary.json"))["qwen_corrected"]
P32 = json.load(open("alignment_study/attnw2_results.json"))["per_model"] if os.path.exists("alignment_study/attnw2_results.json") else {}
rows = []
for k in ["SmolLM2-1.7B", "Llama-3.2-1B", "Mistral-7B-v0.1", "OLMo-1B-hf", "h2o-danube-1.8b-base", "Mistral-7B-Instruct-v0.2", "Llama-2-7b-hf", "Qwen2.5-7B"]:
    a = qc if k == "Qwen2.5-7B" else P31.get(k)
    b = P32.get(k)
    base = (a or b or {}).get("base")
    rows.append([short(k), f(base), f((a or b or {}).get("plain")), f(a["attention"]) if a else "", f(b["sens"]) if b else "(pending)"])
table(["Model", "unconverted", "plain (S)", "received attention (round 31)", "sensitivity metrics (round 32)"], rows)
if not P32:
    w("Round 32's column is pending its evaluator.\n")

# ---- Table 7: merge criteria against merge damage (rounds 20 and 22) ----
NAMES = {"out_pert": "simulated output perturbation (consumer side)", "js": "attention-map Jensen-Shannon (Clark et al.)", "cos_map": "attention-map cosine",
         "corr_rows": "every-row map correlation (CHAI-style)", "corr_last": "last-row map correlation", "cos_k": "key cosine", "cos_v": "value cosine",
         "cos_kv": "key-value weight cosine", "cos_qk": "query-key weight cosine"}
for rnd, path, desc in (("20", "alignment_study/merge_bench_results.json", "Pythia-160m, Pythia-410m, GPT-2 medium (6,552 merges)"),
                        ("22", "alignment_study/merge_bench2_results.json", "Pythia-1.4B, OPT-1.3B, GPT-2 large (21,624 merges)")):
    if not os.path.exists(path):
        continue
    r = json.load(open(path))["pooled"]
    w(f"## Table 7{'a' if rnd == '20' else 'b'}. Within-layer Spearman correlation of each criterion with merge damage (KL), round {rnd}: {desc}\n")
    rows = [[NAMES.get(k, k), f(v["value"], 2), f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]", ", ".join(f(x, 2) for x in v.get("per_model", []))]
            for k, v in sorted(r.items(), key=lambda kv: -kv[1]["value"]) if isinstance(v, dict) and "value" in v]
    table(["Criterion", "Spearman", "95% interval", "per model"], rows)

# ---- Table 8: cross-layer KV sharing under the convention null (round 25) ----
if os.path.exists("alignment_study/xlayer_results.json"):
    S8 = json.load(open("alignment_study/xlayer_results.json"))["summary"]
    w("## Table 8. Adjacent layers' cached keys and values: per-token cosine, against the residual stream (round 25)\n")
    w("Median over layer pairs; the residual stream's similarity is what MiniCache's figure measures. Aligned: after a per-head alignment fitted on other documents; null: the same fit on shuffled tokens.\n")
    rows = [[short(k), f(v["k"], 3), f(v["v"], 3), f(v["hidden"], 2), f(v["normed_input"], 2), f(v["k_aligned"], 2), f(v["k_null"], 2), f(v["v_aligned"], 2), f(v["v_null"], 2)]
            for k, v in S8.items() if isinstance(v, dict) and "k" in v]
    table(["Model", "keys", "values", "residual stream", "normed input", "keys aligned", "keys null", "values aligned", "values null"], rows)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT}: {sum(1 for l in L if l.startswith('## '))} tables")
print("PAPER_TABLES_DONE")
