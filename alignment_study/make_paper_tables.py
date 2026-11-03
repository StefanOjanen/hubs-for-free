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
         "Mistral-7B-v0.1": "Mistral-7B-v0.1", "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini", "yaofu/llama-2-7b-80k": "LLaMA-2-7B-80K"}
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
w("## Table 3. Four models that already share key-value heads, and the multi-head OpenLLaMA-3B, converted to fewer (round 27)\n")
w("WikiText-2 perplexity and seven-task accuracy; g is the number of KV heads merged into one (TinyLlama 4 heads, Qwen2.5-1.5B 2, Mistral-7B-Instruct 8, Qwen2.5-7B 4, OpenLLaMA-3B 32).\n")
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
         "corr_rows": "every-row map correlation (CHAI-style)", "corr_last": "last-row map correlation", "k_act": "key cosine (cached activations)", "v_act": "value cosine (cached activations)",
         "kv_wcos": "key-value weight cosine", "qk_wcos": "query-key weight cosine"}
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

# ---- Table 9: the single-weighting ablation (development), beside round 26's rows ----
AB = sorted(f for f in glob.glob("alignment_study/fuse_ablation_dev/*.json") if not f.endswith("summary.json"))
if AB:
    w("## Table 9. The fit's weightings, one at a time (round 26's rows and the development ablation)\n")
    w("WikiText-2 perplexity on round 24's simulated-merge groups; the queries-only and outputs-only columns are a labelled development run (perplexity only), the others round 26.\n")
    rows = []
    for fpath in AB:
        r = json.load(open(fpath))["evals"]; m = short(os.path.basename(fpath)[:-5])
        for g in (2, 4):
            rows.append([m, g, f(r["base"]["wikitext2_ppl"]), f(r[f"g{g}/fuse_u/sim_align"]["wikitext2_ppl"]), f(r.get(f"g{g}/fuse_q/sim_align", {}).get("wikitext2_ppl")),
                         f(r.get(f"g{g}/fuse_o/sim_align", {}).get("wikitext2_ppl")), f(r[f"g{g}/fuse/sim_align"]["wikitext2_ppl"])])
    table(["Model", "g", "unconverted", "unweighted", "queries only", "outputs only", "both (fusion)"], rows)

# ---- Table 10: a quarter of the heads at 7B (development) ----
QD = sorted(f for f in glob.glob("alignment_study/quad_dev/*.json") if "_repair" not in f and "_fallback" not in f)
if QD:
    w("## Table 10. 7B models at a quarter of their key-value heads, no training (development)\n")
    w("WikiText-2 perplexity on the whole test set at 2,048 tokens; ZS-7 on 500 examples per task; the one-window check of Section 6 (largest ratio over layers 0 to 5).\n")
    lab = {"base": "unconverted", "fuse/paired": "fusion, 2x (pairs by the residual)", "fuse/quad": "fusion, 4x (pairs of pairs by the residual)",
           "fuse/neighbour": "fusion, 4x, neighbouring groups", "mean/neighbour": "averaging, 4x, neighbouring groups"}
    for fpath in QD:
        r = json.load(open(fpath)); m = short(os.path.basename(fpath)[:-5])
        w(f"{m} (summed four-head residual: matched {sum(r['cost_quad']):.2f}, neighbouring {sum(r['cost_neighbour4']):.2f}):\n")
        table(["Conversion", "ppl 2,048", "ZS-7", "one-window check"], [[lab.get(k, k), f(v["ppl_2048"]), f(v["zs7"], 3), f(v.get("check_max_ratio"))] for k, v in r["evals"].items()])

# ---- Table 10b: the hazard at a quarter of the heads on Llama-2-7B: trials, repair steps and the per-layer sweep ----
RP, FB = "alignment_study/quad_dev/Llama-2-7b-hf_repair.json", "alignment_study/quad_dev/Llama-2-7b-hf_fallback.json"
if os.path.exists(RP):
    r = json.load(open(RP))
    w("## Table 10b. Llama-2-7B at a quarter of the heads: which groups of layer 1 carry the event, and the repair (development)\n")
    w("Round 29's one-window check (largest ratio over layers 1 to 5) with layers 0 to 5 converted; each group of layer 1 left unmerged in turn; then the repair's steps.\n")
    table(["Group of layer 1 left unmerged", "check"], [[str(t["group_left_unmerged"]), f(t["max_ratio"])] for t in r["layer1_trials"]]
          + [[f"{r['worst_group']} merged alone", f(max(r["worst_group_alone_ratios"][1:]))]])
    table(["Repair step", "forbidden groups", "check"], [[i + 1, "; ".join(str(g) for g in st["forbidden"]), f(st["max_ratio"])] for i, st in enumerate(r["repair_steps"])]
          + ([["fallback: layer 1 at its pairs", "", f(max(r["fallback_layer1_pairs_ratios"][1:]))]] if "fallback_layer1_pairs_ratios" in r else []))
    for k, v in r.get("evals", {}).items():
        w(f"Evaluated ({k}): perplexity {v['ppl_2048']:.1f}, ZS-7 {v['zs7']:.3f}, check {v['check_max_ratio']:.1f}.\n")
if os.path.exists(FB):
    r = json.load(open(FB))
    w("## Table 10c. Llama-2-7B: early layers kept at pairs, the rest quartered (development)\n")
    w("The check with layers 0 to 5 converted, for the first k layers at the matched pairs and layers k to 5 at groups of four; the smallest k under 2.35 is evaluated in full.\n")
    table(["layers at pairs (k)", "check ratios, layers 0 to 5", "largest"], [[x["layers_at_pairs"], ", ".join(f"{v:.2f}" for v in x["ratios"]), f(x["max_ratio"])] for x in r["sweep"]])
    if r.get("evals"):
        for k, v in r["evals"].items():
            w(f"Chosen k = {r['chosen_layers_at_pairs']} (cache fraction {r['cache_fraction']:.3f}): perplexity {v['ppl_2048']:.2f}, ZS-7 {v['zs7']:.3f}, check {v['check_max_ratio']:.2f}.\n")
    elif r.get("chosen_layers_at_pairs") is None:
        w("No k up to 6 passed the check.\n")

# ---- Table 11: calibration size (development) ----
CD = sorted(glob.glob("alignment_study/calib_dev/*.json"))
if CD:
    w("## Table 11. How much calibration the fusion needs (development)\n")
    w("Perplexity of the fusion at half the key-value heads from the first n C4 documents of 2,048 tokens, with the pairs matched on the same moment and with the 128-document pairs; layers whose matching agrees with the 128-document one.\n")
    rows = []
    for fpath in CD:
        r = json.load(open(fpath)); m = short(os.path.basename(fpath)[:-5])
        for n, v in sorted(r["sizes"].items(), key=lambda kv: int(kv[0])):
            rows.append([m, n, v["tokens"], f(v["ppl_own_pairs"]), f(v["ppl_full_pairs"]), f"{v['layers_agreeing_with_full']} of {v['layers']}"])
    table(["Model", "documents", "tokens", "ppl, own pairs", "ppl, 128-document pairs", "layers agreeing"], rows)

# ---- Table 12: pairs by the residual in Qwen2.5-7B, and whole-test perplexities (development) ----
if os.path.exists("alignment_study/qwen_pairs_dev.json"):
    r = json.load(open("alignment_study/qwen_pairs_dev.json"))
    w("## Table 12a. Qwen2.5-7B with pairs matched by the residual against neighbouring pairs (development)\n")
    table(["Pairing", "ppl, 8 windows", "summed residual", "layers equal to neighbours"],
          [["neighbouring (rounds 27 to 32)", f(r["neighbour"]["ppl"]), f(sum(r["cost_neighbour"]), 3), ""],
           ["matched by the residual", f(r["matched"]["ppl"]), f(sum(r["cost_matched"]), 3), f"{r['layers_where_matched_is_neighbour']} of {r['layers']}"]])
FT = sorted(glob.glob("alignment_study/fulltest_dev/*.json"))
if FT:
    w("## Table 12b. The plain fusion on the whole WikiText-2 test set at 2,048 tokens (development), beside the registered 8-window values\n")
    rows = [[short(os.path.basename(fp)[:-5]), (lambda r: f"{r['windows']}")(json.load(open(fp)))] + [f(json.load(open(fp))[k]) for k in ("base_8_windows", "base_full_2048", "plain_8_windows", "plain_full_2048")] for fp in FT]
    table(["Model", "windows", "unconverted, 8", "unconverted, whole", "fusion, 8", "fusion, whole"], rows)

# ---- Table 5b: round 33, uptraining from three starts at 5M tokens (registered) ----
U33 = sorted(glob.glob("alignment_study/uptrain/*.json"))
if U33 and os.path.exists("alignment_study/uptrain33_results.json"):
    R33 = json.load(open("alignment_study/uptrain33_results.json"))
    w("## Table 5b. Uptraining from three starts at 5M tokens, learning rate 2e-5, half the key-value heads (round 33, registered at 9d73960)\n")
    w("Perplexity on the evaluation windows at the start and after 5M tokens, the zero-shot average after training, and the retained lead (the fused start's log-lead over averaging at the end as a fraction of its lead at the start; registered floor 0.1). " + f"U1 {'pass' if R33['U1']['pass'] else 'fail'}, U2 {'pass' if R33['U2']['pass'] else 'fail'}, U3 {'pass' if R33['U3']['pass'] else 'fail'}, U4 {'pass' if R33['U4']['pass'] else 'fail'}; {R33['passes']} of {R33['of']}.\n")
    rows = []
    for fp in U33:
        r = json.load(open(fp)); m = short(os.path.basename(fp)[:-5]); per = R33["per_model"].get(os.path.basename(fp)[:-5], {})
        for arm, lab in (("mean", "averaging"), ("fuse", "fusion, rebalanced"), ("fuse_sens", "fusion in the sensitivity metric, rebalanced")):
            if arm in r["arms"]:
                c = r["arms"][arm]["curve"]; zs = r["arms"][arm].get("zeroshot_after", {}).get("average")
                rows.append([m, lab, f(c[0]["ppl"]), f(c[-1]["ppl"]), f(zs, 3) if zs is not None else "", f(per["retained"], 3) if arm == "fuse" and per.get("retained") is not None else ""])
    table(["Model", "start", "ppl at 0", "ppl at 5M", "zero-shot after", "retained lead"], rows)

# ---- Table 10d: round 34, the check flagging and choosing the early layers' ratio at a quarter of the heads (registered) ----
Q34 = "alignment_study/quad_check_results.json"
if os.path.exists(Q34) and not json.load(open(Q34)).get("missing"):
    R34 = json.load(open(Q34))
    w("## Table 10d. A quarter of the heads with layers 0 and 1 each unmerged, at pairs or at groups of four (round 34, registered at c3cc48c)\n")
    w("Perplexity on 8 windows relative to the member with layers 0 and 1 unmerged (every other layer at the residual's groups of four), and the one-window check's largest ratio over layers 0 to 5 (threshold 2.35; flagged members marked). Llama-2-7B set the threshold and the fixed rule and is reported, not scored. " + f"R1 {'pass' if R34['R1']['pass'] else 'fail'}, R2 {'pass' if R34['R2']['pass'] else 'fail'}, R3 {'pass' if R34['R3']['pass'] else ('untestable' if not R34['R3'].get('testable') else 'fail')}; {R34['passes']} of {R34['of']}.\n")
    LAB = {"open_llama_3b_v2": "OpenLLaMA-3B", "Mistral-7B-v0.1": "Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct", "Qwen2.5-7B": "Qwen2.5-7B", "Llama-2-7b-hf": "Llama-2-7B (development)"}
    order = [m for m in ("open_llama_3b_v2", "Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.2", "Qwen2.5-7B", "Llama-2-7b-hf") if m in R34["per_model"]]
    settings = [(l0, l1) for l0 in ("none", "pairs", "quads") for l1 in ("none", "pairs", "quads")]
    rows = []
    for m in order:
        a = R34["per_model"][m]; byk = {(x["layer0"], x["layer1"]): x for x in a["members"]}
        rows.append([LAB[m], f(a["reference_ppl"])] + [f"{byk[k]['relative']:.2f} ({byk[k]['max_ratio']:.2f}{', flag' if byk[k]['flag'] else ''})" for k in settings] + [f"{a['chosen'][0]} / {a['chosen'][1]}" if a["chosen"] else ""])
    table(["Model", "reference ppl"] + [f"{l0} / {l1}" for l0, l1 in settings] + ["check's choice (layer 0 / layer 1)"], rows)

# ---- Table 13: the fusion's residual against merge damage at 7B scale (development) ----
RD = [fp for fp in sorted(glob.glob("alignment_study/residual_damage/*.json")) if not fp.endswith("summary.json")]
if RD:
    R13 = {(lambda r: r["model"])(json.load(open(fp))): json.load(open(fp))["spearman"] for fp in RD}
    w("## Table 13. The fusion's own residual against round 29's merge damage (development)\n")
    w("Spearman correlation, within each model, between a forced-pair conversion's damage (perplexity over the reference conversion, round 29's 4 windows) and its matching's cost increase, its forced pair's cost, and the one-window check; then the damage's spread by layer (layer 0 against layers 1 to 4) and the cost increase's correlation within layer 0.\n")
    rows13 = []
    for m, v in R13.items():
        mem = json.load(open(f"alignment_study/residual_damage/{m.split('/')[-1]}.json"))["members"]
        d0 = [x["damage"] for x in mem if x["layer"] == 0]; d1 = [x["damage"] for x in mem if x["layer"] > 0]
        sd = lambda a: (sum((y - sum(a) / len(a)) ** 2 for y in a) / len(a)) ** 0.5 if len(a) > 1 else float("nan")
        rows13.append([short(m), v["members"], f(v["cost_increase"]), f(v["forced_pair_cost"]), f(v["one_window_check"]) if v.get("one_window_check") is not None else "",
                       f"{min(d0):.2f} to {max(d0):.2f}" if d0 else "", f"{min(d1):.2f} to {max(d1):.2f} (sd {sd(d1):.3f})" if d1 else "", f(v["per_layer_cost_increase"]["0"]) if "0" in v.get("per_layer_cost_increase", {}) else ""])
    table(["Model", "members", "cost increase", "forced pair's cost", "one-window check", "damage, layer 0", "damage, layers 1 to 4", "cost increase, layer 0"], rows13)

# ---- Table 14: retrieval of a planted fact (development) and the benchmark ----
ND = sorted(glob.glob("alignment_study/needle_dev/*.json"))
if ND:
    w("## Table 14. Retrieval of a planted fact from a 2,048-token haystack, three depths, five facts (development)\n")
    rows = []
    for fp in ND:
        r = json.load(open(fp)); m = short(os.path.basename(fp)[:-5])
        for arm, v in r["arms"].items():
            rows.append([m, arm, f"{v['hits']} of {v['of']}"] + [str(v["by_depth"][str(d)]) for d in r["depths"]])
    table(["Model", "arm", "retrieved"] + [f"depth {d}%" for d in json.load(open(ND[0]))["depths"]], rows)
RP = sorted(glob.glob("alignment_study/retrieval_protect_dev/*.json"))
if RP:
    w("## Table 14b. Retrieval protection: the retrieval heads' key-value heads kept unmerged, against random and residual-chosen protection (development). protect_random_24 (the first file) draws from all heads; the seed-1 arms draw from heads below 0.1 but their odd-layer leftovers were unrestricted; the seed-2 arms are the clean controls\n")
    rows = []
    for fp in RP:
        r = json.load(open(fp)); m = short(os.path.basename(fp)[:-5])
        for arm, v in r["arms"].items():
            rows.append([m, arm, v["n_protected"], f"{v['cache_fraction']:.3f}", f(v["ppl_8_windows"]), f"{v['needle']['hits']} of {v['needle']['of']}"] + [str(v["needle"]["by_depth"][str(d)]) for d in r["needle"]["depths"]])
    table(["Model", "arm", "protected", "cache fraction", "ppl, 8 windows", "retrieved"] + [f"depth {d}%" for d in json.load(open(RP[0]))["needle"]["depths"]], rows)
KB = sorted(glob.glob("alignment_study/kvfuse_bench/*.json"))
if KB:
    w("## Table 15. The exported checkpoint's cache and decode speed (one laptop, MPS)\n")
    rows = []
    for fp in KB:
        r = json.load(open(fp)); m = short(os.path.basename(fp)[:-5])
        for arm, v in r["arms"].items():
            rows.append([m, arm, v["kv_heads"], f"{v['cache_bytes_at_prefill_plus_decode'] / 2**20:.0f} MiB", f(v["prefill_seconds"], 2), f(v["decode_tokens_per_second"], 1)])
        rows.append([m, "ratio", "", f(r["cache_ratio"], 3), "", f"{r['decode_speedup']:.2f}x"])
    table(["Model", "checkpoint", "KV heads", f"cache at {json.load(open(KB[0]))['prefill'] + json.load(open(KB[0]))['decode']} tokens", "prefill s", "decode tok/s"], rows)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT}: {sum(1 for l in L if l.startswith('## '))} tables")
print("PAPER_TABLES_DONE")
