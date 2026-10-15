# Evaluation of alignment_study/PREREGISTRATION21.md, committed before any
# confirmatory result exists.
#   python alignment_study/eval_sink_arch.py [directory]
import json
import sys

D = (sys.argv[1] if len(sys.argv) > 1 else "alignment_study/sink_arch").rstrip("/")
E = {m: json.load(open(f"{D}/E_{m}.json")) for m in ("gpt2-medium", "pythia-410m", "opt-6.7b", "bert-base-cased", "opus-mt-en-de")}
W = {m: json.load(open(f"{D}/W_{m}.json")) for m in ("TinyLlama_v1.1", "Qwen2.5-1.5B", "pythia-410m")}
C = {
    "G1": {"what": "rotary decoder (Pythia-410m): bias share >= 0.5", "value": E["pythia-410m"]["bias_share"], "pass_": E["pythia-410m"]["bias_share"] >= 0.5},
    "G2": {"what": "learned-absolute-position decoders (GPT-2 medium, OPT-6.7B): bias share < 0.5", "value": [E["gpt2-medium"]["bias_share"], E["opt-6.7b"]["bias_share"]],
           "pass_": E["gpt2-medium"]["bias_share"] < 0.5 and E["opt-6.7b"]["bias_share"] < 0.5},
    "G3": {"what": "decoders: removing the sink costs at least 5 times an ordinary token", "value": [E[m]["zero_over_rand"] for m in ("gpt2-medium", "pythia-410m", "opt-6.7b")],
           "pass_": all(E[m]["zero_over_rand"] >= 5 for m in ("gpt2-medium", "pythia-410m", "opt-6.7b"))},
    "G4": {"what": "encoders (BERT [SEP], OPUS-MT end-of-sentence): removing the sink costs at most 2 times an ordinary token",
           "value": [E["bert-base-cased"]["zero_over_rand"], E["opus-mt-en-de"]["zero_over_rand"]],
           "pass_": E["bert-base-cased"]["zero_over_rand"] <= 2 and E["opus-mt-en-de"]["zero_over_rand"] <= 2},
    "G5": {"what": "StreamingLLM reproduces: keeping the first four tokens lowers NLL against the window alone in every model",
           "value": {m: [r["window"], r["stream"]] for m, r in W.items()}, "pass_": all(r["stream"] < r["window"] for r in W.values())},
    "G6": {"what": "a constant virtual sink recovers at least half of StreamingLLM's improvement in every model", "value": {m: r["recovery"] for m, r in W.items()},
           "pass_": all(r["recovery"] is not None and r["recovery"] >= 0.5 for r in W.values())},
}
checks = {**{f"E_{m}": [r["equivalence_max_logit_diff"], r["restore_check_kl"]] for m, r in E.items()}, **{f"W_{m}": [r["equivalence_max_logit_diff"]] for m, r in W.items()}}
res = {"clauses": C, "passes": int(sum(v["pass_"] for v in C.values())), "checks": checks,
       "reported": {"E": {m: {k: r[k] for k in ("bias_share", "zero_over_rand")} | {mode: r[mode]["kl"] for mode in ("zero", "mean", "rand")} for m, r in E.items()},
                    "W": {m: {k: r[k] for k in ("full", "window", "stream", "stream1", "bias", "recovery", "virtual_weight_mean")} for m, r in W.items()}}}
json.dump(res, open(D + "_results.json", "w"), indent=1)
for k, v in C.items():
    print(k, "PASS" if v["pass_"] else "FAIL", v["what"], json.dumps(v["value"])[:200])
print("passes", res["passes"], "of", len(C)); print("checks", checks)
